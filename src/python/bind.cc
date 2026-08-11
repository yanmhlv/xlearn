//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/*
This file is the implementation of the Python API. It backs the xlearn._core
extension module, which the xlearn Python package wraps.
*/

#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "src/base/format_print.h"
#include "src/base/stringprintf.h"
#include "src/base/timer.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/solver/solver.h"

namespace nb = nanobind;

namespace {

using DenseMatrix = nb::ndarray<const real_t, nb::ndim<2>,
                                nb::c_contig, nb::device::cpu>;
using RealVector = nb::ndarray<const real_t, nb::ndim<1>,
                               nb::c_contig, nb::device::cpu>;
using IndexVector = nb::ndarray<const index_t, nb::ndim<1>,
                                nb::c_contig, nb::device::cpu>;
using Predictions = nb::ndarray<nb::numpy, real_t, nb::ndim<1>>;

//------------------------------------------------------------------------------
// A dataset owned by Python. xLearn::DMatrix leaves the rows it allocated
// behind on destruction -- its callers inside xLearn all call Reset() by hand
// -- so a matrix whose lifetime Python controls is held through this wrapper.
//------------------------------------------------------------------------------
class Data {
 public:
  Data(DenseMatrix data,
       std::optional<RealVector> label,
       std::optional<IndexVector> field_map) {
    const size_t num_row = data.shape(0);
    const size_t num_col = data.shape(1);
    AddRows(num_row, label);
    for (size_t i = 0; i < num_row; ++i) {
      real_t norm = 0.0;
      for (size_t j = 0; j < num_col; ++j) {
        const real_t value = data(i, j);
        norm += value * value;
        AddNode(i, j, value, field_map);
      }
      FinishRow(i, norm);
    }
  }

  // The three arrays scipy.sparse.csr_matrix stores a matrix in.
  Data(RealVector values,
       IndexVector indices,
       IndexVector indptr,
       std::optional<RealVector> label,
       std::optional<IndexVector> field_map) {
    // Before anything is allocated: a constructor that throws leaves ~Data()
    // unrun, and ~DMatrix() does not release the rows.
    if (values.size() != indices.size() || indptr.size() == 0 ||
        indptr(indptr.size() - 1) != values.size()) {
      throw std::runtime_error("Malformed CSR arrays!");
    }
    const size_t num_row = indptr.size() - 1;
    AddRows(num_row, label);
    for (size_t i = 0; i < num_row; ++i) {
      real_t norm = 0.0;
      for (index_t k = indptr(i); k < indptr(i + 1); ++k) {
        const real_t value = values(k);
        norm += value * value;
        AddNode(i, indices(k), value, field_map);
      }
      FinishRow(i, norm);
    }
  }

  ~Data() { matrix_.Reset(); }

  xLearn::DMatrix* matrix() { return &matrix_; }

 private:
  void AddRows(size_t num_row, const std::optional<RealVector>& label) {
    matrix_.has_label = label.has_value();
    for (size_t i = 0; i < num_row; ++i) {
      matrix_.AddRow();
      if (label) {
        matrix_.Y[i] = (*label)(i);
      }
    }
  }

  // A zero feature contributes nothing to any of the score functions, so
  // storing one costs a Node and buys nothing.
  void AddNode(size_t row, index_t column, real_t value,
               const std::optional<IndexVector>& field_map) {
    if (value == 0.0) {
      return;
    }
    matrix_.AddNode(row, column, value, field_map ? (*field_map)(column) : 0);
  }

  // A row of nothing but zeros keeps the 1.0 norm AddRow() gave it.
  void FinishRow(size_t i, real_t norm) {
    if (norm != 0.0) {
      matrix_.norm[i] = 1.0f / norm;
    }
  }

  xLearn::DMatrix matrix_;
};

//------------------------------------------------------------------------------
// One model: the hyper-parameters the caller builds up and the solver they
// drive. This is the entry class of the Python API, mirroring the way the
// command line tool pairs a HyperParam with a Solver.
//------------------------------------------------------------------------------
class Model {
 public:
  explicit Model(const std::string& score_func) {
    hyper_param_.score_func = score_func;
  }

  xLearn::HyperParam& params() { return hyper_param_; }
  const xLearn::HyperParam& params() const { return hyper_param_; }

  void Show() const {
    printf("Info: \n Model: %s\n Loss: %s\n",
           hyper_param_.score_func.c_str(),
           hyper_param_.loss_func.c_str());
  }

  void SetTask(const std::string& task) {
    if (task == "binary") {
      hyper_param_.loss_func = "cross-entropy";
    } else if (task == "reg") {
      hyper_param_.loss_func = "squared";
    } else {
      hyper_param_.loss_func = "unknow";
    }
  }

  const std::string& GetTask() const { return hyper_param_.loss_func; }

  void SetTrainFile(const std::string& path) {
    hyper_param_.train_set_file = path;
    hyper_param_.from_file = true;
  }

  void SetTrainData(Data& data) {
    if (!data.matrix()->has_label) {
      throw std::runtime_error("Train set must have label!");
    }
    hyper_param_.train_dataset = data.matrix();
    hyper_param_.from_file = false;
  }

  void SetTestFile(const std::string& path) {
    hyper_param_.test_set_file = path;
    hyper_param_.from_file = true;
  }

  void SetTestData(Data& data) {
    hyper_param_.test_dataset = data.matrix();
    hyper_param_.from_file = false;
  }

  void SetValidateFile(const std::string& path) {
    if (!hyper_param_.from_file) {
      throw std::runtime_error(
        "Input for validation type Must equal to train's!");
    }
    hyper_param_.validate_set_file = path;
  }

  void SetValidateData(Data& data) {
    hyper_param_.valid_dataset = data.matrix();
  }

  void SetPreModel(const std::string& path) {
    hyper_param_.pre_model_file = path;
  }

  void SetTXTModel(const std::string& path) {
    hyper_param_.txt_model_file = path;
  }

  void Fit(const std::string& model_path) {
    hyper_param_.model_file = model_path;
    hyper_param_.is_train = true;
    RunSolver();
  }

  void CrossValidate() {
    hyper_param_.cross_validation = true;
    hyper_param_.is_train = true;
    RunSolver();
    hyper_param_.cross_validation = false;
  }

  std::vector<real_t> Predict(const std::string& model_path) {
    hyper_param_.model_file = model_path;
    hyper_param_.res_out = false;
    hyper_param_.is_train = false;
    return RunSolver();
  }

  void PredictToFile(const std::string& model_path,
                     const std::string& out_path) {
    hyper_param_.model_file = model_path;
    hyper_param_.output_file = out_path;
    hyper_param_.res_out = true;
    hyper_param_.is_train = false;
    RunSolver();
  }

 private:
  // Solver::SetPredict() has to land after Initialize(), which overwrites the
  // solver's own copy of the hyper-parameters.
  std::vector<real_t> RunSolver() {
    Timer timer;
    timer.tic();
    solver_.Initialize(hyper_param_);
    if (!hyper_param_.is_train) {
      solver_.SetPredict();
    }
    solver_.StartWork();
    std::vector<real_t> result = solver_.GetResult();
    solver_.Clear();
    Color::print_info(
      StringPrintf("Total time cost: %.2f (sec)", timer.toc()), true);
    return result;
  }

  xLearn::HyperParam hyper_param_;
  xLearn::Solver solver_;
};

// Hands the predictions to NumPy without a copy, as an array that owns the
// vector they live in.
Predictions ToNumpy(std::vector<real_t>&& result) {
  auto owned = std::make_unique<std::vector<real_t>>(std::move(result));
  const size_t size = owned->size();
  real_t* data = owned->data();
  nb::capsule owner(owned.release(), [](void *p) noexcept {
    delete static_cast<std::vector<real_t>*>(p);
  });
  return Predictions(data, {size}, owner);
}

// Binds a hyper-parameter under the name the Python API knows it by.
template <typename T>
void BindParam(nb::class_<Model>& model, const char* name,
               T xLearn::HyperParam::*field) {
  model.def_prop_rw(
    name,
    [field](const Model& self) { return self.params().*field; },
    [field](Model& self, const T& value) { self.params().*field = value; });
}

void Hello() {
  std::string logo =
"----------------------------------------------------------------------------------------------\n"
                    "           _\n"
                    "          | |\n"
                    "     __  _| |     ___  __ _ _ __ _ __\n"
                    "     \\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
                    "      >  <| |___|  __/ (_| | |  | | | |\n"
                    "     /_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
                    "        xLearn   -- " XLEARN_VERSION " Version --\n"
"----------------------------------------------------------------------------------------------\n"
"\n";
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier def(Color::FG_DEFAULT);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << green << bold << logo << def << reset;
}

}  // namespace

NB_MODULE(_core, m) {
  // Everything xLearn rejects is reported as a std::runtime_error, and the
  // Python API has always presented all of them under one exception type.
  nb::object error = nb::steal(PyErr_NewExceptionWithDoc(
    "xlearn._core.XLearnError", "Error thrown by xlearn trainer",
    nullptr, nullptr));
  m.attr("XLearnError") = error;
  nb::register_exception_translator(
    [](const std::exception_ptr& caught, void *type) {
      try {
        std::rethrow_exception(caught);
      } catch (const std::runtime_error& e) {
        PyErr_SetString(static_cast<PyObject*>(type), e.what());
      }
    }, error.ptr());

  nb::class_<Data>(m, "DMatrix")
    .def(nb::init<DenseMatrix, std::optional<RealVector>,
                  std::optional<IndexVector>>(),
         nb::arg("data"), nb::arg("label").none(), nb::arg("field_map").none())
    .def(nb::init<RealVector, IndexVector, IndexVector,
                  std::optional<RealVector>, std::optional<IndexVector>>(),
         nb::arg("values"), nb::arg("indices"), nb::arg("indptr"),
         nb::arg("label").none(), nb::arg("field_map").none());

  nb::class_<Model> model(m, "Model");
  model
    .def(nb::init<const std::string&>(), nb::arg("score_func"))
    .def("show", &Model::Show)
    .def("set_train_file", &Model::SetTrainFile, nb::arg("path"))
    .def("set_train_data", &Model::SetTrainData, nb::arg("data"),
         nb::keep_alive<1, 2>())
    .def("set_test_file", &Model::SetTestFile, nb::arg("path"))
    .def("set_test_data", &Model::SetTestData, nb::arg("data"),
         nb::keep_alive<1, 2>())
    .def("set_validate_file", &Model::SetValidateFile, nb::arg("path"))
    .def("set_validate_data", &Model::SetValidateData, nb::arg("data"),
         nb::keep_alive<1, 2>())
    .def("set_pre_model", &Model::SetPreModel, nb::arg("path"))
    .def("set_txt_model", &Model::SetTXTModel, nb::arg("path"))
    .def("fit", &Model::Fit, nb::arg("model_path"),
         nb::call_guard<nb::gil_scoped_release>())
    .def("cv", &Model::CrossValidate,
         nb::call_guard<nb::gil_scoped_release>())
    .def("predict",
         [](Model& self, const std::string& model_path) {
           std::vector<real_t> result;
           {
             nb::gil_scoped_release release;
             result = self.Predict(model_path);
           }
           return ToNumpy(std::move(result));
         },
         nb::arg("model_path"))
    .def("predict_to_file", &Model::PredictToFile, nb::arg("model_path"),
         nb::arg("out_path"), nb::call_guard<nb::gil_scoped_release>())
    .def_prop_rw("task", &Model::GetTask, &Model::SetTask);

  BindParam(model, "metric", &xLearn::HyperParam::metric);
  BindParam(model, "opt", &xLearn::HyperParam::opt_type);
  BindParam(model, "log", &xLearn::HyperParam::log_file);
  BindParam(model, "lr", &xLearn::HyperParam::learning_rate);
  BindParam(model, "k", &xLearn::HyperParam::num_K);
  BindParam(model, "lambda", &xLearn::HyperParam::regu_lambda);
  BindParam(model, "init", &xLearn::HyperParam::model_scale);
  BindParam(model, "epoch", &xLearn::HyperParam::num_epoch);
  BindParam(model, "fold", &xLearn::HyperParam::num_folds);
  BindParam(model, "alpha", &xLearn::HyperParam::alpha);
  BindParam(model, "beta", &xLearn::HyperParam::beta);
  BindParam(model, "lambda_1", &xLearn::HyperParam::lambda_1);
  BindParam(model, "lambda_2", &xLearn::HyperParam::lambda_2);
  BindParam(model, "nthread", &xLearn::HyperParam::thread_number);
  BindParam(model, "block_size", &xLearn::HyperParam::block_size);
  BindParam(model, "stop_window", &xLearn::HyperParam::stop_window);
  BindParam(model, "seed", &xLearn::HyperParam::seed);
  BindParam(model, "quiet", &xLearn::HyperParam::quiet);
  BindParam(model, "on_disk", &xLearn::HyperParam::on_disk);
  BindParam(model, "bin_out", &xLearn::HyperParam::bin_out);
  BindParam(model, "norm", &xLearn::HyperParam::norm);
  BindParam(model, "lock_free", &xLearn::HyperParam::lock_free);
  BindParam(model, "early_stop", &xLearn::HyperParam::early_stop);
  BindParam(model, "sign", &xLearn::HyperParam::sign);
  BindParam(model, "sigmoid", &xLearn::HyperParam::sigmoid);

  m.def("hello", &Hello, "Say hello to user");
}
