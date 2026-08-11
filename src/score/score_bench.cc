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
This file benchmarks the SIMD kernels of the FMScore and FFMScore classes.
*/

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/score/fm_score.h"
#include "src/score/ffm_score.h"

namespace xLearn {
namespace {

const real_t kLearningRate = 0.2;
const real_t kReguLambda = 0.00002;
const real_t kAlpha = 0.3;
const real_t kBeta = 1.0;
const real_t kLambda1 = 0.00001;
const real_t kLambda2 = 0.00002;
const real_t kModelScale = 0.66;
// A plausible mid-training partial gradient.
const real_t kPartialGrad = 0.15;

struct Config {
  const char* kind;
  const char* opt_type;
  index_t num_K;
  index_t num_feat;
  index_t num_field;
  index_t num_nnz;
  // A power of two, so the row cursor can wrap with a mask.
  index_t num_rows;
  const char* suffix;
};

index_t AuxiliarySize(const std::string& opt_type) {
  if (opt_type == "sgd") return 1;
  if (opt_type == "adagrad") return 2;
  return 3;
}

//------------------------------------------------------------------------------
// A model and the pool of rows fed through it. Every configuration draws from
// many rows rather than re-reading one out of L1, which for the widest one
// means tens of megabytes; building that per repetition would dominate the
// measurement, so each configuration is built once and shared.
//------------------------------------------------------------------------------
struct Workload {
  Model model;
  // The rows live in a DMatrix rather than one buffer each, so the walk here
  // has the same locality as the training loop the numbers stand in for.
  DMatrix matrix;
  index_t row_mask;
};

Workload& GetWorkload(const Config& cfg) {
  static std::map<std::string, std::unique_ptr<Workload>> cache;
  std::string key = std::string(cfg.kind) + "/" + cfg.opt_type + "/" +
                    std::to_string(cfg.num_K) + "/" +
                    std::to_string(cfg.num_feat) + "/" +
                    std::to_string(cfg.num_rows);
  std::unique_ptr<Workload>& slot = cache[key];
  if (slot != nullptr) {
    return *slot;
  }
  slot.reset(new Workload());
  slot->model.Initialize(cfg.kind, "squared", cfg.num_feat, cfg.num_field,
                         cfg.num_K, AuxiliarySize(cfg.opt_type), kModelScale);
  slot->row_mask = cfg.num_rows - 1;
  slot->matrix.Reserve(cfg.num_rows, cfg.num_rows * cfg.num_nnz);
  // Scatter the feature ids with a linear congruential sequence so the access
  // pattern spreads over the feature space instead of running sequentially.
  index_t seed = 1;
  for (index_t r = 0; r < cfg.num_rows; ++r) {
    slot->matrix.AddRow();
    for (index_t i = 0; i < cfg.num_nnz; ++i) {
      seed = seed * 1103515245 + 12345;
      slot->matrix.AddNode(r, (seed >> 8) % cfg.num_feat, 1.0,
                           i % cfg.num_field);
    }
  }
  return *slot;
}

// Score forbids copying, so it is initialized in place rather than returned.
template <typename ScoreType>
void InitScore(ScoreType* score, const Config& cfg) {
  std::string opt = cfg.opt_type;
  score->Initialize(kLearningRate, kReguLambda, kAlpha,
                    kBeta, kLambda1, kLambda2, opt);
}

template <typename ScoreType>
void RunScore(benchmark::State& state, Config cfg) {
  Workload& workload = GetWorkload(cfg);
  ScoreType score;
  InitScore(&score, cfg);
  index_t cursor = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        score.CalcScore(workload.matrix.Row(cursor & workload.row_mask),
                        workload.model, 1.0));
    ++cursor;
  }
}

template <typename ScoreType>
void RunGrad(benchmark::State& state, Config cfg) {
  Workload& workload = GetWorkload(cfg);
  ScoreType score;
  InitScore(&score, cfg);
  // Untimed, once per repetition. Without it the parameters drift far from
  // their initial scale over millions of updates and the timings describe a
  // model no training run would ever produce.
  workload.model.Reset();
  index_t cursor = 0;
  for (auto _ : state) {
    score.CalcGrad(workload.matrix.Row(cursor & workload.row_mask),
                   workload.model, kPartialGrad, 1.0);
    ++cursor;
  }
}

void Register(const Config& cfg) {
  std::string stem = std::string(cfg.kind) + "/" + cfg.opt_type;
  std::string tail = "/k=" + std::to_string(cfg.num_K) + cfg.suffix;
  if (std::string(cfg.kind) == "fm") {
    benchmark::RegisterBenchmark(stem + "/score" + tail,
        [cfg](benchmark::State& s) { RunScore<FMScore>(s, cfg); });
    benchmark::RegisterBenchmark(stem + "/grad" + tail,
        [cfg](benchmark::State& s) { RunGrad<FMScore>(s, cfg); });
  } else {
    benchmark::RegisterBenchmark(stem + "/score" + tail,
        [cfg](benchmark::State& s) { RunScore<FFMScore>(s, cfg); });
    benchmark::RegisterBenchmark(stem + "/grad" + tail,
        [cfg](benchmark::State& s) { RunGrad<FFMScore>(s, cfg); });
  }
}

// FM keeps one field and a wide latent vector; FFM a narrow one per field,
// which makes it quadratic in the row length. The oom-cache entry walks a
// feature space far larger than the cache, which is the regime a real
// training run sits in and where memory latency, not SIMD, dominates.
const Config kConfigs[] = {
  {"fm",  "sgd",      4,   10000,  1, 30,  1024, ""},
  {"fm",  "sgd",     16,   10000,  1, 30,  1024, ""},
  {"fm",  "sgd",     64,   10000,  1, 30,  1024, ""},
  {"fm",  "adagrad", 16,   10000,  1, 30,  1024, ""},
  {"fm",  "ftrl",    16,   10000,  1, 30,  1024, ""},
  {"fm",  "sgd",     16, 1000000,  1, 30, 16384, "/oom-cache"},
  {"ffm", "sgd",      4,    2000, 20, 20,  1024, ""},
  {"ffm", "sgd",      8,    2000, 20, 20,  1024, ""},
  {"ffm", "sgd",     16,    2000, 20, 20,  1024, ""},
  {"ffm", "adagrad",  4,    2000, 20, 20,  1024, ""},
  {"ffm", "adagrad", 16,    2000, 20, 20,  1024, ""},
  {"ffm", "ftrl",     4,    2000, 20, 20,  1024, ""},
  {"ffm", "ftrl",    16,    2000, 20, 20,  1024, ""},
  {"ffm", "sgd",      4,  200000, 20, 20,  4096, "/oom-cache"},
};

} // namespace
} // namespace xLearn

int main(int argc, char** argv) {
  for (const xLearn::Config& cfg : xLearn::kConfigs) {
    xLearn::Register(cfg);
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
