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
This file tests the FMScore class.
*/

#include <cmath>

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/score/score_function.h"
#include "src/score/fm_score.h"

namespace xLearn {

TEST(FMScoreTest, calc_score) {
  for (index_t k = 1; k < 100; ++k) {
    // Init hyper_param
    HyperParam param;
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "squared";
    param.score_func = "fm";
    param.num_feature = 3;
    param.num_K = k;
    param.num_field = 3;
    // Init SparseRow
    SparseRow row(param.num_feature);
    for (index_t i = 0; i < param.num_feature; ++i) {
      row[i].feat_id = i;
      row[i].feat_val = 2.0;
    }
    // Init model
    Model model;
    model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K, 2);
    real_t* w = model.GetParameter_w();
    index_t num_w = model.GetNumParameter_w();
    for (index_t i = 0; i < num_w; ++i) {
      w[i] = 1.0;
    }
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for(index_t d = 0; d < model.GetNumK(); d++, v++)
        *v = 1.0;
      for(index_t d = model.GetNumK(); d < k_aligned; d++, v++)
        *v = 0;
      for(index_t d = k_aligned; d < 2*k_aligned; d++, v++)
        *v = 1.0;
    }
    model.GetParameter_b()[0] = 0.0;
    FMScore score;
    for (size_t i = 0; i < 10; ++i) {
      real_t val = score.CalcScore(&row, model);
      EXPECT_FLOAT_EQ(val, 6+k*4*3);
    }
  }
}

TEST(FMScoreTest, calc_score_overflow) {
  for (index_t k = 1; k < 100; ++k) {
    // Init hyper_param
    HyperParam param;
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "squared";
    param.score_func = "fm";
    param.num_feature = 3;
    param.num_K = k;
    param.num_field = 3;
    // Init SparseRow
    SparseRow row(param.num_feature*2);
    for (index_t i = 0; i < param.num_feature*2; ++i) {
      row[i].feat_id = i;
      row[i].feat_val = 2.0;
    }
    // Init model
    Model model;
    model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K, 2);
    real_t* w = model.GetParameter_w();
    index_t num_w = model.GetNumParameter_w();
    for (index_t i = 0; i < num_w; ++i) {
      w[i] = 1.0;
    }
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for(index_t d = 0; d < model.GetNumK(); d++, v++)
        *v = 1.0;
      for(index_t d = model.GetNumK(); d < k_aligned; d++, v++)
        *v = 0;
      for(index_t d = k_aligned; d < 2*k_aligned; d++, v++)
        *v = 1.0;
    }
    model.GetParameter_b()[0] = 0.0;
    FMScore score;
    for (size_t i = 0; i < 10; ++i) {
      real_t val = score.CalcScore(&row, model);
      EXPECT_FLOAT_EQ(val, 6+k*4*3);
    }
  }
}

// The ftrl latent update drives a weight to zero exactly when |z| falls inside
// the l1 band. These two cover either side of that choice, which is the whole
// of the update's branching.
TEST(FMScoreTest, calc_grad_ftrl_inside_l1_band) {
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    model.Initialize("fm", "squared", 3, 3, k, 3);
    SparseRow row(3);
    for (index_t i = 0; i < 3; ++i) {
      row[i].feat_id = i;
      row[i].feat_val = 2.0;
    }
    FMScore score;
    std::string opt_type("ftrl");
    // An l1 no |z| can exceed, so every coordinate lands inside the band.
    score.Initialize(0.1, 0, 0.3, 1.0, 1e10, 0, opt_type);
    score.CalcGrad(&row, model, 1.0);
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t d = 0; d < k_aligned; ++d) {
        EXPECT_FLOAT_EQ(v[j*k_aligned*3 + d], 0.0);
      }
    }
  }
}

TEST(FMScoreTest, calc_grad_ftrl_outside_l1_band) {
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    model.Initialize("fm", "squared", 3, 3, k, 3);
    SparseRow row(3);
    for (index_t i = 0; i < 3; ++i) {
      row[i].feat_id = i;
      row[i].feat_val = 2.0;
    }
    FMScore score;
    std::string opt_type("ftrl");
    // No band at all, so every coordinate takes the computed value.
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
    score.CalcGrad(&row, model, 1.0);
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t d = 0; d < k_aligned; ++d) {
        EXPECT_TRUE(std::isfinite(v[j*k_aligned*3 + d]));
        EXPECT_NE(v[j*k_aligned*3 + d], 0.0);
      }
    }
  }
}

} // namespace xLearn
