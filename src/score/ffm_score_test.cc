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
This file tests the FFMScore class.
*/

#include <cmath>

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/score/score_function.h"
#include "src/score/ffm_score.h"

namespace xLearn {

TEST(FFMScore_Test, calc_score) {
  for (index_t k = 1; k < 100; ++k) {
    // Init hyper_param
    HyperParam param;
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "squared";
    param.score_func = "ffm";
    param.num_feature = 3;
    param.num_K = k;
    param.num_field = 3;
    // Init the row
    RowBuffer buf;
    const index_t kRowLen = param.num_feature;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0, i);
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
    index_t aux_size = model.GetAuxiliarySize();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        for (index_t d = 0; d < k_aligned; d++, v++) {
          *v = (d < model.GetNumK()) ? 1.0 : 0.0;
        }
        for (index_t d = k_aligned; d < aux_size*k_aligned; d++, v++) {
          *v = 1.0;
        }
      }
    }
    model.GetParameter_b()[0] = 0.0;
    FFMScore score;
    for (size_t i = 0; i < 10; ++i) {
      real_t val = score.CalcScore(buf, model);
      // 6 + 24*4*3 = 294
      EXPECT_FLOAT_EQ(val, 6+k*4*3);
    }
  }
}

TEST(FFMScore_Test, calc_score_overflow) {
  for (index_t k = 1; k < 100; ++k) {
    // Init hyper_param
    HyperParam param;
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "squared";
    param.score_func = "ffm";
    param.num_feature = 3;
    param.num_K = k;
    param.num_field = 3;
    // Init the row
    RowBuffer buf;
    const index_t kRowLen = param.num_feature*2;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0, i);
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
    index_t aux_size = model.GetAuxiliarySize();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        for (index_t d = 0; d < k_aligned; d++, v++) {
          *v = (d < model.GetNumK()) ? 1.0 : 0.0;
        }
        for (index_t d = k_aligned; d < aux_size*k_aligned; d++, v++) {
          *v = 1.0;
        }
      }
    }
    model.GetParameter_b()[0] = 0.0;
    FFMScore score;
    for (size_t i = 0; i < 10; ++i) {
      real_t val = score.CalcScore(buf, model);
      EXPECT_FLOAT_EQ(val, 6+k*4*3);
    }
  }
}

// The ftrl latent update drives a weight to zero exactly when |z| falls inside
// the l1 band. These two cover either side of that choice. Only j != f is
// checked: a pair (j1,f1),(j2,f2) touches w[j1][f2] and w[j2][f1], so with one
// field per feature the diagonal is never updated.
TEST(FFMScore_Test, calc_grad_ftrl_inside_l1_band) {
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    model.Initialize("ffm", "squared", 3, 3, k, 3);
    RowBuffer buf;
    const index_t kRowLen = 3;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0, i);
    }
    FFMScore score;
    std::string opt_type("ftrl");
    // An l1 no |z| can exceed, so every coordinate lands inside the band.
    score.Initialize(0.1, 0, 0.3, 1.0, 1e10, 0, opt_type);
    score.CalcGrad(buf, model, 1.0);
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    index_t align0 = 3 * k_aligned;
    index_t align1 = model.GetNumField() * align0;
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        if (j == f) continue;
        real_t* base = v + j*align1 + f*align0;
        for (index_t d = 0; d < model.GetNumK(); ++d) {
          EXPECT_FLOAT_EQ(base[d], 0.0);
        }
      }
    }
  }
}

TEST(FFMScore_Test, calc_grad_ftrl_outside_l1_band) {
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    model.Initialize("ffm", "squared", 3, 3, k, 3);
    RowBuffer buf;
    const index_t kRowLen = 3;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0, i);
    }
    FFMScore score;
    std::string opt_type("ftrl");
    // No band at all, so every coordinate takes the computed value.
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
    score.CalcGrad(buf, model, 1.0);
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    index_t align0 = 3 * k_aligned;
    index_t align1 = model.GetNumField() * align0;
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        if (j == f) continue;
        real_t* base = v + j*align1 + f*align0;
        // Only the K coordinates the model actually carries: the padding up to
        // k_aligned starts at zero on both sides of a pair, so its gradient is
        // zero and it is meant to stay there.
        for (index_t d = 0; d < model.GetNumK(); ++d) {
          EXPECT_TRUE(std::isfinite(base[d]));
          EXPECT_NE(base[d], 0.0);
        }
      }
    }
  }
}

// L2 belongs to ftrl's proximal step, which is the lambda_2 in the weight
// denominator. Feeding lambda_2 * weight into the gradient as well applies it
// twice, and with no loss gradient at all the weights still move -- a nonzero
// weight manufacturing a gradient out of itself. Both ftrl tests above run at
// lambda_2 = 0, where the two placements cannot be told apart.
TEST(FFMScore_Test, calc_grad_ftrl_zero_pg_moves_nothing) {
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    model.Initialize("ffm", "squared", 3, 3, k, 3);
    RowBuffer buf;
    const index_t kRowLen = 3;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0, i);
    }
    FFMScore score;
    std::string opt_type("ftrl");
    // No l1 band, and an l2 large enough that a doubled one is unmissable.
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0.5, opt_type);
    score.CalcGrad(buf, model, 0.0);
    real_t* v = model.GetParameter_v();
    real_t* w = model.GetParameter_w();
    index_t k_aligned = model.get_aligned_k();
    index_t align0 = 3 * k_aligned;
    index_t align1 = model.GetNumField() * align0;
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        // A pair reaches (feature, other field) blocks only, so the diagonal
        // keeps its initial value and says nothing about the update.
        if (j == f) continue;
        real_t* base = v + j*align1 + f*align0;
        for (index_t d = 0; d < model.GetNumK(); ++d) {
          EXPECT_FLOAT_EQ(base[d], 0.0);
        }
      }
    }
    for (index_t i = 0; i < model.GetNumParameter_w(); i += 3) {
      EXPECT_FLOAT_EQ(w[i], 0.0);
    }
  }
}

} // namespace xLearn
