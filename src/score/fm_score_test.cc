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
    // Init the row
    RowBuffer buf;
    const index_t kRowLen = param.num_feature;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0);
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
      real_t val = score.CalcScore(buf, model);
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
    // Init the row
    RowBuffer buf;
    const index_t kRowLen = param.num_feature*2;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0);
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
      real_t val = score.CalcScore(buf, model);
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
    RowBuffer buf;
    const index_t kRowLen = 3;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0);
    }
    FMScore score;
    std::string opt_type("ftrl");
    // An l1 no |z| can exceed, so every coordinate lands inside the band.
    score.Initialize(0.1, 0, 0.3, 1.0, 1e10, 0, opt_type);
    score.CalcGrad(buf, model, 1.0);
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
    RowBuffer buf;
    const index_t kRowLen = 3;
    for (index_t i = 0; i < kRowLen; ++i) {
      buf.Add(i, 2.0);
    }
    FMScore score;
    std::string opt_type("ftrl");
    // No band at all, so every coordinate takes the computed value.
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
    score.CalcGrad(buf, model, 1.0);
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      // Only the K coordinates the model actually carries: the padding up to
      // k_aligned starts at zero, contributes nothing to the latent sum, and so
      // has a zero gradient -- it is meant to stay there.
      for (index_t d = 0; d < model.GetNumK(); ++d) {
        EXPECT_TRUE(std::isfinite(v[j*k_aligned*3 + d]));
        EXPECT_NE(v[j*k_aligned*3 + d], 0.0);
      }
    }
  }
}

namespace {

// Three features at 2.0, every w and every carried v at 1.0, no bias. The
// padding above K stays at zero so it contributes nothing to either term.
void InitUnitModel(Model& model, index_t k, index_t aux_size) {
  model.Initialize("fm", "squared", 3, 3, k, aux_size);
  real_t* w = model.GetParameter_w();
  for (index_t i = 0; i < model.GetNumParameter_w(); ++i) {
    w[i] = 1.0;
  }
  real_t* v = model.GetParameter_v();
  index_t k_aligned = model.get_aligned_k();
  for (index_t j = 0; j < model.GetNumFeature(); ++j) {
    for (index_t d = 0; d < model.GetNumK(); ++d, ++v) {
      *v = 1.0;
    }
    for (index_t d = model.GetNumK(); d < k_aligned*aux_size; ++d, ++v) {
      *v = 0;
    }
  }
  model.GetParameter_b()[0] = 0.0;
}

} // namespace

// The row normalizer belongs on each factor as sqrt(norm), so that a pair
// carries exactly one norm -- the linear term beside it uses the same
// sqrt_norm, and FFM applies one norm per pair too. Every other test in this
// file scores at the default norm = 1.0, where sqrt(norm) and norm are the
// same number and the distinction cannot show up at all.
TEST(FMScoreTest, calc_score_normalized) {
  const real_t kNorm = 0.25;
  for (index_t k = 1; k < 100; ++k) {
    Model model;
    InitUnitModel(model, k, 2);
    RowBuffer buf;
    for (index_t i = 0; i < 3; ++i) {
      buf.Add(i, 2.0);
    }
    FMScore score;
    // Linear: 3 features * 2.0 * sqrt(0.25). Pairwise: 3 pairs, each an inner
    // product of k ones over values 2.0 * 2.0, carrying one factor of 0.25.
    // Scaling the pair by norm twice instead would give 3 + 0.75*k.
    EXPECT_FLOAT_EQ(score.CalcScore(buf, model, kNorm), 3 + 3*k);
  }
}

// The latent update has to scale the way the score did, because it reads the
// sum the score left behind.
TEST(FMScoreTest, calc_grad_sgd_normalized) {
  const real_t kNorm = 0.25;
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    InitUnitModel(model, k, 2);
    RowBuffer buf;
    for (index_t i = 0; i < 3; ++i) {
      buf.Add(i, 2.0);
    }
    FMScore score;
    std::string opt_type("sgd");
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
    score.CalcGrad(buf, model, 1.0, kNorm);
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      // val = 2.0*sqrt(0.25) = 1, s = 3 features * 1.0 * val = 3, so the
      // gradient is 1*(3 - 1*1) = 2 and the weight moves by 0.1*2.
      for (index_t d = 0; d < model.GetNumK(); ++d) {
        EXPECT_FLOAT_EQ(v[j*k_aligned*2 + d], 0.8);
      }
      for (index_t d = model.GetNumK(); d < k_aligned; ++d) {
        EXPECT_FLOAT_EQ(v[j*k_aligned*2 + d], 0.0);
      }
    }
  }
}

// L2 belongs to ftrl's proximal step, which is the lambda_2 in the weight
// denominator. Feeding lambda_2 * weight into the gradient as well applies it
// twice, and the symptom is this: with no loss gradient at all, the weights
// still move, because a nonzero weight manufactures a gradient out of itself.
// Every other ftrl test in this file runs at lambda_2 = 0, where the two
// placements cannot be told apart.
TEST(FMScoreTest, calc_grad_ftrl_zero_pg_moves_nothing) {
  for (index_t k = 1; k < 40; ++k) {
    Model model;
    model.Initialize("fm", "squared", 3, 3, k, 3);
    RowBuffer buf;
    for (index_t i = 0; i < 3; ++i) {
      buf.Add(i, 2.0);
    }
    FMScore score;
    std::string opt_type("ftrl");
    // No l1 band, and an l2 large enough that a doubled one is unmissable.
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0.5, opt_type);
    score.CalcGrad(buf, model, 0.0);

    index_t k_aligned = model.get_aligned_k();
    real_t* v = model.GetParameter_v();
    real_t* w = model.GetParameter_w();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t d = 0; d < k_aligned; ++d) {
        EXPECT_FLOAT_EQ(v[j*k_aligned*3 + d], 0.0);
      }
    }
    for (index_t i = 0; i < model.GetNumParameter_w(); i += 3) {
      EXPECT_FLOAT_EQ(w[i], 0.0);
    }

    // And having settled with no gradient, another empty step is a no-op.
    score.CalcGrad(buf, model, 0.0);
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t d = 0; d < k_aligned; ++d) {
        EXPECT_FLOAT_EQ(v[j*k_aligned*3 + d], 0.0);
      }
    }
  }
}

namespace {

real_t HalfOfPred(real_t pred, void* context, real_t* loss) {
  *loss = pred;
  return pred * 0.5;
}

} // namespace

// Real training takes Step(), which reuses the latent sum CalcScore() just
// filled; the tests above take the split path, which builds it again. The two
// have to agree, and only the split one is otherwise exercised.
TEST(FMScoreTest, step_matches_split_path) {
  const real_t kNorm = 0.25;
  for (index_t k = 1; k < 40; ++k) {
    Model fused;
    Model split;
    InitUnitModel(fused, k, 2);
    InitUnitModel(split, k, 2);
    RowBuffer buf;
    for (index_t i = 0; i < 3; ++i) {
      buf.Add(i, 2.0);
    }
    FMScore score;
    std::string opt_type("sgd");
    score.Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
    EXPECT_TRUE(score.PrefersFusedStep());

    real_t loss = 0;
    real_t fused_loss = score.Step(buf, fused, kNorm, HalfOfPred, nullptr);
    real_t pred = score.CalcScore(buf, split, kNorm);
    score.CalcGrad(buf, split, HalfOfPred(pred, nullptr, &loss), kNorm);

    EXPECT_FLOAT_EQ(fused_loss, loss);
    index_t k_aligned = fused.get_aligned_k();
    real_t* vf = fused.GetParameter_v();
    real_t* vs = split.GetParameter_v();
    for (index_t j = 0; j < fused.GetNumFeature(); ++j) {
      for (index_t d = 0; d < k_aligned; ++d) {
        EXPECT_FLOAT_EQ(vf[j*k_aligned*2 + d], vs[j*k_aligned*2 + d]);
      }
    }
    real_t* wf = fused.GetParameter_w();
    real_t* ws = split.GetParameter_w();
    for (index_t i = 0; i < fused.GetNumParameter_w(); ++i) {
      EXPECT_FLOAT_EQ(wf[i], ws[i]);
    }
  }
}

} // namespace xLearn
