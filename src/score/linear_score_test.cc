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
This file tests the LinearScore class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

#include "src/score/score_function.h"
#include "src/score/linear_score.h"

namespace xLearn {

HyperParam param;
const int kLength = 100;

class LinearScoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.num_param = kLength;
    param.loss_func = "squared";
    param.score_func = "linear";
    param.num_feature = kLength;
  }
};

TEST_F(LinearScoreTest, calc_score) {
  RowBuffer buf;
    const index_t kRowLen = kLength;
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                0, 0, 2);
  real_t* w = model.GetParameter_w();
  index_t num_w = model.GetNumParameter_w();
  for (index_t i = 0; i < num_w; ++i) {
    w[i] = 3.0;
  }
  model.GetParameter_b()[0] = 0.0;
  // Init the row
  for (index_t i = 0; i < kRowLen; ++i) {
    buf.Add(i, 2.0);
  }
  LinearScore score;
  real_t val = score.CalcScore(buf, model);
  EXPECT_FLOAT_EQ(val, 600.0);
}

TEST_F(LinearScoreTest, calc_score_overflow) {
  RowBuffer buf;
    const index_t kRowLen = 2*kLength;
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                0, 0, 2);
  real_t* w = model.GetParameter_w();
  index_t num_w = model.GetNumParameter_w();
  for (index_t i = 0; i < num_w; ++i) {
    w[i] = 3.0;
  }
  model.GetParameter_b()[0] = 0.0;
  // Init the row
  for (index_t i = 0; i < kRowLen; ++i) {
    buf.Add(i, 2.0);
  }
  LinearScore score;
  real_t val = score.CalcScore(buf, model);
  EXPECT_FLOAT_EQ(val, 600.0);
}

// ftrl_linear_grad() is shared by all three score functions, and it had no
// test at all. L2 belongs to ftrl's proximal step -- the lambda_2 in the
// weight denominator -- so the gradient handed to ftrl_update() is the loss
// gradient alone. Applying lambda_2 in both places lets a nonzero weight
// manufacture a gradient out of itself, which shows up with no loss gradient
// at all: the weights move when nothing was learned.
TEST_F(LinearScoreTest, calc_grad_ftrl_zero_pg_moves_nothing) {
  RowBuffer buf;
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                0, 0, 3);
  real_t* w = model.GetParameter_w();
  index_t num_w = model.GetNumParameter_w();
  for (index_t i = 0; i < num_w; i += 3) {
    w[i] = 3.0;
  }
  for (index_t i = 0; i < kLength; ++i) {
    buf.Add(i, 2.0);
  }
  LinearScore score;
  std::string opt_type("ftrl");
  // No l1 band, and an l2 large enough that a doubled one is unmissable.
  score.Initialize(0.1, 0, 0.3, 1.0, 0, 0.5, opt_type);
  score.CalcGrad(buf, model, 0.0);
  for (index_t i = 0; i < num_w; i += 3) {
    EXPECT_FLOAT_EQ(w[i], 0.0);
  }
  EXPECT_FLOAT_EQ(model.GetParameter_b()[0], 0.0);
}

} // namespace xLearn
