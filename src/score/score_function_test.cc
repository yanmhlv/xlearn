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
This file tests the Score class.
*/

#include "gtest/gtest.h"

#include "src/score/score_function.h"
#include "src/data/model_parameters.h"

namespace xLearn {

Score* CreateScore(const char* format_name) {
  return CREATE_SCORE(format_name);
}

TEST(SCORE_TEST, Create_Score) {
  EXPECT_TRUE(CreateScore("linear") != NULL);
  EXPECT_TRUE(CreateScore("fm") != NULL);
  EXPECT_TRUE(CreateScore("ffm") != NULL);
  EXPECT_TRUE(CreateScore("") == NULL);
  EXPECT_TRUE(CreateScore("unknow_name") == NULL);
}

// How many planes each optimizer keeps per weight. The gradient cache travels
// with the model, so this number is baked into every checkpoint.
TEST(SCORE_TEST, auxiliary_size_per_optimizer) {
  EXPECT_EQ(AuxiliarySizeFor(Score::OptType::kSgd), 1u);
  EXPECT_EQ(AuxiliarySizeFor(Score::OptType::kAdaGrad), 2u);
  EXPECT_EQ(AuxiliarySizeFor(Score::OptType::kFtrl), 3u);
}

// A model carries its plane count but not the optimizer that chose it, and
// -pre takes the count from the file while -p still comes from the command
// line. Scoring a 3-plane ftrl checkpoint as sgd reads every weight at the
// wrong stride, which is silent: the run trains and writes a model.
TEST(SCORE_TEST, rejects_a_model_whose_planes_the_optimizer_did_not_make) {
  Model model;
  model.Initialize("linear", "squared", 4, 0, 0, 3);
  Score* score = CreateScore("linear");
  std::string opt_type("sgd");
  score->Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
  EXPECT_DEATH(score->CheckModel(model), "auxiliary");
  // The solver reports rather than aborts, and needs both numbers to say so.
  std::string problem = score->ModelMismatch(model);
  EXPECT_NE(problem.find("3 auxiliary planes"), std::string::npos) << problem;
  EXPECT_NE(problem.find("produces 1"), std::string::npos) << problem;
  delete score;
}

TEST(SCORE_TEST, accepts_a_model_the_optimizer_made) {
  const char* kOpts[] = {"sgd", "adagrad", "ftrl"};
  const index_t kAux[] = {1, 2, 3};
  for (int i = 0; i < 3; ++i) {
    Model model;
    model.Initialize("linear", "squared", 4, 0, 0, kAux[i]);
    Score* score = CreateScore("linear");
    std::string opt_type(kOpts[i]);
    score->Initialize(0.1, 0, 0.3, 1.0, 0, 0, opt_type);
    score->CheckModel(model);
    EXPECT_TRUE(score->ModelMismatch(model).empty()) << kOpts[i];
    delete score;
  }
}

}  // namespace xLearn
