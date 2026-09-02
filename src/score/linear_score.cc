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
This file is the implementation of LinearScore class.
*/

#include <cmath>

#include "src/score/linear_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = wTx (incluing bias term)
real_t LinearScore::CalcScore(RowRef row,
                              Model& model,
                              real_t norm) {
  real_t sqrt_norm = std::sqrt(norm);
  real_t* w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  real_t score = 0.0;
  index_t auxiliary_size = model.GetAuxiliarySize();
  // linear term
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    index_t idx = feat_id * auxiliary_size;
    score += w[idx] * row.val(n) * sqrt_norm;
  }
  // bias
  score += model.GetParameter_b()[0];
  return score;
}

// The row's linear weights, on their way before the row is scored.
void LinearScore::PrefetchParams(RowRef row, Model& model) {
  real_t* w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t auxiliary_size = model.GetAuxiliarySize();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    if (feat_id >= num_feat) continue;
    Prefetch(w + feat_id * auxiliary_size);
  }
}

// Calculate gradient and update current model
void LinearScore::CalcGrad(RowRef row,
                           Model& model,
                           real_t pg,
                           real_t norm) {
  switch (opt_) {
    case OptType::kSgd:
      this->calc_grad_sgd(row, model, pg, norm);
      break;
    case OptType::kAdaGrad:
      this->calc_grad_adagrad(row, model, pg, norm);
      break;
    case OptType::kFtrl:
      this->calc_grad_ftrl(row, model, pg, norm);
      break;
  }
}

// Calculate gradient and update current model using sgd
void LinearScore::calc_grad_sgd(RowRef row,
                                Model& model,
                                real_t pg,
                                real_t norm) {
  this->sgd_linear_grad(row, model, pg, std::sqrt(norm));
}

// Calculate gradient and update current model using adagrad
void LinearScore::calc_grad_adagrad(RowRef row,
                                    Model& model,
                                    real_t pg,
                                    real_t norm) {
  this->adagrad_linear_grad(row, model, pg, std::sqrt(norm));
}

// Calculate gradient and update current model using ftrl
void LinearScore::calc_grad_ftrl(RowRef row,
                                 Model& model,
                                 real_t pg,
                                 real_t norm) {
  this->ftrl_linear_grad(row, model, pg, std::sqrt(norm));
}

} // namespace xLearn
