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
  // linear term
  real_t sqrt_norm = std::sqrt(norm);
  real_t* w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id];
    real_t gradient = regu_lambda_ * wl + pg * row.val(n) * sqrt_norm;
    wl -= (learning_rate_ * gradient);
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t g = pg;
  wb -= learning_rate_ * g;
}

// Calculate gradient and update current model using adagrad
void LinearScore::calc_grad_adagrad(RowRef row,
                                    Model& model,
                                    real_t pg,
                                    real_t norm) {
  // linear term
  real_t sqrt_norm = std::sqrt(norm);
  real_t* w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    index_t idx_g = feat_id * 2;
    index_t idx_c = idx_g + 1;
    real_t gradient = regu_lambda_ * w[idx_g] +
                      pg * row.val(n) * sqrt_norm;
    // Hold the updated cache in a register: writing it to w[] and reading it
    // straight back puts a store-to-load round trip on the critical path,
    // ahead of a square root that is already the longest link in it.
    real_t cache = w[idx_c] + gradient * gradient;
    w[idx_c] = cache;
    w[idx_g] -= (learning_rate_ * gradient * InvSqrt(cache));
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbg = w[1];
  real_t g = pg;
  wbg += g*g;
  wb -= learning_rate_ * g * InvSqrt(wbg);
}

// Calculate gradient and update current model using ftrl
void LinearScore::calc_grad_ftrl(RowRef row,
                                 Model& model,
                                 real_t pg,
                                 real_t norm) {
  // linear term
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*3];
    real_t &wlg = w[feat_id*3+1];
    real_t &wlz = w[feat_id*3+2];
    real_t g = lambda_2_*wl+pg*row.val(n)*sqrt_norm; 
    real_t old_wlg = wlg;
    wlg += g*g;
    real_t sigma = (std::sqrt(wlg)-std::sqrt(old_wlg)) * inv_alpha_;
    wlz += (g-sigma*wl);
    int sign = wlz > 0 ? 1:-1;
    if (sign*wlz <= lambda_1_) {
      wl = 0;
    } else {
      wl = (sign*lambda_1_-wlz) / 
           ((beta_ + std::sqrt(wlg)) *
            inv_alpha_ + lambda_2_);
    }
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbg = w[1];
  real_t &wbz = w[2];
  real_t g = pg;
  real_t old_wbg = wbg;
  wbg += g*g;
  real_t sigma = (std::sqrt(wbg)-std::sqrt(old_wbg)) * inv_alpha_;
  wbz += (g-sigma*wb);
  int sign = wbz > 0 ? 1:-1;
  if (sign*wbz <= lambda_1_) {
    wb = 0;
  } else {
    wb = (sign*lambda_1_-wbz) / 
         ((beta_ + std::sqrt(wbg)) *
          inv_alpha_ + lambda_2_);
  }
}

} // namespace xLearn
