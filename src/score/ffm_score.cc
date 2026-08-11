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
This file is the implementation of FFMScore class.
*/

#include "src/score/ffm_score.h"
#include "src/base/math.h"
#include "src/base/simd.h"

namespace xLearn {

// y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Using SIMD to accelerate vector operation.
real_t FFMScore::CalcScore(const SparseRow* row,
                           Model& model,
                           real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sum_w = 0;
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  index_t aux_size = model.GetAuxiliarySize();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    sum_w += (iter->feat_val * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  sum_w += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = aux_size * model.get_aligned_k();
  index_t align1 = num_field * align0;
  int align = kAlign * aux_size;
  w = model.GetParameter_v();
  const index_t unrolled_end = align0 - align0 % (4*align);
  Float4 total0 = Float4::Zero();
  Float4 total1 = Float4::Zero();
  Float4 total2 = Float4::Zero();
  Float4 total3 = Float4::Zero();
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat || f1 >= num_field) continue;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      // To avoid unseen feature in Prediction
      if (j2 >= num_feat || f2 >= num_field) continue;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      Float4 val = Float4::Broadcast(v1*v2*norm);
      auto accumulate = [&](Float4 acc, index_t d) {
        return MulAdd(Float4::Load(w1_base + d) * Float4::Load(w2_base + d),
                      val, acc);
      };
      index_t d = 0;
      for (; d < unrolled_end; d += 4*align) {
        total0 = accumulate(total0, d);
        total1 = accumulate(total1, d + align);
        total2 = accumulate(total2, d + 2*align);
        total3 = accumulate(total3, d + 3*align);
      }
      // A short K leaves only this tail, and with no second chain to hide it
      // a fused multiply-add would just lengthen the one accumulator.
      for (; d < align0; d += align) {
        total0 = total0 + Float4::Load(w1_base + d)
                          * Float4::Load(w2_base + d) * val;
      }
    }
  }
  real_t sum_v = ((total0 + total1) + (total2 + total3)).Sum();

  return sum_v + sum_w;
}

// Calculate gradient and update current model.
// Using the SIMD to accelerate vector operation.
void FFMScore::CalcGrad(const SparseRow* row,
                        Model& model,
                        real_t pg,
                        real_t norm) {
  // Using sgd
  if (opt_type_.compare("sgd") == 0) {
    this->calc_grad_sgd(row, model, pg, norm);
  }
  // Using adagrad
  else if (opt_type_.compare("adagrad") == 0) {
    this->calc_grad_adagrad(row, model, pg, norm);
  }
  // Using ftrl 
  else if (opt_type_.compare("ftrl") == 0) {
    this->calc_grad_ftrl(row, model, pg, norm);
  } 
  else {
    LOG(FATAL) << "Unknow optimization method: " << opt_type_;
  }
}

// Calculate gradient and update current model using sgd
// TODO(aksnzhy): solve unseen feature
void FFMScore::calc_grad_sgd(const SparseRow* row,
                             Model& model,
                             real_t pg,
                             real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id];
    real_t g = regu_lambda_*wl+pg*iter->feat_val*sqrt_norm;
    wl -= (learning_rate_ * g);
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t g = pg;
  wb -= (learning_rate_ * g);
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = model.GetAuxiliarySize() * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * model.GetAuxiliarySize();
  w = model.GetParameter_v();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 lr = Float4::Broadcast(learning_rate_);
  Float4 lamb = Float4::Broadcast(regu_lambda_);
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    // To avoid unseen feature
    if (j1 >= num_feat || f1 >= num_field) continue;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      // To avoid unseen feature
      if (j2 >= num_feat || f2 >= num_field) continue;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      Float4 val = Float4::Broadcast(v1*v2*norm);
      Float4 pgv = val * pg_all;
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        Float4 weight1 = Float4::Load(w1);
        Float4 weight2 = Float4::Load(w2);
        Float4 grad1 = MulAdd(lamb, weight1, pgv * weight2);
        Float4 grad2 = MulAdd(lamb, weight2, pgv * weight1);
        NegMulAdd(lr, grad1, weight1).Store(w1);
        NegMulAdd(lr, grad2, weight2).Store(w2);
      }
    }
  }

}

// Calculate gradient and update current model using adagrad
// TODO(aksnzhy): solve unseen feature
void FFMScore::calc_grad_adagrad(const SparseRow* row,
                                 Model& model,
                                 real_t pg,
                                 real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*2];
    real_t &wlg = w[feat_id*2+1];
    real_t g = regu_lambda_*wl+pg*iter->feat_val*sqrt_norm;
    wlg += g*g;
    wl -= learning_rate_ * g * InvSqrt(wlg);
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbg = w[1];
  real_t g = pg;
  wbg += g*g;
  wb -= learning_rate_ * g * InvSqrt(wbg);
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = 2 * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * 2;
  w = model.GetParameter_v();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 lr = Float4::Broadcast(learning_rate_);
  Float4 lamb = Float4::Broadcast(regu_lambda_);
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    // To avoid unseen feature
    if (j1 >= num_feat || f1 >= num_field) continue;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      // To avoid unseen feature
      if (j2 >= num_feat || f2 >= num_field) continue;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      Float4 val = Float4::Broadcast(v1*v2*norm);
      Float4 pgv = val * pg_all;
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        real_t *wg1 = w1 + kAlign;
        real_t *wg2 = w2 + kAlign;
        Float4 weight1 = Float4::Load(w1);
        Float4 weight2 = Float4::Load(w2);
        Float4 weight_grad1 = Float4::Load(wg1);
        Float4 weight_grad2 = Float4::Load(wg2);
        Float4 grad1 = MulAdd(lamb, weight1, pgv * weight2);
        Float4 grad2 = MulAdd(lamb, weight2, pgv * weight1);
        weight_grad1 = MulAdd(grad1, grad1, weight_grad1);
        weight_grad2 = MulAdd(grad2, grad2, weight_grad2);
        NegMulAdd(lr, RSqrt(weight_grad1) * grad1, weight1).Store(w1);
        NegMulAdd(lr, RSqrt(weight_grad2) * grad2, weight2).Store(w2);
        weight_grad1.Store(wg1);
        weight_grad2.Store(wg2);
      }
    }
  }
}

// Calculate gradient and update current model using ftrl
void FFMScore::calc_grad_ftrl(const SparseRow* row,
                              Model& model,
                              real_t pg,
                              real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/  
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*3];
    real_t &wlg = w[feat_id*3+1];
    real_t &wlz = w[feat_id*3+2];
    real_t g = lambda_2_*wl+pg*iter->feat_val*sqrt_norm; 
    real_t old_wlg = wlg;
    wlg += g*g;
    real_t sigma = (sqrt(wlg)-sqrt(old_wlg)) / alpha_;
    wlz += (g-sigma*wl);
    int sign = wlz > 0 ? 1:-1;
    if (sign*wlz <= lambda_1_) {
      wl = 0;
    } else {
      wl = (sign*lambda_1_-wlz) / 
           ((beta_ + sqrt(wlg)) / 
            alpha_ + lambda_2_);
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
  real_t sigma = (sqrt(wbg)-sqrt(old_wbg)) / alpha_;
  wbz += (g-sigma*wb);
  int sign = wbz > 0 ? 1:-1;
  if (sign*wbz <= lambda_1_) {
    wb = 0;
  } else {
    wb = (sign*lambda_1_-wbz) / 
         ((beta_ + sqrt(wbg)) / 
          alpha_ + lambda_2_);
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = 3 * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * 3;
  w = model.GetParameter_v();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 alpha = Float4::Broadcast(alpha_);
  Float4 beta = Float4::Broadcast(beta_);
  Float4 l1 = Float4::Broadcast(lambda_1_);
  Float4 l2 = Float4::Broadcast(lambda_2_);
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    // To avoid unseen feature
    if (j1 >= num_feat || f1 >= num_field) continue;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      // To avoid unseen feature
      if (j2 >= num_feat || f2 >= num_field) continue;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      Float4 val = Float4::Broadcast(v1*v2*norm);
      Float4 pgv = val * pg_all;
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        real_t *wg1 = w1 + kAlign;
        real_t *wg2 = w2 + kAlign;
        real_t *z1 = w1 + kAlign * 2;
        real_t *z2 = w2 + kAlign * 2;
        Float4 weight1 = Float4::Load(w1);
        Float4 weight2 = Float4::Load(w2);
        Float4 weight_grad1 = Float4::Load(wg1);
        Float4 weight_grad2 = Float4::Load(wg2);
        Float4 z_val1 = Float4::Load(z1);
        Float4 z_val2 = Float4::Load(z2);
        Float4 grad1 = MulAdd(l2, weight1, pgv * weight2);
        Float4 grad2 = MulAdd(l2, weight2, pgv * weight1);
        Float4 grad_sq1 = grad1 * grad1;
        Float4 grad_sq2 = grad2 * grad2;
        Float4 sigma1 = (Sqrt(weight_grad1 + grad_sq1)
                         - Sqrt(weight_grad1)) / alpha;
        Float4 sigma2 = (Sqrt(weight_grad2 + grad_sq2)
                         - Sqrt(weight_grad2)) / alpha;
        z_val1 = NegMulAdd(sigma1, weight1, z_val1 + grad1);
        z_val2 = NegMulAdd(sigma2, weight2, z_val2 + grad2);
        z_val1.Store(z1);
        z_val2.Store(z2);
        weight_grad1 = weight_grad1 + grad_sq1;
        weight_grad2 = weight_grad2 + grad_sq2;
        weight_grad1.Store(wg1);
        weight_grad2.Store(wg2);
        // Update w. Where |z| is inside the l1 band the weight is driven to
        // zero, so the branch becomes a select over the whole vector.
        Float4 numerator1 = CopySign(l1, z_val1) - z_val1;
        Float4 numerator2 = CopySign(l1, z_val2) - z_val2;
        Float4 denominator1 = (beta + Sqrt(weight_grad1)) / alpha + l2;
        Float4 denominator2 = (beta + Sqrt(weight_grad2)) / alpha + l2;
        IfThenZeroElse(Abs(z_val1) <= l1,
                       numerator1 / denominator1).Store(w1);
        IfThenZeroElse(Abs(z_val2) <= l1,
                       numerator2 / denominator2).Store(w2);
      }
    }
  }
}

} // namespace xLearn
