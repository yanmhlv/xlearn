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

#include <vector>

#include "src/score/ffm_score.h"
#include "src/base/math.h"
#include "src/base/simd.h"

namespace xLearn {

namespace {

//------------------------------------------------------------------------------
// One usable feature of a row, with its share of the address arithmetic
// already done. Every function below walks the row as pairs, so it is
// quadratic in the row length, while a feature's own bounds check and index
// multiply are not: resolved once here, they are saved from each of the
// pairs the feature takes part in.
//------------------------------------------------------------------------------
struct Term {
  real_t* base;       // GetParameter_v() + feat_id * align1
  index_t field_off;  // field_id * align0
  real_t val;
};

// The usable entries of row, in order. Storage is reused between calls on the
// same thread, which is what the lock-free trainers need: they share a model
// but never a row.
const std::vector<Term>& RowTerms(const SparseRow* row,
                                  Model& model,
                                  index_t align0) {
  static thread_local std::vector<Term> terms;
  terms.clear();
  real_t* v = model.GetParameter_v();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  index_t align1 = num_field * align0;
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    // To avoid unseen feature in Prediction
    if (iter->feat_id >= num_feat || iter->field_id >= num_field) continue;
    terms.push_back({v + iter->feat_id * align1,
                     iter->field_id * align0,
                     iter->feat_val});
  }
  return terms;
}

} // namespace

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
  int align = kAlign * aux_size;
  const std::vector<Term>& terms = RowTerms(row, model, align0);
  const size_t num_term = terms.size();
  const index_t unrolled_end = align0 - align0 % (4*align);
  Float4 total0 = Float4::Zero();
  Float4 total1 = Float4::Zero();
  Float4 total2 = Float4::Zero();
  Float4 total3 = Float4::Zero();
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Float4 val = Float4::Broadcast(v1*terms[j].val);
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
  switch (opt_) {
    case kSgd:
      this->calc_grad_sgd(row, model, pg, norm);
      break;
    case kAdaGrad:
      this->calc_grad_adagrad(row, model, pg, norm);
      break;
    case kFtrl:
      this->calc_grad_ftrl(row, model, pg, norm);
      break;
  }
}

// Calculate gradient and update current model using sgd
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
  index_t align = kAlign * model.GetAuxiliarySize();
  const std::vector<Term>& terms = RowTerms(row, model, align0);
  const size_t num_term = terms.size();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 lr = Float4::Broadcast(learning_rate_);
  Float4 lamb = Float4::Broadcast(regu_lambda_);
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Float4 val = Float4::Broadcast(v1*terms[j].val);
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
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*2];
    real_t &wlg = w[feat_id*2+1];
    real_t g = regu_lambda_*wl+pg*iter->feat_val*sqrt_norm;
    real_t cache = wlg + g*g;
    wlg = cache;
    wl -= learning_rate_ * g * InvSqrt(cache);
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
  index_t align = kAlign * 2;
  const std::vector<Term>& terms = RowTerms(row, model, align0);
  const size_t num_term = terms.size();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 lr = Float4::Broadcast(learning_rate_);
  Float4 lamb = Float4::Broadcast(regu_lambda_);
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Float4 val = Float4::Broadcast(v1*terms[j].val);
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
  // Every step below scales by 1/alpha. Dividing a vector by a loop-invariant
  // scalar keeps the divider busy for what a reciprocal and a multiply settle
  // once per row.
  real_t inv_alpha = 1.0 / alpha_;
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
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
    real_t sigma = (sqrt(wlg)-sqrt(old_wlg)) * inv_alpha;
    wlz += (g-sigma*wl);
    int sign = wlz > 0 ? 1:-1;
    if (sign*wlz <= lambda_1_) {
      wl = 0;
    } else {
      wl = (sign*lambda_1_-wlz) /
           ((beta_ + sqrt(wlg)) *
            inv_alpha + lambda_2_);
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
  real_t sigma = (sqrt(wbg)-sqrt(old_wbg)) * inv_alpha;
  wbz += (g-sigma*wb);
  int sign = wbz > 0 ? 1:-1;
  if (sign*wbz <= lambda_1_) {
    wb = 0;
  } else {
    wb = (sign*lambda_1_-wbz) /
         ((beta_ + sqrt(wbg)) *
          inv_alpha + lambda_2_);
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = 3 * model.get_aligned_k();
  index_t align = kAlign * 3;
  const std::vector<Term>& terms = RowTerms(row, model, align0);
  const size_t num_term = terms.size();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 rcp_alpha = Float4::Broadcast(inv_alpha);
  Float4 beta = Float4::Broadcast(beta_);
  Float4 l1 = Float4::Broadcast(lambda_1_);
  Float4 l2 = Float4::Broadcast(lambda_2_);
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Float4 val = Float4::Broadcast(v1*terms[j].val);
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
                         - Sqrt(weight_grad1)) * rcp_alpha;
        Float4 sigma2 = (Sqrt(weight_grad2 + grad_sq2)
                         - Sqrt(weight_grad2)) * rcp_alpha;
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
        Float4 denominator1 = MulAdd(beta + Sqrt(weight_grad1),
                                     rcp_alpha, l2);
        Float4 denominator2 = MulAdd(beta + Sqrt(weight_grad2),
                                     rcp_alpha, l2);
        IfThenZeroElse(Abs(z_val1) <= l1,
                       numerator1 / denominator1).Store(w1);
        IfThenZeroElse(Abs(z_val2) <= l1,
                       numerator2 / denominator2).Store(w2);
      }
    }
  }
}

} // namespace xLearn
