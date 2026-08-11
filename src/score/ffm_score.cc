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

#include <cmath>
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
const std::vector<Term>& RowTerms(RowRef row,
                                  Model& model,
                                  index_t align0) {
  static thread_local std::vector<Term> terms;
  terms.clear();
  real_t* v = model.GetParameter_v();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  index_t align1 = num_field * align0;
  for (index_t n = 0; n < row.len; ++n) {
    // To avoid unseen feature in Prediction
    if (row.feat(n) >= num_feat || row.field(n) >= num_field) continue;
    terms.push_back({v + row.feat(n) * align1,
                     row.field(n) * align0,
                     row.val(n)});
  }
  return terms;
}

// Whether the latent planes can be walked eight lanes at a time -- the same
// rule FM uses, see WideLanes() there.
inline bool WideLanes(index_t aligned_k) { return aligned_k % 8 == 0; }

//------------------------------------------------------------------------------
// The latent half of each kernel, one per optimizer plus the score.
//
// A (feature, field) block holds each auxiliary quantity as a whole plane of
// aligned_k -- weights, then the gradient cache, then FTRL's z -- rather than
// interleaving them every four floats. Two things follow. Scoring reads only
// the first plane, so it touches half the cache lines it used to under adagrad
// and a third under FTRL. And the vector width stops being part of the layout,
// which is what lets these dispatch on it the way FM's do.
//------------------------------------------------------------------------------

template <int N>
real_t LatentScore(const std::vector<Term>& terms,
                   index_t aligned_k,
                   real_t norm) {
  const size_t num_term = terms.size();
  const index_t kStep = Vec<N>::Lanes();
  const index_t unrolled_end = aligned_k - aligned_k % (4*kStep);
  Vec<N> total0 = Vec<N>::Zero();
  Vec<N> total1 = Vec<N>::Zero();
  Vec<N> total2 = Vec<N>::Zero();
  Vec<N> total3 = Vec<N>::Zero();
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1 = base1 + terms[j].field_off;
      real_t* w2 = terms[j].base + field_off1;
      Vec<N> val = Vec<N>::Broadcast(v1*terms[j].val);
      auto accumulate = [&](Vec<N> acc, index_t d) {
        return MulAdd(Vec<N>::Load(w1 + d) * Vec<N>::Load(w2 + d), val, acc);
      };
      index_t d = 0;
      for (; d < unrolled_end; d += 4*kStep) {
        total0 = accumulate(total0, d);
        total1 = accumulate(total1, d + kStep);
        total2 = accumulate(total2, d + 2*kStep);
        total3 = accumulate(total3, d + 3*kStep);
      }
      // A short K leaves only this tail, and with no second chain to hide it
      // a fused multiply-add would just lengthen the one accumulator.
      for (; d < aligned_k; d += kStep) {
        total0 = total0 + Vec<N>::Load(w1 + d) * Vec<N>::Load(w2 + d) * val;
      }
    }
  }
  return ((total0 + total1) + (total2 + total3)).Sum();
}

template <int N>
void LatentSgd(const std::vector<Term>& terms,
               index_t aligned_k,
               real_t pg,
               real_t norm,
               real_t learning_rate,
               real_t regu_lambda) {
  const size_t num_term = terms.size();
  Vec<N> pg_all = Vec<N>::Broadcast(pg);
  Vec<N> lr = Vec<N>::Broadcast(learning_rate);
  Vec<N> lamb = Vec<N>::Broadcast(regu_lambda);
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Vec<N> pgv = Vec<N>::Broadcast(v1*terms[j].val) * pg_all;
      for (index_t d = 0; d < aligned_k; d += Vec<N>::Lanes()) {
        real_t* w1 = w1_base + d;
        real_t* w2 = w2_base + d;
        Vec<N> weight1 = Vec<N>::Load(w1);
        Vec<N> weight2 = Vec<N>::Load(w2);
        Vec<N> grad1 = MulAdd(lamb, weight1, pgv * weight2);
        Vec<N> grad2 = MulAdd(lamb, weight2, pgv * weight1);
        NegMulAdd(lr, grad1, weight1).Store(w1);
        NegMulAdd(lr, grad2, weight2).Store(w2);
      }
    }
  }
}

template <int N>
void LatentAdagrad(const std::vector<Term>& terms,
                   index_t aligned_k,
                   real_t pg,
                   real_t norm,
                   real_t learning_rate,
                   real_t regu_lambda) {
  const size_t num_term = terms.size();
  Vec<N> pg_all = Vec<N>::Broadcast(pg);
  Vec<N> lr = Vec<N>::Broadcast(learning_rate);
  Vec<N> lamb = Vec<N>::Broadcast(regu_lambda);
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Vec<N> pgv = Vec<N>::Broadcast(v1*terms[j].val) * pg_all;
      for (index_t d = 0; d < aligned_k; d += Vec<N>::Lanes()) {
        real_t* w1 = w1_base + d;
        real_t* w2 = w2_base + d;
        real_t* wg1 = w1 + aligned_k;
        real_t* wg2 = w2 + aligned_k;
        Vec<N> weight1 = Vec<N>::Load(w1);
        Vec<N> weight2 = Vec<N>::Load(w2);
        Vec<N> weight_grad1 = Vec<N>::Load(wg1);
        Vec<N> weight_grad2 = Vec<N>::Load(wg2);
        Vec<N> grad1 = MulAdd(lamb, weight1, pgv * weight2);
        Vec<N> grad2 = MulAdd(lamb, weight2, pgv * weight1);
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

template <int N>
void LatentFtrl(const std::vector<Term>& terms,
                index_t aligned_k,
                real_t pg,
                real_t norm,
                real_t inv_alpha_val,
                real_t beta_val,
                real_t lambda_1_val,
                real_t lambda_2_val) {
  const size_t num_term = terms.size();
  Vec<N> pg_all = Vec<N>::Broadcast(pg);
  Vec<N> inv_alpha = Vec<N>::Broadcast(inv_alpha_val);
  Vec<N> beta = Vec<N>::Broadcast(beta_val);
  Vec<N> l1 = Vec<N>::Broadcast(lambda_1_val);
  Vec<N> l2 = Vec<N>::Broadcast(lambda_2_val);
  for (size_t i = 0; i < num_term; ++i) {
    real_t* base1 = terms[i].base;
    index_t field_off1 = terms[i].field_off;
    real_t v1 = terms[i].val * norm;
    for (size_t j = i+1; j < num_term; ++j) {
      real_t* w1_base = base1 + terms[j].field_off;
      real_t* w2_base = terms[j].base + field_off1;
      Vec<N> pgv = Vec<N>::Broadcast(v1*terms[j].val) * pg_all;
      for (index_t d = 0; d < aligned_k; d += Vec<N>::Lanes()) {
        real_t* w1 = w1_base + d;
        real_t* w2 = w2_base + d;
        real_t* wg1 = w1 + aligned_k;
        real_t* wg2 = w2 + aligned_k;
        real_t* z1 = w1 + aligned_k * 2;
        real_t* z2 = w2 + aligned_k * 2;
        Vec<N> weight1 = Vec<N>::Load(w1);
        Vec<N> weight2 = Vec<N>::Load(w2);
        Vec<N> weight_grad1 = Vec<N>::Load(wg1);
        Vec<N> weight_grad2 = Vec<N>::Load(wg2);
        Vec<N> z_val1 = Vec<N>::Load(z1);
        Vec<N> z_val2 = Vec<N>::Load(z2);
        Vec<N> grad1 = MulAdd(l2, weight1, pgv * weight2);
        Vec<N> grad2 = MulAdd(l2, weight2, pgv * weight1);
        Vec<N> grad_sq1 = grad1 * grad1;
        Vec<N> grad_sq2 = grad2 * grad2;
        Vec<N> sigma1 = (Sqrt(weight_grad1 + grad_sq1)
                         - Sqrt(weight_grad1)) * inv_alpha;
        Vec<N> sigma2 = (Sqrt(weight_grad2 + grad_sq2)
                         - Sqrt(weight_grad2)) * inv_alpha;
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
        Vec<N> numerator1 = CopySign(l1, z_val1) - z_val1;
        Vec<N> numerator2 = CopySign(l1, z_val2) - z_val2;
        Vec<N> denominator1 = MulAdd(beta + Sqrt(weight_grad1),
                                     inv_alpha, l2);
        Vec<N> denominator2 = MulAdd(beta + Sqrt(weight_grad2),
                                     inv_alpha, l2);
        IfThenZeroElse(Abs(z_val1) <= l1,
                       numerator1 / denominator1).Store(w1);
        IfThenZeroElse(Abs(z_val2) <= l1,
                       numerator2 / denominator2).Store(w2);
      }
    }
  }
}

} // namespace

// y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Using SIMD to accelerate vector operation.
real_t FFMScore::CalcScore(RowRef row,
                           Model& model,
                           real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sum_w = 0;
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t aux_size = model.GetAuxiliarySize();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    sum_w += (row.val(n) * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  sum_w += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  const std::vector<Term>& terms = RowTerms(row, model, aux_size * aligned_k);
  real_t sum_v = WideLanes(aligned_k)
                     ? LatentScore<8>(terms, aligned_k, norm)
                     : LatentScore<4>(terms, aligned_k, norm);

  return sum_v + sum_w;
}

// Calculate gradient and update current model.
// Using the SIMD to accelerate vector operation.
void FFMScore::CalcGrad(RowRef row,
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
void FFMScore::calc_grad_sgd(RowRef row,
                             Model& model,
                             real_t pg,
                             real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id];
    real_t g = regu_lambda_*wl+pg*row.val(n)*sqrt_norm;
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
  index_t aligned_k = model.get_aligned_k();
  const std::vector<Term>& terms =
      RowTerms(row, model, model.GetAuxiliarySize() * aligned_k);
  if (WideLanes(aligned_k)) {
    LatentSgd<8>(terms, aligned_k, pg, norm, learning_rate_, regu_lambda_);
  } else {
    LatentSgd<4>(terms, aligned_k, pg, norm, learning_rate_, regu_lambda_);
  }
}

// Calculate gradient and update current model using adagrad
void FFMScore::calc_grad_adagrad(RowRef row,
                                 Model& model,
                                 real_t pg,
                                 real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*2];
    real_t &wlg = w[feat_id*2+1];
    real_t g = regu_lambda_*wl+pg*row.val(n)*sqrt_norm;
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
  index_t aligned_k = model.get_aligned_k();
  const std::vector<Term>& terms =
      RowTerms(row, model, model.GetAuxiliarySize() * aligned_k);
  if (WideLanes(aligned_k)) {
    LatentAdagrad<8>(terms, aligned_k, pg, norm, learning_rate_, regu_lambda_);
  } else {
    LatentAdagrad<4>(terms, aligned_k, pg, norm, learning_rate_, regu_lambda_);
  }
}

// Calculate gradient and update current model using ftrl
void FFMScore::calc_grad_ftrl(RowRef row,
                              Model& model,
                              real_t pg,
                              real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
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
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  const std::vector<Term>& terms =
      RowTerms(row, model, model.GetAuxiliarySize() * aligned_k);
  if (WideLanes(aligned_k)) {
    LatentFtrl<8>(terms, aligned_k, pg, norm,
                  inv_alpha_, beta_, lambda_1_, lambda_2_);
  } else {
    LatentFtrl<4>(terms, aligned_k, pg, norm,
                  inv_alpha_, beta_, lambda_1_, lambda_2_);
  }
}

} // namespace xLearn
