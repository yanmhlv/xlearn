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
This file is the implementation of FMScore class.
*/

#include <algorithm>
#include <cmath>
#include <vector>

#include "src/score/fm_score.h"
#include "src/base/math.h"
#include "src/base/simd.h"

namespace xLearn {

namespace {

// Scratch space for the per-row latent sums. The solver shares one Score
// across the whole thread pool, so this has to be per thread; outliving the
// call is what saves an allocation on every single row.
real_t* ZeroedScratch(index_t size) {
  static thread_local std::vector<real_t> buffer;
  if (buffer.size() < size) {
    buffer.resize(size);
  }
  std::fill_n(buffer.begin(), size, 0.0f);
  return buffer.data();
}

// Sum of weighted * (s - weighted) over the row's latent factors.
//
// A single accumulator would serialise the whole reduction on the latency of
// one add, so a wide K spreads it over kChains independent ones. That setup
// does not pay for itself on a narrow K, where one chain leaves this as the
// plain loop it started as.
template <int kChains>
real_t LatentSum(const SparseRow* row,
                 Model& model,
                 real_t norm,
                 const real_t* s,
                 index_t num_feat,
                 index_t aligned_k,
                 index_t align0) {
  Float4 total[kChains];
  for (int c = 0; c < kChains; ++c) {
    total[c] = Float4::Zero();
  }
  const index_t step = kChains * kAlign;
  const index_t unrolled_end = aligned_k - aligned_k % step;
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat) continue;
    real_t* w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(entry.feat_val * norm);
    index_t d = 0;
    for (; d < unrolled_end; d += step) {
      for (int c = 0; c < kChains; ++c) {
        index_t at = d + c*kAlign;
        Float4 weighted = Float4::Load(w+at) * val;
        Float4 partial = Float4::Load(s+at) - weighted;
        // Several chains in flight hide the longer latency of the fused
        // multiply-add; a single one would only be lengthened by it.
        total[c] = kChains == 1 ? total[c] + weighted * partial
                                : MulAdd(weighted, partial, total[c]);
      }
    }
    for (; d < aligned_k; d += kAlign) {
      Float4 weighted = Float4::Load(w+d) * val;
      total[0] = total[0] + weighted * (Float4::Load(s+d) - weighted);
    }
  }
  Float4 sum = total[0];
  for (int c = 1; c < kChains; ++c) {
    sum = sum + total[c];
  }
  return sum.Sum();
}

} // namespace

// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SIMD to accelerate vector operation.
real_t FMScore::CalcScore(const SparseRow* row,
                          Model& model,
                          real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  real_t t = 0;
  index_t aux_size = model.GetAuxiliarySize();
  for (const Node& entry : *row) {
    index_t feat_id = entry.feat_id;
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    t += (entry.feat_val * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  t += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * aux_size;
  real_t* s = ZeroedScratch(aligned_k);
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      Float4 sum = Float4::Load(s+d);
      Float4 const weight = Float4::Load(w+d);
      MulAdd(weight, val, sum).Store(s+d);
    }
  }
  real_t t_all = aligned_k >= 4*kAlign
      ? LatentSum<4>(row, model, norm, s, num_feat, aligned_k, align0)
      : LatentSum<1>(row, model, norm, s, num_feat, aligned_k, align0);
  t_all *= 0.5;
  t_all += t;
  return t_all;
}

// Calculate gradient and update current model parameters.
// Using SIMD to accelerate vector operation.
void FMScore::CalcGrad(const SparseRow* row,
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
void FMScore::calc_grad_sgd(const SparseRow* row,
                            Model& model,
                            real_t pg,
                            real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/  
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (const Node& entry : *row) {
    index_t feat_id = entry.feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id];
    real_t g = regu_lambda_*wl+pg*entry.feat_val*sqrt_norm;
    wl -= learning_rate_ * g;
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t g = pg;
  wb -= learning_rate_ * g;
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * 
                   model.GetAuxiliarySize();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 lr = Float4::Broadcast(learning_rate_);
  Float4 lamb = Float4::Broadcast(regu_lambda_);
  real_t* s = ZeroedScratch(aligned_k);
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      Float4 sum = Float4::Load(s+d);
      Float4 const weight = Float4::Load(w+d);
      MulAdd(weight, val, sum).Store(s+d);
    }
  }
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
  // To avoid unseen feature
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    Float4 pgv = pg_all * val;
    for(index_t d = 0; d < aligned_k; d += kAlign) {
      Float4 sum = Float4::Load(s+d);
      Float4 weight = Float4::Load(w+d);
      Float4 grad = MulAdd(lamb, weight, pgv * NegMulAdd(weight, val, sum));
      NegMulAdd(lr, grad, weight).Store(w+d);
    }
  }
}

// Calculate gradient and update current model using adagrad
void FMScore::calc_grad_adagrad(const SparseRow* row,
                                Model& model,
                                real_t pg,
                                real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (const Node& entry : *row) {
    index_t feat_id = entry.feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*2];
    real_t &wlg = w[feat_id*2+1];
    real_t g = regu_lambda_*wl+pg*entry.feat_val*sqrt_norm;
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
  index_t align0 = model.get_aligned_k() * 
                   model.GetAuxiliarySize();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 lr = Float4::Broadcast(learning_rate_);
  Float4 lamb = Float4::Broadcast(regu_lambda_);
  real_t* s = ZeroedScratch(aligned_k);
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      Float4 sum = Float4::Load(s+d);
      Float4 const weight = Float4::Load(w+d);
      MulAdd(weight, val, sum).Store(s+d);
    }
  }
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    Float4 pgv = pg_all * val;
    for(index_t d = 0; d < aligned_k; d += kAlign) {
      Float4 sum = Float4::Load(s+d);
      Float4 weight = Float4::Load(w+d);
      Float4 weight_grad = Float4::Load(w+aligned_k+d);
      Float4 grad = MulAdd(lamb, weight, pgv * NegMulAdd(weight, val, sum));
      weight_grad = MulAdd(grad, grad, weight_grad);
      NegMulAdd(lr, RSqrt(weight_grad) * grad, weight).Store(w+d);
      weight_grad.Store(w+aligned_k+d);
    }
  }
}

// Calculate gradient and update current model using ftrl
void FMScore::calc_grad_ftrl(const SparseRow* row,
                             Model& model,
                             real_t pg,
                             real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = std::sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  for (const Node& entry : *row) {
    index_t feat_id = entry.feat_id;
    // To avoid unseen feature
    if (feat_id >= num_feat) continue;
    real_t &wl = w[feat_id*3];
    real_t &wlg = w[feat_id*3+1];
    real_t &wlz = w[feat_id*3+2];
    real_t g = lambda_2_*wl+pg*entry.feat_val*sqrt_norm; 
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
  index_t align0 = model.get_aligned_k() * 
                   model.GetAuxiliarySize();
  Float4 pg_all = Float4::Broadcast(pg);
  Float4 inv_alpha = Float4::Broadcast(inv_alpha_);
  Float4 beta = Float4::Broadcast(beta_);
  Float4 l1 = Float4::Broadcast(lambda_1_);
  Float4 l2 = Float4::Broadcast(lambda_2_);
  real_t* s = ZeroedScratch(aligned_k);
 for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      Float4 sum = Float4::Load(s+d);
      Float4 const weight = Float4::Load(w+d);
      MulAdd(weight, val, sum).Store(s+d);
    }
  }
  for (const Node& entry : *row) {
    index_t j1 = entry.feat_id;
    // To avoid unseen feature
    if (j1 >= num_feat) continue;
    real_t v1 = entry.feat_val;
    real_t *w_base = model.GetParameter_v() + j1 * align0;
    Float4 val = Float4::Broadcast(v1*norm);
    Float4 pgv = pg_all * val;
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      real_t* w = w_base + d;
      real_t* wg = w_base + aligned_k + d;
      real_t* z = w_base + aligned_k*2 + d;
      Float4 sum = Float4::Load(s+d);
      Float4 weight = Float4::Load(w);
      Float4 weight_grad = Float4::Load(wg);
      Float4 z_val = Float4::Load(z);
      Float4 grad = MulAdd(l2, weight, pgv * NegMulAdd(weight, val, sum));
      Float4 grad_sq = grad * grad;
      Float4 sigma = (Sqrt(weight_grad + grad_sq)
                      - Sqrt(weight_grad)) * inv_alpha;
      z_val = NegMulAdd(sigma, weight, z_val + grad);
      z_val.Store(z);
      weight_grad = weight_grad + grad_sq;
      weight_grad.Store(wg);
      // Update w. Where |z| is inside the l1 band the weight is driven to
      // zero, so the branch becomes a select over the whole vector.
      Float4 numerator = CopySign(l1, z_val) - z_val;
      Float4 denominator = MulAdd(beta + Sqrt(weight_grad), inv_alpha, l2);
      IfThenZeroElse(Abs(z_val) <= l1,
                     numerator / denominator).Store(w);
    }
  }
}

} // namespace xLearn
