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
// std::vector guarantees only 16 bytes, so the run is over-allocated by one
// cache line and handed out from the first line boundary inside it -- the
// wide path loads 32 bytes at a time, and at 16-byte alignment every other
// one would straddle a line.
thread_local std::vector<real_t> scratch;
thread_local real_t* scratch_begin = nullptr;

real_t* ZeroedScratch(index_t size) {
  const index_t kPad = kCacheLineByte / sizeof(real_t);
  if (scratch.size() < size + kPad) {
    scratch.resize(size + kPad);
    scratch_begin = nullptr;
  }
  if (scratch_begin == nullptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(scratch.data());
    uintptr_t pad = (kCacheLineByte - addr % kCacheLineByte) % kCacheLineByte;
    scratch_begin = scratch.data() + pad / sizeof(real_t);
  }
  std::fill_n(scratch_begin, size, 0.0f);
  return scratch_begin;
}

// The scratch as the last ZeroedScratch() caller on this thread left it.
// Step() reaches the latent sum CalcScore() just filled in rather than
// accumulating it a second time.
real_t* FilledScratch() { return scratch_begin; }

// Where a feature's latent block lives, resolved once per row rather than per
// coordinate.
struct LatentLayout {
  index_t num_feat;
  index_t aligned_k;
  index_t align0;
};

LatentLayout LayoutOf(Model& model) {
  index_t aligned_k = model.get_aligned_k();
  return LatentLayout{model.GetNumFeature(), aligned_k,
                      aligned_k * model.GetAuxiliarySize()};
}

// Whether the latent planes can be walked eight lanes at a time. kAlign pads K
// to a multiple of four, so four always divides aligned_k and eight does from
// k = 8 up. Highway caps the request to what the target has, and the loops
// step by the capped Lanes(), which divides the request either way -- so a
// four-lane machine running the wide path stays correct.
inline bool WideLanes(index_t aligned_k) { return aligned_k % 8 == 0; }

// Leaves s = sum_j (v_j * x_j * sqrt(norm)) in the caller's buffer and returns
// sum_{i<j} (v_i x_i)(v_j x_j) * norm, from one walk over the row's latent
// blocks.
//
// The normalizer goes on each factor as sqrt(norm), never on the product.
// Instance normalization scores x/||x||, and norm is 1/||x||^2, so a feature
// scales by sqrt(norm) -- the same sqrt_norm the linear term uses in
// CalcScore() below. A pair then carries exactly one norm. Putting the whole
// norm on each factor squares it, which weights the pairwise term by
// 1/||x||^4 and silently trains a differently-scaled model; every latent
// update below repeats this scaling and has to keep agreeing with it, since
// they read the s this leaves behind.
//
// Each block folds into the pairwise total against the s built from the blocks
// before it, so that sum is formed a pair at a time rather than as the textbook
// half of (s^2 - sum of squares). That identity costs the same three vector
// ops but needs both terms to exist as floats and then cancels them against
// each other: at large latent magnitudes it loses every significant digit,
// and past sqrt(FLT_MAX) it overflows to inf - inf. This form has no such
// range, and it leaves s behind, which is exactly what the gradient needs next.
//
// The pairwise total is only ever wanted summed, so it stays in registers
// rather than getting a scratch plane of its own. s already carries a
// loop-carried dependency through memory, and giving the pair term a second
// one measured 20% on a wide K; kChains independent accumulators keep that
// dependency off the critical path, for the same reason a reduction spreads.
//
// The gradient wants s and not the pairwise total, so it instantiates this
// with kWantPair false and that half compiles away.
template <bool kWantPair, int kChains, int N, int AK>
real_t AccumulateChained(RowRef row,
                         Model& model,
                         real_t norm,
                         const LatentLayout& lay,
                         real_t* s) {
  const index_t aligned_k = AK != 0 ? AK : lay.aligned_k;
  const index_t step = kChains * Vec<N>::Lanes();
  const index_t unrolled_end = aligned_k - aligned_k % step;
  const real_t sqrt_norm = std::sqrt(norm);
  Vec<N> total[kChains];
  for (int c = 0; c < kChains; ++c) {
    total[c] = Vec<N>::Zero();
  }
  for (index_t n = 0; n < row.len; ++n) {
    index_t j1 = row.feat(n);
    // To avoid unseen feature in Prediction
    if (j1 >= lay.num_feat) continue;
    real_t* w = model.GetParameter_v() + j1 * lay.align0;
    Vec<N> val = Vec<N>::Broadcast(row.val(n) * sqrt_norm);
    auto fold = [&](int chain, index_t d) {
      Vec<N> x = Vec<N>::Load(w+d) * val;
      Vec<N> old_s = Vec<N>::Load(s+d);
      if (kWantPair) {
        total[chain] = MulAdd(x, old_s, total[chain]);
      }
      (old_s + x).Store(s+d);
    };
    index_t d = 0;
    for (; d < unrolled_end; d += step) {
      for (int c = 0; c < kChains; ++c) {
        fold(c, d + c * Vec<N>::Lanes());
      }
    }
    // A K too short to fill the chains leaves only this, and one chain is all
    // there is to put it on.
    for (; d < aligned_k; d += Vec<N>::Lanes()) {
      fold(0, d);
    }
  }
  if (!kWantPair) {
    return 0;
  }
  Vec<N> sum = total[0];
  for (int c = 1; c < kChains; ++c) {
    sum = sum + total[c];
  }
  return sum.Sum();
}

// Picks how many accumulators to run, which only the latent width can answer:
// a chain needs a vector block of its own to work on, and how many blocks a
// width has depends on how wide the target's vectors turned out to be. Eight
// lanes on AVX2 leave k=16 with two blocks where four lanes leave it with
// four, and asking for chains the width cannot fill puts every block back on
// the one tail chain -- which is where the dependency showed up in the first
// place.
template <bool kWantPair, int N, int AK>
real_t AccumulateBlocks(RowRef row,
                        Model& model,
                        real_t norm,
                        const LatentLayout& lay,
                        real_t* s) {
  const index_t blocks = (AK != 0 ? AK : lay.aligned_k) / Vec<N>::Lanes();
  if (blocks >= 4) {
    return AccumulateChained<kWantPair, 4, N, AK>(row, model, norm, lay, s);
  }
  if (blocks >= 2) {
    return AccumulateChained<kWantPair, 2, N, AK>(row, model, norm, lay, s);
  }
  return AccumulateChained<kWantPair, 1, N, AK>(row, model, norm, lay, s);
}

// And on the plane length itself, where it is one of the two worth compiling
// for. Almost every model is trained at k <= 8, where the block loop runs
// once: told so it unrolls away, and with it the compare, the branch and the
// pointer bumps that cost as much as the one block they guard.
template <bool kWantPair, int N>
real_t Accumulate(RowRef row,
                  Model& model,
                  real_t norm,
                  const LatentLayout& lay,
                  real_t* s) {
  if (lay.aligned_k == 8) {
    return AccumulateBlocks<kWantPair, N, 8>(row, model, norm, lay, s);
  }
  if (lay.aligned_k == 4) {
    return AccumulateBlocks<kWantPair, N, 4>(row, model, norm, lay, s);
  }
  return AccumulateBlocks<kWantPair, N, 0>(row, model, norm, lay, s);
}

//------------------------------------------------------------------------------
// The latent update, one per optimizer.
//
// Free functions templated on the vector width rather than members, because
// the width is picked per call from aligned_k and a member cannot be
// specialized on it without moving the bodies into the header. The linear and
// bias terms stay with their callers: they are scalar, so the width does not
// reach them.
//------------------------------------------------------------------------------

template <int N, int AK>
void LatentSgdAt(RowRef row,
                 Model& model,
                 real_t pg,
                 real_t norm,
                 const LatentLayout& lay,
                 const real_t* s,
                 real_t learning_rate,
                 real_t regu_lambda) {
  const index_t aligned_k = AK != 0 ? AK : lay.aligned_k;
  const real_t sqrt_norm = std::sqrt(norm);
  Vec<N> pg_all = Vec<N>::Broadcast(pg);
  Vec<N> lr = Vec<N>::Broadcast(learning_rate);
  Vec<N> lamb = Vec<N>::Broadcast(regu_lambda);
  for (index_t n = 0; n < row.len; ++n) {
    index_t j1 = row.feat(n);
    // To avoid unseen feature
    if (j1 >= lay.num_feat) continue;
    real_t* w = model.GetParameter_v() + j1 * lay.align0;
    Vec<N> val = Vec<N>::Broadcast(row.val(n) * sqrt_norm);
    Vec<N> pgv = pg_all * val;
    for (index_t d = 0; d < aligned_k; d += Vec<N>::Lanes()) {
      Vec<N> sum = Vec<N>::Load(s+d);
      Vec<N> weight = Vec<N>::Load(w+d);
      Vec<N> grad = MulAdd(lamb, weight, pgv * NegMulAdd(weight, val, sum));
      NegMulAdd(lr, grad, weight).Store(w+d);
    }
  }
}

template <int N, int AK>
void LatentAdagradAt(RowRef row,
                     Model& model,
                     real_t pg,
                     real_t norm,
                     const LatentLayout& lay,
                     const real_t* s,
                     real_t learning_rate,
                     real_t regu_lambda) {
  const index_t aligned_k = AK != 0 ? AK : lay.aligned_k;
  const real_t sqrt_norm = std::sqrt(norm);
  Vec<N> pg_all = Vec<N>::Broadcast(pg);
  Vec<N> lr = Vec<N>::Broadcast(learning_rate);
  Vec<N> lamb = Vec<N>::Broadcast(regu_lambda);
  for (index_t n = 0; n < row.len; ++n) {
    index_t j1 = row.feat(n);
    // To avoid unseen feature
    if (j1 >= lay.num_feat) continue;
    real_t* w = model.GetParameter_v() + j1 * lay.align0;
    Vec<N> val = Vec<N>::Broadcast(row.val(n) * sqrt_norm);
    Vec<N> pgv = pg_all * val;
    for (index_t d = 0; d < aligned_k; d += Vec<N>::Lanes()) {
      Vec<N> sum = Vec<N>::Load(s+d);
      Vec<N> weight = Vec<N>::Load(w+d);
      Vec<N> weight_grad = Vec<N>::Load(w+aligned_k+d);
      Vec<N> grad = MulAdd(lamb, weight, pgv * NegMulAdd(weight, val, sum));
      weight_grad = MulAdd(grad, grad, weight_grad);
      NegMulAdd(lr, RSqrt(weight_grad) * grad, weight).Store(w+d);
      weight_grad.Store(w+aligned_k+d);
    }
  }
}

template <int N, int AK>
void LatentFtrlAt(RowRef row,
                  Model& model,
                  real_t pg,
                  real_t norm,
                  const LatentLayout& lay,
                  const real_t* s,
                  real_t inv_alpha_val,
                  real_t beta_val,
                  real_t lambda_1_val,
                  real_t lambda_2_val) {
  const index_t aligned_k = AK != 0 ? AK : lay.aligned_k;
  const real_t sqrt_norm = std::sqrt(norm);
  Vec<N> pg_all = Vec<N>::Broadcast(pg);
  Vec<N> inv_alpha = Vec<N>::Broadcast(inv_alpha_val);
  Vec<N> beta = Vec<N>::Broadcast(beta_val);
  Vec<N> l1 = Vec<N>::Broadcast(lambda_1_val);
  Vec<N> l2 = Vec<N>::Broadcast(lambda_2_val);
  for (index_t n = 0; n < row.len; ++n) {
    index_t j1 = row.feat(n);
    // To avoid unseen feature
    if (j1 >= lay.num_feat) continue;
    real_t* w_base = model.GetParameter_v() + j1 * lay.align0;
    Vec<N> val = Vec<N>::Broadcast(row.val(n) * sqrt_norm);
    Vec<N> pgv = pg_all * val;
    for (index_t d = 0; d < aligned_k; d += Vec<N>::Lanes()) {
      real_t* w = w_base + d;
      real_t* wg = w_base + aligned_k + d;
      real_t* z = w_base + aligned_k*2 + d;
      Vec<N> sum = Vec<N>::Load(s+d);
      Vec<N> weight = Vec<N>::Load(w);
      Vec<N> weight_grad = Vec<N>::Load(wg);
      Vec<N> z_val = Vec<N>::Load(z);
      // The loss gradient alone; l2 is the proximal term in the denominator
      // below, and adding it here as well would apply it twice.
      Vec<N> grad = pgv * NegMulAdd(weight, val, sum);
      Vec<N> grad_sq = grad * grad;
      Vec<N> sigma = (Sqrt(weight_grad + grad_sq)
                      - Sqrt(weight_grad)) * inv_alpha;
      z_val = NegMulAdd(sigma, weight, z_val + grad);
      z_val.Store(z);
      weight_grad = weight_grad + grad_sq;
      weight_grad.Store(wg);
      // Update w. Where |z| is inside the l1 band the weight is driven to
      // zero, so the branch becomes a select over the whole vector.
      Vec<N> numerator = CopySign(l1, z_val) - z_val;
      Vec<N> denominator = MulAdd(beta + Sqrt(weight_grad), inv_alpha, l2);
      IfThenZeroElse(Abs(z_val) <= l1,
                     numerator / denominator).Store(w);
    }
  }
}

// Each of the three above under a compile-time plane length, so that the call
// sites stay a choice of width and nothing else. See Accumulate() for why
// the length is worth compiling in.
template <int N>
void LatentSgd(RowRef row,
               Model& model,
               real_t pg,
               real_t norm,
               const LatentLayout& lay,
               const real_t* s,
               real_t learning_rate,
               real_t regu_lambda) {
  if (lay.aligned_k == 8) {
    LatentSgdAt<N, 8>(row, model, pg, norm, lay, s,
                      learning_rate, regu_lambda);
  } else if (lay.aligned_k == 4) {
    LatentSgdAt<N, 4>(row, model, pg, norm, lay, s,
                      learning_rate, regu_lambda);
  } else {
    LatentSgdAt<N, 0>(row, model, pg, norm, lay, s,
                      learning_rate, regu_lambda);
  }
}

template <int N>
void LatentAdagrad(RowRef row,
                   Model& model,
                   real_t pg,
                   real_t norm,
                   const LatentLayout& lay,
                   const real_t* s,
                   real_t learning_rate,
                   real_t regu_lambda) {
  if (lay.aligned_k == 8) {
    LatentAdagradAt<N, 8>(row, model, pg, norm, lay, s,
                          learning_rate, regu_lambda);
  } else if (lay.aligned_k == 4) {
    LatentAdagradAt<N, 4>(row, model, pg, norm, lay, s,
                          learning_rate, regu_lambda);
  } else {
    LatentAdagradAt<N, 0>(row, model, pg, norm, lay, s,
                          learning_rate, regu_lambda);
  }
}

template <int N>
void LatentFtrl(RowRef row,
                Model& model,
                real_t pg,
                real_t norm,
                const LatentLayout& lay,
                const real_t* s,
                real_t inv_alpha_val,
                real_t beta_val,
                real_t lambda_1_val,
                real_t lambda_2_val) {
  if (lay.aligned_k == 8) {
    LatentFtrlAt<N, 8>(row, model, pg, norm, lay, s,
                       inv_alpha_val, beta_val, lambda_1_val, lambda_2_val);
  } else if (lay.aligned_k == 4) {
    LatentFtrlAt<N, 4>(row, model, pg, norm, lay, s,
                       inv_alpha_val, beta_val, lambda_1_val, lambda_2_val);
  } else {
    LatentFtrlAt<N, 0>(row, model, pg, norm, lay, s,
                       inv_alpha_val, beta_val, lambda_1_val, lambda_2_val);
  }
}

} // namespace

// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SIMD to accelerate vector operation.
real_t FMScore::CalcScore(RowRef row,
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
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    t += (row.val(n) * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  t += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  LatentLayout lay = LayoutOf(model);
  real_t* s = ZeroedScratch(lay.aligned_k);
  return t + (WideLanes(lay.aligned_k)
                  ? Accumulate<true, 8>(row, model, norm, lay, s)
                  : Accumulate<true, 4>(row, model, norm, lay, s));
}

// The row's linear weights and latent blocks, on their way before the row is
// scored. Only the first line of each block: the rest follow it in memory,
// which the hardware prefetcher picks up once the kernel reaches them.
void FMScore::PrefetchParams(RowRef row, Model& model) {
  real_t* w = model.GetParameter_w();
  real_t* v = model.GetParameter_v();
  index_t auxiliary_size = model.GetAuxiliarySize();
  LatentLayout lay = LayoutOf(model);
  for (index_t n = 0; n < row.len; ++n) {
    index_t feat_id = row.feat(n);
    if (feat_id >= lay.num_feat) continue;
    Prefetch(w + feat_id * auxiliary_size);
    Prefetch(v + feat_id * lay.align0);
  }
}

// Calculate gradient and update current model parameters.
// Using SIMD to accelerate vector operation.
void FMScore::CalcGrad(RowRef row,
                       Model& model,
                       real_t pg,
                       real_t norm) {
  LatentLayout lay = LayoutOf(model);
  real_t* s = ZeroedScratch(lay.aligned_k);
  if (WideLanes(lay.aligned_k)) {
    Accumulate<false, 8>(row, model, norm, lay, s);
  } else {
    Accumulate<false, 4>(row, model, norm, lay, s);
  }
  this->latent_grad(row, model, pg, norm, s);
}

// Score and update in one pass.
//
// CalcScore() leaves the latent sum in the scratch buffer and the gradient
// needs exactly it, over parameters the linear update below cannot have
// touched -- so the accumulation the split path repeats is skipped here.
real_t FMScore::Step(RowRef row,
                     Model& model,
                     real_t norm,
                     PartialGrad partial_grad,
                     void* context) {
  real_t loss = 0;
  real_t pred = this->CalcScore(row, model, norm);
  real_t pg = partial_grad(pred, context, &loss);
  this->latent_grad(row, model, pg, norm, FilledScratch());
  return loss;
}

// Dispatch the latent update on the optimizer. The linear and bias terms are
// updated inside each, because their update rule differs the same way.
void FMScore::latent_grad(RowRef row,
                          Model& model,
                          real_t pg,
                          real_t norm,
                          const real_t* s) {
  switch (opt_) {
    case OptType::kSgd:
      this->calc_grad_sgd(row, model, pg, norm, s);
      break;
    case OptType::kAdaGrad:
      this->calc_grad_adagrad(row, model, pg, norm, s);
      break;
    case OptType::kFtrl:
      this->calc_grad_ftrl(row, model, pg, norm, s);
      break;
  }
}

// Calculate gradient and update current model using sgd
void FMScore::calc_grad_sgd(RowRef row,
                            Model& model,
                            real_t pg,
                            real_t norm,
                            const real_t* s) {
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
  LatentLayout lay = LayoutOf(model);
  if (WideLanes(lay.aligned_k)) {
    LatentSgd<8>(row, model, pg, norm, lay, s,
                 learning_rate_, regu_lambda_);
  } else {
    LatentSgd<4>(row, model, pg, norm, lay, s,
                 learning_rate_, regu_lambda_);
  }
}

// Calculate gradient and update current model using adagrad
void FMScore::calc_grad_adagrad(RowRef row,
                                Model& model,
                                real_t pg,
                                real_t norm,
                            const real_t* s) {
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
  LatentLayout lay = LayoutOf(model);
  if (WideLanes(lay.aligned_k)) {
    LatentAdagrad<8>(row, model, pg, norm, lay, s,
                     learning_rate_, regu_lambda_);
  } else {
    LatentAdagrad<4>(row, model, pg, norm, lay, s,
                     learning_rate_, regu_lambda_);
  }
}

// Calculate gradient and update current model using ftrl
void FMScore::calc_grad_ftrl(RowRef row,
                             Model& model,
                             real_t pg,
                             real_t norm,
                            const real_t* s) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  this->ftrl_linear_grad(row, model, pg, norm);
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  LatentLayout lay = LayoutOf(model);
  if (WideLanes(lay.aligned_k)) {
    LatentFtrl<8>(row, model, pg, norm, lay, s,
                  inv_alpha_, beta_, lambda_1_, lambda_2_);
  } else {
    LatentFtrl<4>(row, model, pg, norm, lay, s,
                  inv_alpha_, beta_, lambda_1_, lambda_2_);
  }
}

} // namespace xLearn
