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
This file is the implementation of CrossEntropyLoss class.
*/

#include "src/loss/cross_entropy_loss.h"

#include <cmath>
#include <thread>
#include<atomic>

namespace xLearn {

namespace {

// log(1 + exp(-y*pred)) and its partial gradient, both functions of the one
// exp() computed here.
//
// The exponent is taken on the side that cannot overflow: written directly as
// log1p(exp(-y*pred)), a confidently wrong prediction overflows exp() to
// infinity and the epoch's accumulated loss never recovers. Sharing the one
// exp() between the two results is the other half of the reason this is a
// function -- as separate expressions the second call cannot be folded away,
// because exp() sets errno.
inline real_t ce_loss_and_grad(real_t y, real_t pred, real_t* pg) {
  real_t t = y * pred;
  real_t u = std::exp(-std::fabs(t));
  // sigmoid(-|t|), which is the factor the gradient wants where the
  // prediction agrees with the label; its complement is what the other side
  // wants.
  real_t s = u / (1 + u);
  *pg = -y * (t >= 0 ? s : 1 - s);
  return std::log1p(u) + std::max(-t, real_t(0));
}

// The loss on its own, for the evaluation pass. The gradient is dead once
// this is inlined, so asking for it costs nothing and keeps one copy of the
// formula above.
inline real_t ce_loss(real_t y, real_t pred) {
  real_t pg;
  return ce_loss_and_grad(y, pred, &pg);
}

// Score::Step() hands the prediction back here to be turned into the gradient
// it should apply. The context is the label, which is all this needs.
real_t ce_partial_grad(real_t pred, void* context, real_t* loss) {
  real_t pg = 0;
  *loss = ce_loss_and_grad(*static_cast<const real_t*>(context), pred, &pg);
  return pg;
}

} // namespace

// Calculate loss in one thread.
static void ce_evaluate_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    real_t y = (*label)[i] > 0 ? 1.0 : -1.0;
    (*tmp_sum) += ce_loss(y, (*pred)[i]);
  }
}

//------------------------------------------------------------------------------
// Calculate loss in multi-thread:
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void CrossEntropyLoss::Evaluate(const std::vector<real_t>& pred,
                               const std::vector<real_t>& label) {
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  total_example_ += pred.size();
  // multi-thread training
  std::vector<real_t> sum(threadNumber_, 0);
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(pred.size(), threadNumber_, i);
    size_t end_idx = getEnd(pred.size(), threadNumber_, i);
    pool_->enqueue(std::bind(ce_evaluate_thread,
                             &pred,
                             &label,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
  // Accumulate loss
  for (size_t i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}


// Calculate gradient in one thread.
static void ce_gradient_thread(const DMatrix* matrix,
                               Model* model,
                               Score* score_func,
                               bool is_norm,
                               const std::vector<RowMeta>* rows,
                               real_t* sum,
                               size_t start_idx,
                               size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *sum = 0;
  // Which loop, decided once for the whole batch. Step() lets a score
  // function keep what scoring and the gradient share, but it reaches the
  // loss through a pointer it cannot inline -- worth it only where there is
  // something to share.
  if (score_func->PrefersFusedStep()) {
    for (size_t i = start_idx; i < end_idx; ++i) {
      RowMeta m = NextRow(matrix, rows, i, end_idx);
      real_t norm = is_norm ? m.norm : 1.0;
      real_t y = m.y > 0 ? 1.0 : -1.0;
      *sum += score_func->Step(matrix->Row(m), *model, norm,
                               ce_partial_grad, &y);
    }
    return;
  }
  for (size_t i = start_idx; i < end_idx; ++i) {
    RowMeta m = NextRow(matrix, rows, i, end_idx);
    RowRef row = matrix->Row(m);
    real_t norm = is_norm ? m.norm : 1.0;
    real_t pred = score_func->CalcScore(row, *model, norm);
    real_t y = m.y > 0 ? 1.0 : -1.0;
    real_t pg;
    *sum += ce_loss_and_grad(y, pred, &pg);
    score_func->CalcGrad(row, *model, pg, norm);
  }
}

//------------------------------------------------------------------------------
// Calculate gradient in multi-thread
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void CrossEntropyLoss::CalcGrad(const DMatrix* matrix,
                                Model& model,
                                const std::vector<RowMeta>* rows) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  // multi-thread training
  int count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  for (int i = 0; i < count; ++i) {
    index_t start_idx = getStart(row_len, count, i);
    index_t end_idx = getEnd(row_len, count, i);
    pool_->enqueue(std::bind(ce_gradient_thread,
                             matrix,
                             &model,
                             score_func_,
                             norm_,
                             rows,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

} // namespace xLearn
