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
This file is the implementation of SquaredLoss class.
*/

#include "src/loss/squared_loss.h"

namespace xLearn {

// Calculate loss in one thread.
static void sq_evaluate_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    real_t error = (*label)[i] - (*pred)[i];
    (*tmp_sum) += (error*error);
  }
  *tmp_sum *= 0.5;
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
void SquaredLoss::Evaluate(const std::vector<real_t>& pred,
                          const std::vector<real_t>& label) {
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  total_example_ += pred.size();
  // multi-thread training
  std::vector<real_t> sum(threadNumber_, 0);
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(pred.size(), threadNumber_, i);
    size_t end_idx = getEnd(pred.size(), threadNumber_, i);
    pool_->enqueue(std::bind(sq_evaluate_thread,
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

// Score::Step() hands the prediction back here to be turned into the gradient
// it should apply. The context is the label, which is all this needs. The
// reported loss is the squared error; the caller halves the total once.
static real_t sq_partial_grad(real_t pred, void* context, real_t* loss) {
  real_t error = pred - *static_cast<const real_t*>(context);
  *loss = error * error;
  return error;
}

// Calculate gradient in one thread
void sq_gradient_thread(const DMatrix* matrix,
                        Model* model,
                        Score* score_func,
                        bool is_norm,
                        const std::vector<RowMeta>* rows,
                        real_t* sum,
                        index_t start,
                        index_t end) {
  CHECK_GE(end, start);
  *sum = 0;
  bool prefetch_params = WantsParamPrefetch(*model);
  // Which loop, decided once for the whole batch -- see ce_gradient_thread().
  if (score_func->PrefersFusedStep()) {
    for (size_t i = start; i < end; ++i) {
      RowMeta m = NextRow(matrix, rows, score_func, model, prefetch_params,
                            i, end);
      real_t norm = is_norm ? m.norm : 1.0;
      *sum += score_func->Step(matrix->Row(m), *model, norm,
                               sq_partial_grad, &m.y);
    }
    *sum *= 0.5;
    return;
  }
  for (size_t i = start; i < end; ++i) {
    RowMeta m = NextRow(matrix, rows, score_func, model, prefetch_params,
                            i, end);
    RowRef row = matrix->Row(m);
    real_t norm = is_norm ? m.norm : 1.0;
    real_t pred = score_func->CalcScore(row, *model, norm);
    // loss
    real_t error = m.y - pred;
    *sum += (error*error);
    // partial gradient: -error
    real_t pg = pred - m.y;
    score_func->CalcGrad(row, *model, pg, norm);
  }
  *sum *= 0.5;
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
void SquaredLoss::CalcGrad(const DMatrix* matrix,
                           Model& model,
                           const std::vector<RowMeta>* rows) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  int count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  for (int i = 0; i < count; ++i) {
    size_t start = getStart(row_len, count, i);
    size_t end = getEnd(row_len, count, i);
    pool_->enqueue(std::bind(sq_gradient_thread,
                             matrix,
                             &model,
                             score_func_,
                             norm_,
                             rows,
                             &(sum[i]),
                             start,
                             end));
  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

} // namespace xLearn
