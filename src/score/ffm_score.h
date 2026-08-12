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
This file defines the FFMScore (field-aware factorization machine) class.
*/

#ifndef XLEARN_LOSS_FFM_SCORE_H_
#define XLEARN_LOSS_FFM_SCORE_H_

#include "src/base/common.h"
#include "src/score/score_function.h"

namespace xLearn {

//------------------------------------------------------------------------------
// FFMScore is used to implement field-aware factorization machines,
// in which the score function is:
//   y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Here leave out the bias and linear term.
//------------------------------------------------------------------------------
class FFMScore : public Score {
public:
 // One row's usable features, with the address arithmetic each takes part in
 // resolved once. Defined in the .cc, beside the kernels that walk it.
 struct Term;

 // Constructor and Destructor
 FFMScore() { }
 ~FFMScore() { }

 // Given one example and current model, this method
 // returns the ffm score.
 real_t CalcScore(RowRef row,
                  Model& model,
                  real_t norm = 1.0);

 // Calculate gradient and update current
 // model parameters.
 void CalcGrad(RowRef row,
               Model& model,
               real_t pg,
               real_t norm = 1.0);

 // PrefetchParams() is deliberately not implemented. A feature's blocks for
 // every field span eleven cache lines on production data, so fetching the
 // first buys nothing, and the pairs then read them at scattered field
 // offsets: measured flat on a 104 MB model, where FM's gained 23%.

 // Scoring resolves the row into terms and the gradient wants exactly those,
 // over latent blocks the linear update cannot have moved -- so the split path
 // resolves them a second time for nothing. That measured 7% of an epoch.
 bool PrefersFusedStep() const { return true; }

 real_t Step(RowRef row,
             Model& model,
             real_t norm,
             PartialGrad partial_grad,
             void* context);

 protected:
  // Dispatch the latent update on the optimizer, for a row already resolved.
  void latent_grad(RowRef row,
                   const std::vector<Term>& terms,
                   Model& model,
                   real_t pg,
                   real_t norm);

  // Calculate gradient and update model using sgd
  void calc_grad_sgd(RowRef row,
                     const std::vector<Term>& terms,
                     Model& model,
                     real_t pg,
                     real_t norm);

  // Calculate gradient and update model using adagrad
  void calc_grad_adagrad(RowRef row,
                         const std::vector<Term>& terms,
                         Model& model,
                         real_t pg,
                         real_t norm);

  // Calculate gradient and update model using ftrl
  void calc_grad_ftrl(RowRef row,
                      const std::vector<Term>& terms,
                      Model& model,
                      real_t pg,
                      real_t norm);

 private:
  DISALLOW_COPY_AND_ASSIGN(FFMScore);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_FFM_SCORE_H_
