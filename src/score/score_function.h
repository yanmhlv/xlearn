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
This file defines the Score class, including linear score,
FM score, FFM score, and etc.
*/

#ifndef XLEARN_LOSS_SCORE_FUNCTION_H_
#define XLEARN_LOSS_SCORE_FUNCTION_H_

#include <cmath>
#include <vector>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/data/model_parameters.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Score is an abstract class, which can be implemented by different
// score functions such as LinearScore (liner_score.h), FMScore (fm_score.h)
// FFMScore (ffm_score.h) etc. On common, we initial a Score function and
// pass its pointer to a Loss class like this:
//
//  Score* score = new FMScore();
//  score->CalcScore(row, model, norm);
//  score->CalcGrad(row, model, pg, norm);
//
// In general, the CalcGrad() will be used in loss function.
//------------------------------------------------------------------------------
class Score {
 public:
  // Which optimizer CalcGrad dispatches to. Resolved once in Initialize:
  // matching the name on every row costs a string compare per example.
  enum class OptType {
    kSgd,
    kAdaGrad,
    kFtrl
  };

  // Constructor and Destructor
  Score() { }
  virtual ~Score() { }

  // Invoke this function before we use this class.
  virtual void Initialize(real_t learning_rate,
                          real_t regu_lambda,
                          real_t alpha,
                          real_t beta,
                          real_t lambda_1,
                          real_t lambda_2,
                          const std::string& opt_type) {
    learning_rate_ = learning_rate;
    regu_lambda_ = regu_lambda;
    alpha_ = alpha;
    beta_ = beta;
    lambda_1_ = lambda_1;
    lambda_2_ = lambda_2;
    // Every ftrl step scales by 1/alpha. Dividing a vector by a scalar that
    // never changes keeps the divider busy for what one reciprocal settles.
    inv_alpha_ = 1.0 / alpha;
    if (opt_type.compare("sgd") == 0) {
      opt_ = OptType::kSgd;
    } else if (opt_type.compare("adagrad") == 0) {
      opt_ = OptType::kAdaGrad;
    } else if (opt_type.compare("ftrl") == 0) {
      opt_ = OptType::kFtrl;
    } else {
      LOG(FATAL) << "Unknow optimization method: " << opt_type;
    }
  }

  // Given one example and current model, this method
  // returns the score
  virtual real_t CalcScore(RowRef row,
                           Model& model,
                           real_t norm = 1.0) = 0;

  // Calculate gradient and update current
  // model parameters
  virtual void CalcGrad(RowRef row,
                        Model& model,
                        real_t pg,
                        real_t norm = 1.0) = 0;

  // Turns a prediction into the partial gradient to apply, and reports the
  // loss it came from. A function pointer and a context rather than a
  // std::function: this runs once per example and must not allocate.
  typedef real_t (*PartialGrad)(real_t pred, void* context, real_t* loss);

  // Whether Step() saves this score function real work.
  //
  // Asked once per batch, not once per row, because the answer decides which
  // loop a Loss runs. FM says yes: its latent sum is the same in both halves,
  // over parameters the linear update cannot have touched, so the split path
  // builds it twice. Linear and FFM say no, and it is not a wash for them --
  // routing a score function with nothing to share through Step() puts an
  // indirect call it cannot inline around a body that does very little, which
  // measured 17% on LR before this existed.
  virtual bool PrefersFusedStep() const { return false; }

  // Score one example and apply the resulting gradient, returning the loss.
  // Only called when PrefersFusedStep() is true.
  virtual real_t Step(RowRef row,
                      Model& model,
                      real_t norm,
                      PartialGrad partial_grad,
                      void* context) {
    real_t loss = 0;
    real_t pg = partial_grad(this->CalcScore(row, model, norm),
                             context, &loss);
    this->CalcGrad(row, model, pg, norm);
    return loss;
  }

 protected:
  // One ftrl-proximal step over the three slots at p -- the weight, the sum
  // of squared gradients, and the dual accumulator -- read into registers and
  // written back once.
  //
  // Taken as references into the parameter array instead, the three alias one
  // another as far as the compiler can prove, so the store to z forces a
  // reload and a second square root of a sum the line above already has.
  void ftrl_update(real_t* p, real_t g) {
    real_t weight = p[0];
    real_t sqrt_old_n = std::sqrt(p[1]);
    real_t n = p[1] + g * g;
    real_t sqrt_n = std::sqrt(n);
    real_t sigma = (sqrt_n - sqrt_old_n) * inv_alpha_;
    real_t z = p[2] + (g - sigma * weight);
    // Both arms, then a select. Which way z falls is a coin toss the branch
    // predictor cannot learn, and the mispredict costs more than the divide
    // it was there to skip.
    real_t sign = std::copysign(1.0f, z);
    real_t weight_next = (sign * lambda_1_ - z) /
                         ((beta_ + sqrt_n) * inv_alpha_ + lambda_2_);
    p[0] = std::fabs(z) <= lambda_1_ ? 0.0f : weight_next;
    p[1] = n;
    p[2] = z;
  }

  // The ftrl update of the linear weights and the bias, shared by all three
  // score functions: the rule is the same wherever a feature has one weight,
  // and only the latent term below it differs.
  void ftrl_linear_grad(RowRef row, Model& model, real_t pg, real_t norm) {
    real_t sqrt_norm = std::sqrt(norm);
    real_t* w = model.GetParameter_w();
    index_t num_feat = model.GetNumFeature();
    for (index_t n = 0; n < row.len; ++n) {
      index_t feat_id = row.feat(n);
      // To avoid unseen feature
      if (feat_id >= num_feat) continue;
      real_t* p = w + feat_id * 3;
      this->ftrl_update(p, lambda_2_ * p[0] + pg * row.val(n) * sqrt_norm);
    }
    // bias
    this->ftrl_update(model.GetParameter_b(), pg);
  }

  real_t learning_rate_;
  real_t regu_lambda_;
  real_t alpha_;
  real_t beta_;
  real_t lambda_1_;
  real_t lambda_2_;
  real_t inv_alpha_;
  OptType opt_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Score);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_score_registry, Score);

#define REGISTER_SCORE(format_name, score_name)             \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_score_registry,                                \
      Score,                                                \
      format_name,                                          \
      score_name)

#define CREATE_SCORE(format_name)                           \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_score_registry,                                \
      format_name)

}  // namespace xLearn

#endif  // XLEARN_LOSS_SCORE_FUNCTION_H_
