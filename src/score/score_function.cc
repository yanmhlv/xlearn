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
This file is the implementation of the base Score class.
*/

#include "src/base/stringprintf.h"
#include "src/score/score_function.h"
#include "src/score/linear_score.h"
#include "src/score/fm_score.h"
#include "src/score/ffm_score.h"

namespace xLearn {

// Reported rather than fatal: a mismatch here is reached by a plausible
// command line -- a pre-trained model plus a different -p -- so the caller
// gets to print it the way it prints every other bad argument.
std::string Score::ModelMismatch(Model& model) const {
  const index_t want = AuxiliarySizeFor(opt_);
  const index_t got = model.GetAuxiliarySize();
  if (got == want) return std::string();
  return StringPrintf(
      "This model keeps %u auxiliary planes per weight and the chosen "
      "optimizer produces %u. A model carries the gradient cache of the "
      "optimizer that trained it, so it can only be used with that optimizer. "
      "Retrain, or run with the optimizer this model was trained with.",
      got, want);
}

void Score::CheckModel(Model& model) const {
  const std::string problem = this->ModelMismatch(model);
  if (!problem.empty()) {
    LOG(FATAL) << problem;
  }
}

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_score_registry, Score);
REGISTER_SCORE("linear", LinearScore);
REGISTER_SCORE("fm", FMScore);
REGISTER_SCORE("ffm", FFMScore);

}  // namespace xLearn
