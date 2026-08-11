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
This file benchmarks the Loss classes over a batch, which is the level the
solver actually drives the score functions at.
*/

#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "src/base/thread_pool.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/loss/squared_loss.h"
#include "src/score/fm_score.h"

namespace xLearn {
namespace {

const index_t kNumRows = 4096;
const index_t kNumFeat = 10000;
const index_t kNumNnz = 30;
const index_t kNumK = 16;
const real_t kModelScale = 0.66;

// One batch and one model, shared by every repetition. Rebuilding them per
// repetition would cost more than the work being measured. The thread pool is
// not cached with them, because it is the axis the benchmarks vary.
struct Workload {
  DMatrix matrix;
  Model model;
  std::vector<real_t> pred;
};

Workload& GetWorkload() {
  static std::unique_ptr<Workload> cached;
  if (cached != nullptr) {
    return *cached;
  }
  cached.reset(new Workload());
  cached->model.Initialize("fm", "squared", kNumFeat, 1, kNumK, 2, kModelScale);
  index_t seed = 1;
  for (index_t r = 0; r < kNumRows; ++r) {
    cached->matrix.AddRow();
    cached->matrix.Y[r] = (r % 2 == 0) ? 1.0 : -1.0;
    cached->matrix.norm[r] = 1.0;
    for (index_t i = 0; i < kNumNnz; ++i) {
      seed = seed * 1103515245 + 12345;
      cached->matrix.AddNode(r, (seed >> 8) % kNumFeat, 1.0);
    }
  }
  cached->pred.resize(kNumRows);
  return *cached;
}

// range(0) is the pool size, range(1) the lock_free flag. Both matter: without
// lock_free, CalcGrad splits the batch into exactly one piece no matter how
// many threads the pool has, so the pool size alone tells you nothing.
template <typename LossType>
void RunCalcGrad(benchmark::State& state) {
  Workload& workload = GetWorkload();
  ThreadPool pool(state.range(0));
  FMScore score;
  std::string opt = "adagrad";
  score.Initialize(0.2, 0.00002, 0.3, 1.0, 0.00001, 0.00002, opt);
  LossType loss;
  loss.Initialize(&score, &pool, true, state.range(1) != 0);
  // Untimed: keeps the parameters from drifting away over many passes.
  workload.model.Reset();
  for (auto _ : state) {
    loss.CalcGrad(&workload.matrix, workload.model);
  }
  state.SetItemsProcessed(state.iterations() * kNumRows);
}

template <typename LossType>
void RunPredict(benchmark::State& state) {
  Workload& workload = GetWorkload();
  ThreadPool pool(state.range(0));
  FMScore score;
  std::string opt = "adagrad";
  score.Initialize(0.2, 0.00002, 0.3, 1.0, 0.00001, 0.00002, opt);
  LossType loss;
  loss.Initialize(&score, &pool, true, false);
  for (auto _ : state) {
    loss.Predict(&workload.matrix, workload.model, workload.pred);
  }
  state.SetItemsProcessed(state.iterations() * kNumRows);
}

} // namespace
} // namespace xLearn

// Parallelism is what the batch level adds over the per-row kernels the score
// benchmarks already cover, so that is the axis these vary. The gradient
// arguments read as {threads, lock_free}.
BENCHMARK(xLearn::RunCalcGrad<xLearn::SquaredLoss>)
    ->Name("loss/squared/grad")
    ->Args({1, 0})->Args({4, 0})->Args({4, 1})->UseRealTime();
BENCHMARK(xLearn::RunCalcGrad<xLearn::CrossEntropyLoss>)
    ->Name("loss/cross_entropy/grad")
    ->Args({1, 0})->Args({4, 0})->Args({4, 1})->UseRealTime();
BENCHMARK(xLearn::RunPredict<xLearn::SquaredLoss>)
    ->Name("loss/squared/predict")->Arg(1)->Arg(4)->UseRealTime();

BENCHMARK_MAIN();
