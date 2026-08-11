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
This file benchmarks the SIMD kernels of the FMScore and FFMScore classes.
*/

#include <cstdio>
#include <string>
#include <vector>

#include "src/base/bench.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/score/fm_score.h"
#include "src/score/ffm_score.h"

namespace xLearn {
namespace {

// Every configuration draws from the same pool of rows so that the kernels
// walk over many different latent vectors rather than re-reading one row out
// of L1. The count is a power of two so the cursor can wrap with a mask.
const index_t kNumRows = 1024;
const index_t kNumRowsBig = 16384;

const real_t kLearningRate = 0.2;
const real_t kReguLambda = 0.00002;
const real_t kAlpha = 0.3;
const real_t kBeta = 1.0;
const real_t kLambda1 = 0.00001;
const real_t kLambda2 = 0.00002;
const real_t kModelScale = 0.66;
// A plausible mid-training partial gradient.
const real_t kPartialGrad = 0.15;

index_t AuxiliarySize(const std::string& opt_type) {
  if (opt_type == "sgd") return 1;
  if (opt_type == "adagrad") return 2;
  return 3;
}

// Scatter the feature ids with a linear congruential sequence so the access
// pattern is spread over the feature space instead of running sequentially.
std::vector<SparseRow> MakeRows(index_t num_rows,
                                index_t num_nnz,
                                index_t num_feat,
                                index_t num_field) {
  std::vector<SparseRow> rows(num_rows);
  index_t seed = 1;
  for (index_t r = 0; r < num_rows; ++r) {
    rows[r].resize(num_nnz);
    for (index_t i = 0; i < num_nnz; ++i) {
      seed = seed * 1103515245 + 12345;
      rows[r][i].feat_id = (seed >> 8) % num_feat;
      rows[r][i].field_id = i % num_field;
      rows[r][i].feat_val = 1.0;
    }
  }
  return rows;
}

template <typename ScoreType>
void BenchScore(Benchmark* bench,
                const std::string& kind,
                const std::string& opt_type,
                index_t num_K,
                index_t num_feat,
                index_t num_field,
                index_t num_nnz,
                index_t num_rows,
                const std::string& suffix) {
  Model model;
  model.Initialize(kind, "squared", num_feat, num_field,
                   num_K, AuxiliarySize(opt_type), kModelScale);
  std::vector<SparseRow> rows =
      MakeRows(num_rows, num_nnz, num_feat, num_field);
  const index_t row_mask = num_rows - 1;

  std::string opt = opt_type;
  ScoreType score;
  score.Initialize(kLearningRate, kReguLambda, kAlpha,
                   kBeta, kLambda1, kLambda2, opt);

  char name[128];
  index_t cursor = 0;
  snprintf(name, sizeof(name), "%s/%s/score/k=%u%s",
           kind.c_str(), opt_type.c_str(), num_K, suffix.c_str());
  bench->Run(name,
             [&] {
               DoNotOptimize(score.CalcScore(&rows[cursor & row_mask],
                                            model, 1.0));
               ++cursor;
             },
             [&] { model.Reset(); cursor = 0; });

  snprintf(name, sizeof(name), "%s/%s/grad/k=%u%s",
           kind.c_str(), opt_type.c_str(), num_K, suffix.c_str());
  bench->Run(name,
             [&] {
               score.CalcGrad(&rows[cursor & row_mask], model,
                              kPartialGrad, 1.0);
               ++cursor;
             },
             [&] { model.Reset(); cursor = 0; });
}

} // namespace
} // namespace xLearn

int main() {
  using namespace xLearn;
  Benchmark bench;

  // FM: one field, a wide latent vector, a model that stays in cache.
  const index_t kFMFeat = 10000;
  const index_t kFMNnz = 30;
  BenchScore<FMScore>(&bench, "fm", "sgd", 4, kFMFeat, 1, kFMNnz, kNumRows, "");
  BenchScore<FMScore>(&bench, "fm", "sgd", 16, kFMFeat, 1, kFMNnz, kNumRows, "");
  BenchScore<FMScore>(&bench, "fm", "sgd", 64, kFMFeat, 1, kFMNnz, kNumRows, "");
  BenchScore<FMScore>(&bench, "fm", "adagrad", 16, kFMFeat, 1, kFMNnz, kNumRows, "");
  BenchScore<FMScore>(&bench, "fm", "ftrl", 16, kFMFeat, 1, kFMNnz, kNumRows, "");

  // FM over a feature space too large for the cache, which is the regime a
  // real training run sits in and where memory latency, not SIMD, dominates.
  BenchScore<FMScore>(&bench, "fm", "sgd", 16, 1000000, 1, kFMNnz,
                      kNumRowsBig, "/oom-cache");

  // FFM: a narrow latent vector per field, quadratic in the row length.
  const index_t kFFMFeat = 2000;
  const index_t kFFMField = 20;
  const index_t kFFMNnz = 20;
  BenchScore<FFMScore>(&bench, "ffm", "sgd", 4, kFFMFeat, kFFMField,
                       kFFMNnz, kNumRows, "");
  BenchScore<FFMScore>(&bench, "ffm", "sgd", 16, kFFMFeat, kFFMField,
                       kFFMNnz, kNumRows, "");
  BenchScore<FFMScore>(&bench, "ffm", "adagrad", 4, kFFMFeat, kFFMField,
                       kFFMNnz, kNumRows, "");
  BenchScore<FFMScore>(&bench, "ffm", "ftrl", 4, kFFMFeat, kFFMField,
                       kFFMNnz, kNumRows, "");

  bench.Report();
  return 0;
}
