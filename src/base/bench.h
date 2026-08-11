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
This file provides a micro-benchmark harness for the hot kernels.
*/

#ifndef XLEARN_BASE_BENCH_H_
#define XLEARN_BASE_BENCH_H_

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "src/base/common.h"

//------------------------------------------------------------------------------
// Force a value to be materialized so the optimizer cannot delete the
// computation that produced it.
//------------------------------------------------------------------------------
template <typename T>
inline void DoNotOptimize(const T& value) {
  static volatile T sink;
  sink = value;
}

//------------------------------------------------------------------------------
// Benchmark times a callable and reports nanoseconds per call. The iteration
// count is calibrated so that one repetition runs for kTargetMs, and the
// reported figure is the median over kRepetitions, which rejects the
// occasional descheduling spike:
//
//   Benchmark bench;
//   bench.Run("fm/calc_score/k=16",
//             [&] { DoNotOptimize(score.CalcScore(&row, model)); },
//             [&] { model.Reset(); });
//   bench.Report();
//
// The reset callable runs before every repetition and is not timed. Kernels
// that update the model in place need it, otherwise the parameters drift far
// from their initial scale over millions of calls and the timings measure a
// model no training run would ever produce.
//------------------------------------------------------------------------------
class Benchmark {
 public:
  Benchmark() { }

  template <typename Body, typename Reset>
  void Run(const std::string& name, Body body, Reset reset) {
    size_t iters = Calibrate(body, reset);
    std::vector<double> timings;
    for (size_t rep = 0; rep < kRepetitions; ++rep) {
      reset();
      timings.push_back(TimeOne(body, iters) / iters);
    }
    std::sort(timings.begin(), timings.end());
    results_.push_back(Result{name,
                              timings[timings.size()/2],
                              timings.front(),
                              iters});
  }

  template <typename Body>
  void Run(const std::string& name, Body body) {
    Run(name, body, [] { });
  }

  void Report() const {
    printf("%-34s %12s %12s %10s\n",
           "benchmark", "median_ns", "min_ns", "iters");
    for (std::vector<Result>::const_iterator iter = results_.begin();
         iter != results_.end(); ++iter) {
      printf("%-34s %12.2f %12.2f %10zu\n",
             iter->name.c_str(), iter->median_ns,
             iter->min_ns, iter->iters);
    }
  }

 private:
  struct Result {
    std::string name;
    double median_ns;
    double min_ns;
    size_t iters;
  };

  static const size_t kRepetitions = 9;
  static const size_t kTargetMs = 40;

  template <typename Body>
  static double TimeOne(Body body, size_t iters) {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    for (size_t i = 0; i < iters; ++i) {
      body();
    }
    return std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - begin).count();
  }

  template <typename Body, typename Reset>
  static size_t Calibrate(Body body, Reset reset) {
    size_t iters = 1;
    for (;;) {
      reset();
      double elapsed_ns = TimeOne(body, iters);
      if (elapsed_ns > kTargetMs * 1e6) {
        return iters;
      }
      // Jump straight to the count the measured rate implies, but never
      // trust a rate derived from a run too short to time accurately.
      size_t next = elapsed_ns < 1e5
                  ? iters * 16
                  : (size_t)(iters * (kTargetMs * 1e6 / elapsed_ns));
      iters = std::max(next, iters + 1);
    }
  }

  std::vector<Result> results_;

  DISALLOW_COPY_AND_ASSIGN(Benchmark);
};

#endif  // XLEARN_BASE_BENCH_H_
