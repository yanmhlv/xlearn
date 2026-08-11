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
This file benchmarks the Parser classes, which sit on the ingest path of
every training run.
*/

#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "src/base/stringprintf.h"
#include "src/data/data_structure.h"
#include "src/reader/parser.h"

namespace xLearn {
namespace {

const index_t kNumLines = 20000;
const index_t kNumFeat = 30;
const index_t kNumField = 20;

// One buffer per format, built once and parsed over and over. Parse() copies
// each line out before touching it, so the same buffer serves every iteration.
std::string MakeBuffer(const std::string& format) {
  std::string out;
  out.reserve(kNumLines * kNumFeat * 12);
  index_t seed = 1;
  for (index_t line = 0; line < kNumLines; ++line) {
    out += (line % 2 == 0) ? "1" : "0";
    for (index_t i = 0; i < kNumFeat; ++i) {
      seed = seed * 1103515245 + 12345;
      index_t feat_id = (seed >> 8) % 100000;
      real_t value = 1.0 + (seed >> 16) % 100 / 100.0;
      if (format == "libsvm") {
        StringAppendF(&out, " %u:%.4f", feat_id, value);
      } else if (format == "ffm") {
        StringAppendF(&out, " %u:%u:%.4f", i % kNumField, feat_id, value);
      } else {
        StringAppendF(&out, ",%.4f", value);
      }
    }
    out += "\n";
  }
  return out;
}

void RunParse(benchmark::State& state, const std::string& format) {
  static std::string buffer;
  buffer = MakeBuffer(format);

  Parser* parser = nullptr;
  LibsvmParser libsvm;
  FFMParser ffm;
  CSVParser csv;
  if (format == "libsvm") {
    parser = &libsvm;
  } else if (format == "ffm") {
    parser = &ffm;
  } else {
    parser = &csv;
  }
  parser->setLabel(true);
  parser->setSplitor(format == "csv" ? "," : " ");

  DMatrix matrix;
  for (auto _ : state) {
    parser->Parse(&buffer[0], buffer.size(), matrix, true);
    benchmark::DoNotOptimize(matrix.row_length);
  }
  state.SetBytesProcessed(state.iterations() * buffer.size());
  state.SetItemsProcessed(state.iterations() * kNumLines);
}

} // namespace
} // namespace xLearn

int main(int argc, char** argv) {
  for (const char* format : {"libsvm", "ffm", "csv"}) {
    std::string name = std::string("parser/") + format;
    std::string owned = format;
    benchmark::RegisterBenchmark(name,
        [owned](benchmark::State& s) { xLearn::RunParse(s, owned); });
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
