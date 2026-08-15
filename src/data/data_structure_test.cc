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
This file tests data_structure.h file.
*/

#include "gtest/gtest.h"

#include "src/data/data_structure.h"

namespace xLearn {

const size_t kLength = 10;

TEST(DMATRIX_TEST, Reserve) {
  DMatrix matrix;
  matrix.Reserve(kLength, kLength * 4);
  // Reserving buys capacity, not rows.
  EXPECT_EQ(matrix.row_length, 0);
  EXPECT_EQ(matrix.Y.empty(), true);
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.AddNode(i, i, 1.0, 0);
  }
  EXPECT_EQ(matrix.row_length, kLength);
  EXPECT_EQ(matrix.offset.size(), kLength + 1);
}

TEST(DMATRIX_TEST, Reset) {
  DMatrix matrix;
  matrix.Reset();
  EXPECT_EQ(matrix.has_label, true);
  EXPECT_EQ(matrix.hash_value_1, 0);
  EXPECT_EQ(matrix.hash_value_2, 0);
  EXPECT_EQ(matrix.row_length, 0);
  EXPECT_EQ(matrix.Y.empty(), true);
  EXPECT_EQ(matrix.norm.empty(), true);
  EXPECT_EQ(matrix.keys.empty(), true);
}

TEST(DMATRIX_TEST, AddData) {
  DMatrix matrix;
  matrix.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.Y[i] = 1.0;
    matrix.norm[i] = 1.0;
    matrix.AddNode(i, i, 66.66, i);
  }
  matrix.SetHash(1234, 5678);
  for (size_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(matrix.row_length, kLength);
    EXPECT_FLOAT_EQ(matrix.Y[i], 1.0);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.0);
    RowRef row = matrix.Row(i);
    for (index_t j = 0; j < row.len; ++j) {
      EXPECT_EQ(row.feat(j), i);
      EXPECT_EQ(row.field(j), i);
      EXPECT_FLOAT_EQ(row.val(j), 66.66);
    }
  }
  EXPECT_EQ(matrix.hash_value_1, 1234);
  EXPECT_EQ(matrix.hash_value_2, 5678);
}

TEST(DMATRIX_TEST, Serialize_and_Deserialize) {
  DMatrix matrix;
  matrix.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  matrix.SetHash(1234, 5678);
  // Serialize
#ifndef _MSC_VER
  matrix.Serialize("/tmp/test.bin");
#else
  matrix.Serialize("../../test.bin");
#endif
  matrix.Reset();
  EXPECT_EQ(matrix.has_label, true);
  EXPECT_EQ(matrix.hash_value_1, 0);
  EXPECT_EQ(matrix.hash_value_2, 0);
  EXPECT_EQ(matrix.row_length, 0);
  EXPECT_EQ(matrix.Y.empty(), true);
  EXPECT_EQ(matrix.norm.empty(), true);
  EXPECT_EQ(matrix.keys.empty(), true);
  // Deserialize
#ifndef _MSC_VER
  matrix.Deserialize("/tmp/test.bin");
#else
  matrix.Deserialize("../../test.bin");
#endif
  EXPECT_EQ(matrix.row_length, kLength);
  EXPECT_EQ(matrix.hash_value_1, 1234);
  EXPECT_EQ(matrix.hash_value_2, 5678);
  EXPECT_EQ(matrix.has_label, true);
  for (size_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(matrix.Y[i], i);
    EXPECT_EQ(matrix.norm[i], 0.25);
    RowRef row = matrix.Row(i);
    for (index_t j = 0; j < row.len; ++j) {
      EXPECT_EQ(row.field(j), i);
      EXPECT_EQ(row.feat(j), i);
      EXPECT_FLOAT_EQ(row.val(j), 2.5);
    }
  }
#ifndef _MSC_VER
  RemoveFile("/tmp/test.bin");
#else
  RemoveFile("../../test.bin");
#endif
}

#ifndef _MSC_VER
// A cache truncated inside its last vector passed every check below: each one
// compares sizes, and the sizes come from length fields written before the
// truncation point, while the tail that never arrived reads back as zeros.
// norm is the last vector written, and a row at norm = 0 scores and gradients
// as though it carried no features at all -- so the run trains, finishes, and
// is quietly worse.
TEST(DMATRIX_TEST, Deserialize_rejects_truncation) {
  const char* filename = "/tmp/test_truncated.bin";
  DMatrix matrix;
  matrix.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  matrix.SetHash(1234, 5678);
  matrix.Serialize(filename);

  FILE* file = OpenFileOrDie(filename, "r");
  uint64 full_size = GetFileSize(file);
  Close(file);
  // One element inside the norm payload, past every length field that would
  // otherwise catch it.
  ASSERT_EQ(truncate(filename, full_size - sizeof(real_t)), 0);

  DMatrix truncated;
  EXPECT_DEATH(truncated.Deserialize(filename), "");
  RemoveFile(filename);
}

// Serialize publishes by rename, so a reader never opens a partial file and
// nothing is left under the pending name.
TEST(DMATRIX_TEST, Serialize_leaves_no_pending_file) {
  const std::string filename("/tmp/test_pending.bin");
  DMatrix matrix;
  matrix.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  matrix.SetHash(1234, 5678);
  matrix.Serialize(filename);

  EXPECT_TRUE(FileExist(filename.c_str()));
  EXPECT_FALSE(FileExist(PendingName(filename).c_str()));
  RemoveFile(filename.c_str());
}
#endif

// Publishing has to replace a cache that is already there, which is the case a
// first run on a clean machine never reaches and every re-run does. It is also
// where the two platforms disagree: POSIX rename() overwrites and the Windows
// CRT refuses, so this runs everywhere rather than under the guard above.
TEST(DMATRIX_TEST, Serialize_replaces_an_existing_file) {
#ifndef _MSC_VER
  const std::string filename("/tmp/test_overwrite.bin");
#else
  const std::string filename("../../test_overwrite.bin");
#endif
  DMatrix first;
  first.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    first.AddRow();
    first.AddNode(i, i, 2.5, i);
    first.Y[i] = i;
    first.norm[i] = 0.25;
  }
  first.SetHash(1234, 5678);
  first.Serialize(filename);

  DMatrix second;
  second.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    second.AddRow();
    second.AddNode(i, i, 7.5, i);
    second.Y[i] = i;
    second.norm[i] = 0.5;
  }
  second.SetHash(8765, 4321);
  second.Serialize(filename);

  DMatrix loaded;
  loaded.Deserialize(filename);
  EXPECT_EQ(loaded.hash_value_1, 8765);
  EXPECT_EQ(loaded.hash_value_2, 4321);
  for (size_t i = 0; i < kLength; ++i) {
    EXPECT_FLOAT_EQ(loaded.norm[i], 0.5);
    RowRef row = loaded.Row(i);
    for (index_t j = 0; j < row.len; ++j) {
      EXPECT_FLOAT_EQ(row.val(j), 7.5);
    }
  }
  RemoveFile(filename.c_str());
}

TEST(DMATRIX_TEST, Find_Max_Feat_and_Field) {
  DMatrix matrix;
  matrix.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  matrix.SetHash(1234, 5678);
  EXPECT_EQ(matrix.MaxFeat(), 9);
  EXPECT_EQ(matrix.MaxField(), 9);
}

TEST(DMATRIX_TEST, CopyFrom) {
  DMatrix matrix;
  matrix.Reset();
  for (size_t i = 0; i < kLength; ++i) {
    matrix.AddRow();
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  matrix.SetHash(1234, 5678);
  // Copy matrix
  DMatrix new_matrix;
  new_matrix.CopyFrom(&matrix);
  matrix.Reset();
  // Check
  EXPECT_EQ(new_matrix.row_length, kLength);
  EXPECT_EQ(new_matrix.hash_value_1, 1234);
  EXPECT_EQ(new_matrix.hash_value_2, 5678);
  EXPECT_EQ(new_matrix.has_label, true);
  for (size_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(new_matrix.Y[i], i);
    EXPECT_EQ(new_matrix.norm[i], 0.25);
    RowRef row = new_matrix.Row(i);
    for (index_t j = 0; j < row.len; ++j) {
      EXPECT_EQ(row.field(j), i);
      EXPECT_EQ(row.feat(j), i);
      EXPECT_FLOAT_EQ(row.val(j), 2.5);
    }
  }
}

}  // namespace xLearn
