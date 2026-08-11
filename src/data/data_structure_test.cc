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
