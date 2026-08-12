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
This file defines the basic data structures.
*/

#ifndef XLEARN_DATA_DATA_STRUCTURE_H_
#define XLEARN_DATA_DATA_STRUCTURE_H_

#include <vector>
#include <algorithm>

#include "src/base/common.h"
#include "src/base/file_util.h"
#include "src/base/stl-util.h"

namespace xLearn {

//------------------------------------------------------------------------------
// We use 32-bits float to store the real number during computation, 
// such as the model parameter and the gradient.
//------------------------------------------------------------------------------
typedef float real_t;

//------------------------------------------------------------------------------
// We use 32-bits unsigned integer to store the index 
// of the feature and the model parameters.
//------------------------------------------------------------------------------
typedef uint32 index_t;

//------------------------------------------------------------------------------
// We use SIMD to accelerate our training, and hence some
// parameters will be aligned.
//------------------------------------------------------------------------------
const int kAlign = 4;
const int kAlignByte = kAlign * sizeof(real_t);

//------------------------------------------------------------------------------
// What the latent factors are allocated on. A cache line rather than one
// vector's worth: the wide latent path issues 32-byte loads, and starting the
// array at merely kAlignByte would put every other one across a line boundary
// -- which costs more than the wider vector saves.
//------------------------------------------------------------------------------
const int kCacheLineByte = 64;

//------------------------------------------------------------------------------
// MetricInfo stores the evaluation metric information, which
// will be printed for users during the training.
//------------------------------------------------------------------------------
struct MetricInfo {
  real_t loss_val;    /* Loss value */
  real_t metric_val;  /* Metric value */
};

//------------------------------------------------------------------------------
// RowRef is a borrowed view of one row's features: three parallel columns and
// the length they share. It owns nothing and is passed by value.
//
// A column the data never varies is not stored at all: a libsvm file carries
// no fields, and one-hot data carries no values but 1.0. Reading through
// val() and field() is what lets one set of kernels serve both shapes -- and
// what lets a linear model touch four bytes per feature where the field, the
// feature and the value stored together cost twelve, two thirds of which it
// never read.
//
// Feature and field ids share one array rather than getting one each, because
// nothing reads a field without also reading its feature. Separating them
// turned FFM's one random access per row into two into distant arrays, which
// measured 12-14% slower even though it moved fewer bytes.
//
// A RowRef points into vectors that grow, so it is invalidated by the next
// AddNode() on the matrix it came from.
//------------------------------------------------------------------------------
struct RowRef {
  /* Feature ids, each followed by its field id when stride is 2 */
  const index_t* keys = nullptr;
  const real_t* vals = nullptr;  /* len entries, or null when all are 1.0 */
  index_t len = 0;
  /* 2 when the data carries fields, 1 when it does not */
  index_t stride = 1;

  inline index_t feat(index_t i) const { return keys[i * stride]; }
  inline index_t field(index_t i) const {
    return stride == 2 ? keys[i * stride + 1] : 0;
  }
  inline real_t val(index_t i) const { return vals == nullptr ? 1.0f : vals[i]; }
};

//------------------------------------------------------------------------------
// Everything about one row except its features: its label, its normalizer, and
// where in the feature columns it lives.
//
// This is what a caller walking the rows in an order of its own carries, in
// place of an index into the matrix. An index costs four random accesses per
// row -- into Y, into norm, and into both ends of offset -- where a gathered
// array of these is read straight through and leaves only the features random.
// On a shuffled FFM epoch that difference measured 174 ms against 115.
//------------------------------------------------------------------------------
struct RowMeta {
  index_t begin;
  index_t len;
  real_t y;
  real_t norm;
};

//------------------------------------------------------------------------------
// One row's features, owned. DMatrix hands out RowRefs into its own columns;
// this is for the callers that have a row and no matrix to put it in -- the
// score tests and benchmarks, and scoring a single example.
//------------------------------------------------------------------------------
class RowBuffer {
 public:
  void Add(index_t feat_id, real_t feat_val, index_t field_id = 0) {
    keys_.push_back(feat_id);
    keys_.push_back(field_id);
    vals_.push_back(feat_val);
  }

  void Clear() {
    keys_.clear();
    vals_.clear();
  }

  // Always stride 2 -- a hand-built row is a handful of nodes in a test or a
  // benchmark, so the storage never matters and keeping one shape here keeps
  // the conversion trivial.
  operator RowRef() const {
    RowRef row;
    row.keys = keys_.data();
    row.vals = vals_.data();
    row.len = vals_.size();
    row.stride = 2;
    return row;
  }

 private:
  std::vector<index_t> keys_;
  std::vector<real_t> vals_;
};

//------------------------------------------------------------------------------
// DMatrix holds a run of rows in compressed sparse row form: an offset array
// that cuts three parallel feature columns into rows.
//
// The columns are separate rather than interleaved because the three model
// families do not read the same ones -- a linear or FM model never looks at a
// field. A column the data never varies is not stored at all, and Row() hands
// out a null pointer for it. Together that is the difference between ~44 and
// ~132 bytes of node per row on libsvm data, which is the difference between
// an epoch streaming out of cache and out of DRAM.
//
// We can use the DMatrix like this:
//
//    DMatrix matrix;
//    for (int i = 0; i < 10; ++i) {
//      matrix.AddRow();
//      matrix.Y[i] = 0;
//      matrix.AddNode(i, feat_id, feat_val, field_id);
//    }
//
//    /* Serialize and Deserialize */
//    matrix.Serialize("/tmp/test.bin");
//    DMatrix new_matrix;
//    new_matrix.Deserialize("/tmp/test.bin");
//
//    /* We can access the matrix */
//    for (index_t i = 0; i < matrix.row_length; ++i) {
//      ... matrix.Y[i] ...          /* access y */
//      RowRef r = matrix.Row(i);
//      for (index_t j = 0; j < r.len; ++j) {
//        ... r.feat(j) ... r.val(j) ... r.field(j) ...
//      }
//    }
//
//    /* We can also get the max index of feature or field */
//    index_t max_feat = matrix.MaxFeat();
//    index_t max_field = matrix.MaxField();
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// A binary cache carries no schema, so nothing in the file says which layout
// wrote it, and the only other gate is a hash of the *text* the cache was
// built from -- which a cache written by an older build passes. The magic
// comes first so that such a file fails on its opening bytes rather than
// being read as this layout's rows; the version is bumped whenever the body
// below changes. A reader that does not recognize either regenerates.
//------------------------------------------------------------------------------
const uint64 kDMatrixMagic = 0x584C4E5F444D5458ULL;  // "XTMD_NLX" on disk

// Bump this whenever anything below Serialize()'s header changes -- the field
// order, a column's encoding, key_stride's meaning. A stale cache whose magic
// and version still match is read as the current layout and takes the process
// down with it, which is not a hypothetical: it happened once during the move
// to interleaved keys, because the body changed and this did not.
const uint32 kDMatrixVersion = 2;

struct DMatrix {
  DMatrix() { }

  // The view of row i. Invalidated by the next AddNode().
  inline RowRef Row(index_t i) const {
    return Row(Meta(i));
  }

  // The view of the row a RowMeta was taken from.
  inline RowRef Row(const RowMeta& meta) const {
    RowRef r;
    r.keys = keys.data() + meta.begin * key_stride;
    r.vals = vals.empty() ? nullptr : vals.data() + meta.begin;
    r.len = meta.len;
    r.stride = key_stride;
    return r;
  }

  inline RowMeta Meta(index_t i) const {
    RowMeta m;
    m.begin = offset[i];
    m.len = offset[i+1] - m.begin;
    m.y = Y[i];
    m.norm = norm[i];
    return m;
  }

  // Start fetching the columns of the row a RowMeta names, without waiting.
  //
  // A shuffled epoch reads its metadata in order and its columns at random,
  // so this is the fetch that has nowhere to hide: issued far enough ahead,
  // the line has arrived by the time the row is scored. A hint only -- the
  // epoch trains bit-identically with it removed.
  inline void PrefetchRow(const RowMeta& meta) const {
    Prefetch(keys.data() + meta.begin * key_stride);
    if (!vals.empty()) {
      Prefetch(vals.data() + meta.begin);
    }
  }

  // Every row's metadata in one array, for a caller that wants to visit the
  // rows in an order of its own -- it permutes this and walks it, rather than
  // permuting indices and paying to look each one up.
  void GatherRows(std::vector<RowMeta>* rows) const {
    rows->resize(row_length);
    for (index_t i = 0; i < row_length; ++i) {
      (*rows)[i] = Meta(i);
    }
  }

  // Empty the matrix but keep the buffers, so a reader that parses block after
  // block into the same matrix does not hand a large arena back to the
  // allocator and immediately grow it again.
  void Clear() {
    row_length = 0;
    offset.assign(1, 0);
    keys.clear();
    vals.clear();
    key_stride = 1;
    Y.clear();
    norm.clear();
    max_feat = 0;
    max_field = 0;
    has_vals = false;
  }

  // Empty the matrix and release the buffers.
  void Reset() {
    *this = DMatrix();
  }

  // Reserve for a matrix of about this shape. The callers that know the exact
  // size -- the binary cache, and the Python arrays -- avoid the repeated
  // reallocation and copy that growing a multi-gigabyte column otherwise costs.
  void Reserve(index_t rows, index_t nodes) {
    offset.reserve(rows + 1);
    Y.reserve(rows);
    norm.reserve(rows);
    keys.reserve(nodes * key_stride);
  }

  // Open a new row. Rows are built in order, one at a time.
  void AddRow() {
    Y.push_back(0);
    // 1.0 by default, which means no instance-wise normalization
    norm.push_back(1.0);
    offset.push_back(NodeCount());
    row_length++;
  }

  // Add a node to the row AddRow() last opened.
  // We don't use the 'field' by default because it
  // will only be used in the ffm tasks.
  void AddNode(index_t row_id,
               index_t feat_id,
               real_t feat_val,
               index_t field_id = 0) {
    CHECK_EQ(row_id, row_length - 1);
    // A column materializes the moment an entry disagrees with the value its
    // absence stands for. Backfilling costs one pass over what has been read
    // so far, at most once per matrix, and on one-hot data never.
    //
    // Tracked by a flag rather than by the column being non-empty, because the
    // very first node of a matrix materializes a column when there is nothing
    // yet to backfill -- which would leave it empty and materialized at once.
    if (!has_vals && feat_val != 1.0f) {
      has_vals = true;
      vals.assign(NodeCount(), 1.0f);
    }
    if (key_stride == 1 && field_id != 0) {
      // Widen the key array in place: every feature id so far gains a zero
      // field beside it.
      std::vector<index_t> widened;
      widened.reserve((keys.size() + 1) * 2);
      for (size_t i = 0; i < keys.size(); ++i) {
        widened.push_back(keys[i]);
        widened.push_back(0);
      }
      keys.swap(widened);
      key_stride = 2;
    }
    keys.push_back(feat_id);
    if (key_stride == 2) { keys.push_back(field_id); }
    if (has_vals) { vals.push_back(feat_val); }
    offset.back() = NodeCount();
    if (feat_id > max_feat) { max_feat = feat_id; }
    if (field_id > max_field) { max_field = field_id; }
  }

  // The hash value is used to identify the difference
  // between two data matrix, and it can be generated by HashFile() 
  // method (in file_util.h) and this value will be used when reading 
  // txt data from disk file. We can cache the binary data in disk file 
  // to accelerate the reading of disk file.
  void SetHash(uint64 hash_1, uint64 hash_2) {
    hash_value_1 = hash_1;
    hash_value_2 = hash_2;
  }

  // Copy another data matrix to this matrix.
  void CopyFrom(const DMatrix* matrix) {
    CHECK_NOTNULL(matrix);
    *this = *matrix;
  }

  // Reads the header a binary cache opens with, returning false unless it is
  // one of ours at a version we still understand. Shared with hash_binary() in
  // the reader, so that the two cannot disagree about where the header ends.
  static bool ReadHeader(FILE* file, uint64* hash_1, uint64* hash_2) {
    uint64 magic = 0;
    uint32 version = 0;
    *hash_1 = 0;
    *hash_2 = 0;
    if (ReadDataFromDisk(file, (char*)&magic, sizeof(magic)) != sizeof(magic) ||
        magic != kDMatrixMagic) {
      return false;
    }
    if (ReadDataFromDisk(file, (char*)&version, sizeof(version))
            != sizeof(version) ||
        version != kDMatrixVersion) {
      return false;
    }
    return ReadDataFromDisk(file, (char*)hash_1, sizeof(*hash_1))
               == sizeof(*hash_1) &&
           ReadDataFromDisk(file, (char*)hash_2, sizeof(*hash_2))
               == sizeof(*hash_2);
  }

  // Serialize current DMatrix to disk file.
  void Serialize(const std::string& filename) {
    CHECK_NE(filename.empty(), true);
    CHECK_EQ(row_length + 1, offset.size());
    CHECK_EQ(row_length, Y.size());
    CHECK_EQ(row_length, norm.size());
#ifndef _MSC_VER
    FILE* file = OpenFileOrDie(filename.c_str(), "w");
#else
    FILE* file = OpenFileOrDie(filename.c_str(), "wb");
#endif
    WriteDataToDisk(file, (char*)&kDMatrixMagic, sizeof(kDMatrixMagic));
    WriteDataToDisk(file, (char*)&kDMatrixVersion, sizeof(kDMatrixVersion));
    WriteDataToDisk(file, (char*)&hash_value_1, sizeof(hash_value_1));
    WriteDataToDisk(file, (char*)&hash_value_2, sizeof(hash_value_2));
    WriteDataToDisk(file, (char*)&row_length, sizeof(row_length));
    WriteDataToDisk(file, (char*)&max_feat, sizeof(max_feat));
    WriteDataToDisk(file, (char*)&max_field, sizeof(max_field));
    WriteDataToDisk(file, (char*)&has_label, sizeof(has_label));
    // An elided column writes as a zero-length vector and reads back empty,
    // so the elision round-trips without a flags word.
    WriteDataToDisk(file, (char*)&key_stride, sizeof(key_stride));
    WriteVectorToFile(file, offset);
    WriteVectorToFile(file, keys);
    WriteVectorToFile(file, vals);
    WriteVectorToFile(file, Y);
    WriteVectorToFile(file, norm);
    Close(file);
  }

  // Deserialize the DMatrix from disk file.
  void Deserialize(const std::string& filename) {
    CHECK(!filename.empty());
    this->Reset();
#ifndef _MSC_VER
    FILE* file = OpenFileOrDie(filename.c_str(), "r");
#else
    FILE* file = OpenFileOrDie(filename.c_str(), "rb");
#endif
    CHECK(ReadHeader(file, &hash_value_1, &hash_value_2));
    ReadDataFromDisk(file, (char*)&row_length, sizeof(row_length));
    ReadDataFromDisk(file, (char*)&max_feat, sizeof(max_feat));
    ReadDataFromDisk(file, (char*)&max_field, sizeof(max_field));
    ReadDataFromDisk(file, (char*)&has_label, sizeof(has_label));
    ReadDataFromDisk(file, (char*)&key_stride, sizeof(key_stride));
    ReadVectorFromFile(file, offset);
    ReadVectorFromFile(file, keys);
    ReadVectorFromFile(file, vals);
    ReadVectorFromFile(file, Y);
    ReadVectorFromFile(file, norm);
    Close(file);
    // What a truncated or foreign file is actually caught by: the header only
    // says the layout was ours, not that the body survived.
    CHECK_EQ(row_length + 1, offset.size());
    CHECK_EQ(offset.front(), 0);
    CHECK(key_stride == 1 || key_stride == 2);
    CHECK_EQ(offset.back() * key_stride, keys.size());
    CHECK(vals.empty() || vals.size() == offset.back());
    CHECK_EQ(row_length, Y.size());
    CHECK_EQ(row_length, norm.size());
    has_vals = !vals.empty();
  }

  // The max index of feature or field in current data matrix, used to size the
  // model parameters. Kept as the columns grow, because the alternative is a
  // full pass over the arena once per sampled batch.
  inline index_t MaxFeat() const { return max_feat; }
  inline index_t MaxField() const { return max_field; }

  /* The DMatrix has a hash value that is generated
  from the TXT file. These two values are used to check 
  whether we can use binary file to speedup data reading */
  uint64 hash_value_1 = 0;
  uint64 hash_value_2 = 0;
  /* Row length of current matrix */
  index_t row_length = 0;
  /* Row i owns the columns over [offset[i], offset[i+1]). Always row_length+1
  long, so a row's length is one subtraction and the last entry is the node
  count. */
  std::vector<index_t> offset{0};
  /* Feature ids, each followed by its field id when key_stride is 2. One
  array rather than two because nothing reads a field without its feature. */
  std::vector<index_t> keys;
  /* Empty when every value is 1.0, and empty means exactly that */
  std::vector<real_t> vals;
  /* (0 or -1) for negative and (+1) for positive
  examples, and others value for regression */
  std::vector<real_t> Y;
  /* Used for instance-wise normalization */
  std::vector<real_t> norm;
  /* If current dataset has label y */
  bool has_label = true;
  index_t max_feat = 0;
  index_t max_field = 0;
  /* Whether the optional columns are being stored. Only meaningful while a
  matrix is being built; a loaded one infers it from what the file carried. */
  bool has_vals = false;
  /* 2 once any node carried a nonzero field, 1 until then */
  index_t key_stride = 1;

  // Nodes added so far, which is what offset counts.
  inline index_t NodeCount() const { return keys.size() / key_stride; }
};

}  // namespace xLearn

#endif  // XLEARN_DATA_DATA_STRUCTURE_H_
