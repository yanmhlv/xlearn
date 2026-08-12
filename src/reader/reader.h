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
This file defines the Reader class that is responsible for
reading data from data source.
*/

#ifndef XLEARN_READER_READER_H_
#define XLEARN_READER_READER_H_

#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/base/scoped_ptr.h"
#include "src/base/thread_pool.h"
#include "src/data/data_structure.h"
#include "src/reader/parser.h"

namespace xLearn {

const size_t kDefautBlockSize = 500;  // 500 MB

// Fisher-Yates, over the metadata itself. mt19937 looks like the expensive
// choice here and is not: it generates 624 words at a time, so a draw costs a
// few cycles of buffer read, where a small stateless generator pays the full
// latency of its dependent multiplies on every one. Replacing it with
// splitmix64 and a multiply-shift bound took this from 7.3% of an LR epoch to
// 10.8%.
//
// Which partner a step swaps with depends only on the step, never on the
// array, so the draws can run ahead of the swaps -- far enough ahead that the
// partner's line is on its way before the swap reaches it. std::shuffle draws
// and swaps in lockstep and waits on DRAM for every one of them, which cost
// 17% of an LR epoch once the epoch itself got fast.
inline void ShuffleOrder(std::vector<RowMeta>& rows, std::mt19937& rng) {
  const size_t kLookahead = 8;
  size_t n = rows.size();
  if (n < 2) return;
  std::uniform_int_distribution<size_t> draw;
  typedef std::uniform_int_distribution<size_t>::param_type upto;
  size_t pending[kLookahead];
  size_t ahead = n - 1;
  for (size_t k = 0; k < kLookahead && ahead >= 1; ++k, --ahead) {
    pending[k] = draw(rng, upto(0, ahead));
    Prefetch(&rows[pending[k]]);
  }
  for (size_t i = n - 1, k = 0; i >= 1; --i, k = (k + 1) % kLookahead) {
    size_t j = pending[k];
    if (ahead >= 1) {
      pending[k] = draw(rng, upto(0, ahead));
      Prefetch(&rows[pending[k]]);
      --ahead;
    }
    std::swap(rows[i], rows[j]);
  }
}

//------------------------------------------------------------------------------
// Reader is an abstract class which can be implemented in different way,
// such as the InmemReader that reads data from memory, and the OndiskReader
// that reads data from disk file for large-scale machine learning.
//
// We can use the Reader class like this (Pseudo code):
//
//   #include "reader.h"
//
//   /* or new InmemReader() */
//   Reader* reader = new OndiskReader();
//  
//   reader->Initialize(filename = "/tmp/testdata");
//  
//   DMatrix* matrix = nullptr;
// 
//   Loop {
//
//      size_t num_samples = reader->Sample(DMatrix);
//
//      /* The reader will return 0 when reaching the end of
//      data source, and then we can invoke Reset() to return
//      to the beginning of data */
//  
//      if (num_samples == 0) {
//        reader->Reset()
//      }
//
//      /* use data matrix ... */
//
//   }
//
// For now, the Reader can parse three kinds of file format, including
// the libsvm format, the libffm format, and the CSV format.
//------------------------------------------------------------------------------
class Reader {
 public:
  // Constructor and Destructor
  Reader() : 
    shuffle_(false), 
    bin_out_(true),
    block_size_(kDefautBlockSize) {  }
  virtual ~Reader() {  }

  // We need to invoke the Initialize() function before
  // we start to sample data. We can shuffle data before 
  // training, and this is good for SGD.
  virtual void Initialize(const std::string& filename) = 0;
  virtual void Initialize(xLearn::DMatrix* &dmatrix) = 0;

  // Sample data from disk or from memory buffer.
  // Return the number of record in each sampling.
  // Samples() will return 0 when reaching end of the data.
  virtual index_t Samples(DMatrix* &matrix) = 0;

  // Return to the beginning of the data.
  virtual void Reset() = 0;

  // Free the memory of data matrix.
  virtual void Clear() = 0;

  // Return the Reader type, which can be 
  // 'in-memory' or 'on-disk' so far.
  virtual std::string Type() = 0;

  // Set the size of the block buffer.
  inline void SetBlockSize(size_t size) { 
    CHECK_GT(size, 0);
    block_size_ = size; 
  }

  // Whether current dataset has label y ?
  bool inline has_label() { return has_label_; }

  // Do not generate bin file
  void SetNoBin() {
    bin_out_ = false;  
  }

  // Set random seed
  void SetSeed(int seed) {
    rng_.seed(seed);
  }

  // If shuffle data ?
  virtual void SetShuffle(bool shuffle) {
    shuffle_ = shuffle;
  }

  // The rows of the matrix from Samples(), in the order they should be
  // visited, or null to walk it in its own order.
  //
  // Shuffling is a property of the traversal, not of the data: permuting the
  // rows themselves would mean moving every feature of every one of them, once
  // per epoch. What is permuted instead is this array of RowMeta rather than an
  // array of indices, so that the walk over it stays sequential -- see RowMeta.
  // Only the gradient walk takes this: Predict() must leave pred[i] lined up
  // with Y[i].
  virtual const std::vector<RowMeta>* Rows() const { return nullptr; }

 protected:
  /* Input file name */
  std::string filename_;
  /* Parser for a block of memory buffer */
  Parser* parser_;
  /* If this data has label y ?
  This value will be set automatically
  in initialization */
  bool has_label_;
  /* If shuffle data ? */
  bool shuffle_;
  /* Generate bin file ? */
  bool bin_out_;
  /* Split string for data items */
  std::string splitor_;
  /* A block of memory to store the data. Owned: the paths that allocate it and
  the paths that do not both reach the destructor, and only one of them has a
  buffer to release. */
  scoped_array<char> block_;
  /* Block size */
  size_t block_size_;
  /* Draws the shuffle order. Carried across epochs rather than re-seeded at
  each one, so every epoch visits the rows in a different permutation --
  reshuffling is what makes SGD converge faster than a fixed order. */
  std::mt19937 rng_{1};

  // Check current file format and return
  // "libsvm", "ffm", or "csv".
  // Program crashes for unknow format.
  // This function will also check if current
  // data has the label y.
  std::string check_file_format();

  // Find the last '\n' in block and 
  // shrink back file pointer.
  void shrink_block(char* block, size_t* ret, FILE* file);

  // Create parser for different file format
  Parser* CreateParser(const char* format_name) {
    return CREATE_PARSER(format_name);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(Reader);
};

//------------------------------------------------------------------------------
// Sampling data from memory buffer.
// For in-memory sampling, the Reader will automatically convert
// txt data to binary data, and uses this binary data in the next time.
//------------------------------------------------------------------------------
class InmemReader : public Reader {
 public:
  // Constructor and Destructor
  InmemReader() : pos_(0) { }
  ~InmemReader() { }

  // Pre-load all the data into memory buffer.
  virtual void Initialize(const std::string& filename);
  virtual void Initialize(xLearn::DMatrix* &dmatrix) { }

  // Sample data from the memory buffer.
  virtual index_t Samples(DMatrix* &matrix);

  // Return to the beginning of the data.
  virtual void Reset();

  // Free the memory of data matrix.
  virtual void Clear() {
    data_buf_.Reset();
  }

  // Rows are walked through this rather than copied into shuffled order.
  const std::vector<RowMeta>* Rows() const {
    return shuffle_ ? &rows_ : nullptr;
  }

  // Return the Reader type
  virtual std::string Type() {
    return "in-memory";
  }

  // If shuffle data ?
  virtual inline void SetShuffle(bool shuffle) {
    this->shuffle_ = shuffle;
    if (shuffle_ && !rows_.empty()) {
      ShuffleOrder(rows_, this->rng_);
    }
  }

  // Get data buffer
  virtual inline DMatrix* GetMatrix() {
    return &data_buf_;
  }

 protected:
  /* Reader will load all the data
  into this buffer */
  DMatrix data_buf_;
  /* Number of record at each sampling */
  index_t num_samples_;
  /* Position for sampling */
  index_t pos_;
  /* For random shuffle */
  std::vector<RowMeta> rows_;

  // Check whehter current path has a binary file.
  bool hash_binary(const std::string& filename);

  // Initialize Reader from existing binary file.
  void init_from_binary();

  // Initialize Reader from a new txt file.
  void init_from_txt();

 private:
  DISALLOW_COPY_AND_ASSIGN(InmemReader);
};

//------------------------------------------------------------------------------
// Sampling data from disk file.
// OndiskReader is used to train very big data, which cannot be
// loaded into main memory of current single machine.
//------------------------------------------------------------------------------
// TODO(chao): binary-cache
class OndiskReader : public Reader {
 public:
  // Constructor and Destructor
  OndiskReader() { }
  ~OndiskReader() { 
    Clear();
    Close(file_ptr_); 
  }

  // Create parser and open file
  virtual void Initialize(const std::string& filename);
  virtual void Initialize(xLearn::DMatrix* &dmatrix) { }

  // Sample data from disk file
  virtual index_t Samples(DMatrix* &matrix);

  // Return to the head of file
  virtual void Reset();

  // Free the memory of data matrix.
  virtual void Clear() {
    data_samples_.Reset();
  }

  // Return the Reader type
  virtual std::string Type() {
    return "on-disk";
  }

  // We cannot set shuffle for OndiskReader
  void inline SetShuffle(bool shuffle) {
    if (shuffle == true) {
      LOG(ERR) << "Cannot set shuffle for OndiskReader.";
    }
    this->shuffle_ = false;
  }

 protected:
  /* Sample() parses each block into this. The only reader that still needs a
  matrix of its own: the in-memory ones hand back the buffer they loaded. */
  DMatrix data_samples_;
  /* Maintain the file pointer */
  FILE* file_ptr_; 
 
 private:
  DISALLOW_COPY_AND_ASSIGN(OndiskReader);
};

class FromDMReader : public Reader {
 public:
  // Constructor and Destructor
  FromDMReader() : pos_(0) { }
  ~FromDMReader() { }

  virtual void Initialize(const std::string& filename) { };
  virtual void Initialize(xLearn::DMatrix* &dmatrix);

  virtual index_t Samples(DMatrix* &matrix);

  // Return to the beginning of the data.
  virtual void Reset() { pos_ = 0; }

  // Free the memory of data matrix. The matrix is owned by whoever handed it
  // to Initialize(), so there is nothing here to release.
  virtual void Clear() { }

  // Return the Reader type
  virtual std::string Type() {
    return "from-dmatrix";
  }

  // If shuffle data ?
  virtual inline void SetShuffle(bool shuffle) {
    this->shuffle_ = shuffle;
    if (shuffle_ && !rows_.empty()) {
      ShuffleOrder(rows_, this->rng_);
    }
  }

  // Rows are walked through this rather than copied into shuffled order.
  const std::vector<RowMeta>* Rows() const {
    return shuffle_ ? &rows_ : nullptr;
  }

 protected:
  DMatrix* data_ptr_;
  /* Number of record at each sampling */
  index_t num_samples_;
  /* Position for sampling */
  index_t pos_;
  /* For random shuffle */
  std::vector<RowMeta> rows_;


 private:
  DISALLOW_COPY_AND_ASSIGN(FromDMReader);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_reader_registry, Reader);

#define REGISTER_READER(format_name, reader_name)           \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_reader_registry,                               \
      Reader,                                               \
      format_name,                                          \
      reader_name)

#define CREATE_READER(format_name)                          \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_reader_registry,                               \
      format_name)

} // namespace xLearn

#endif // XLEARN_READER_READER_H_
