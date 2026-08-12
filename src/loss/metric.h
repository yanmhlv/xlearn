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
This file defines the Metric class.
*/

#ifndef XLEARN_LOSS_METRIC_H_
#define XLEARN_LOSS_METRIC_H_

#include <math.h>

#include <algorithm>
#include <cmath>

#include "src/base/common.h"
#include "src/base/math.h"
#include "src/base/class_register.h"
#include "src/base/thread_pool.h"
#include "src/data/data_structure.h"

namespace xLearn {

// Bucket size used by AUC
const index_t kMaxBucketSize = 1e6;

//------------------------------------------------------------------------------
// A metric is a function that is used to judge the performance of 
// your model. A metric function is similar to an loss function, except
// that the results from evaluating a metric are not used when training
// the model.  The Metric class is an abstract class, which can be 
// implemented by real Metric functions, such as AccMetric, PrecMetric, 
// RecallMetric, F1Metric, MAEMetric, RMSDMetric, MAPEMetric, etc.
// We can use the Metric class like this:
// 
//    AccMetric metric;
//    std::vector<real_t> pred;
//    while (1) {
//      int tmp = reader->Samples(matrix);
//      if (tmp == 0) { break; }
//      pred.resize(tmp);
//      metric->Accumulate(matrix->Y, pred);
//    }
//    real_t metric_val = metric.GetMetric();  
//------------------------------------------------------------------------------
class Metric {
 public:
  // Constructor and Destructor
  Metric() { }
  virtual ~Metric() { }

  void Initialize(ThreadPool* pool) {
    CHECK_NOTNULL(pool);
    pool_ = pool;
    threadNumber_ = pool_->ThreadNumber();
  }

  // Accumulate counters during the training.
  virtual void Accumulate(const std::vector<real_t>& Y,
                          const std::vector<real_t>& pred) = 0;

  // Reset counters
  virtual void Reset() = 0;

  // Return the final metric value.
  virtual real_t GetMetric() = 0;

  // Return the metric type.
  virtual std::string metric_type() = 0;

  // Compare two metric value, which is used in early-stop.
  virtual bool cmp(const real_t a, const real_t b) = 0;

 protected:
  /* Pointer of thread pool */
  ThreadPool* pool_;
  /* Thread number used by Metric */
  size_t threadNumber_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Metric);
};

 /*********************************************************
  *  For classification                                   *
  *********************************************************/

//------------------------------------------------------------------------------
// Accuracy is used as statistical measure of how well a binary classification
// test correctly identifies or excludes a condition. That is, the Accuracy 
// is the proportion of true results (both true positives and true negatives)
// among the total number of cases examined. 
//------------------------------------------------------------------------------
class AccMetric : public Metric {
 public:
  // Constructor and Destructor
  AccMetric()
   : total_example_(0),
     true_pred_(0) { }
  ~AccMetric() { }

  // Accumulate counters in one thread
  static void acc_accum_thread(const std::vector<real_t>* Y,
                               const std::vector<real_t>* pred,
                               index_t* true_pred,
                               size_t start_idx,
                               size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *true_pred = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      real_t p_label = (*pred)[i] > 0 ? 1 : -1;
      real_t r_label = (*Y)[i] > 0 ? 1 : -1;
      if (p_label == r_label) {
        (*true_pred)++;
      }
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    total_example_ += Y.size();
    // multi-thread training
    std::vector<index_t> sum(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(acc_accum_thread,
                               &Y,
                               &pred,
                               &(sum[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum.size(); ++i) {
      true_pred_ += sum[i];
    }
  }

  // Reset counters
  void Reset() {
    total_example_ = 0;
    true_pred_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return (true_pred_ * 1.0) / total_example_;
  }

  // Metric type
  std::string metric_type() { 
    return "Accuracy"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a >= b ? true : false;
  }

 protected:
  index_t total_example_;
  index_t true_pred_;

 private:
  DISALLOW_COPY_AND_ASSIGN(AccMetric);
};

//------------------------------------------------------------------------------
// Precision is a description of random errors, a measure of 
// statistical variability. In the field of information retrieval, 
// precision is the fraction of retrieved documents that are relevant 
// to the query.
//------------------------------------------------------------------------------
class PrecMetric : public Metric {
 public:
  // Constructor and Destructor
  PrecMetric() 
   : true_positive_(0),
     false_positive_(0) { }
  ~PrecMetric() { }

  // Accumulate counters in one thread
  static void prec_accum_thread(const std::vector<real_t>* Y,
                                const std::vector<real_t>* pred,
                                index_t* true_pos,
                                index_t* false_pos,
                                size_t start_idx,
                                size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *true_pos = 0;
    *false_pos = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      real_t p_label = (*pred)[i] > 0 ? 1 : -1;
      real_t r_label = (*Y)[i] > 0 ? 1 : -1;
      if (p_label > 0 && r_label > 0) {
        (*true_pos)++;
      } else if (p_label > 0 && r_label < 0) {
        (*false_pos)++;
      }
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    // multi-thread training
    std::vector<index_t> sum_1(threadNumber_, 0);
    std::vector<index_t> sum_2(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(prec_accum_thread,
                               &Y,
                               &pred,
                               &(sum_1[i]),
                               &(sum_2[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum_1.size(); ++i) {
      true_positive_ += sum_1[i];
    }
    for (size_t i = 0; i < sum_2.size(); ++i) {
      false_positive_ += sum_2[i];
    }
  }

  // Reset counters
  void Reset() {
    true_positive_ = 0;
    false_positive_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return (true_positive_ * 1.0) / 
           (true_positive_ + false_positive_);
  }

  // Metric type
  std::string metric_type() { 
    return "Precision"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a >= b ? true : false;
  }

 protected:
  index_t true_positive_;
  index_t false_positive_;

 private:
  DISALLOW_COPY_AND_ASSIGN(PrecMetric);
};

//------------------------------------------------------------------------------
// Recall is the ratio of correctly predicted positive observations 
// to the all observations in actual class.  In information retrieval, 
// recall is the fraction of the relevant documents that are 
// successfully retrieved.
//------------------------------------------------------------------------------
class RecallMetric : public Metric {
 public:
  // Constructor and Destructor
  RecallMetric() 
   : true_positive_(0),
     false_negative_(0) { }
  ~RecallMetric() { }

  // Accumulate counters in one thread
  static void recall_accum_thread(const std::vector<real_t>* Y,
                                const std::vector<real_t>* pred,
                                index_t* true_pos,
                                index_t* false_neg,
                                size_t start_idx,
                                size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *true_pos = 0;
    *false_neg = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      real_t p_label = (*pred)[i] > 0 ? 1 : -1;
      real_t r_label = (*Y)[i] > 0 ? 1 : -1;
      if (p_label > 0 && r_label > 0) {
        (*true_pos)++;
      } else if (p_label < 0 && r_label > 0) {
        (*false_neg)++;
      }
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    // multi-thread training
    std::vector<index_t> sum_1(threadNumber_, 0);
    std::vector<index_t> sum_2(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(recall_accum_thread,
                               &Y,
                               &pred,
                               &(sum_1[i]),
                               &(sum_2[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum_1.size(); ++i) {
      true_positive_ += sum_1[i];
    }
    for (size_t i = 0; i < sum_2.size(); ++i) {
      false_negative_ += sum_2[i];
    }
  }

  // Reset counters
  void Reset() {
    true_positive_ = 0;
    false_negative_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return (true_positive_ * 1.0) / 
           (true_positive_ + false_negative_);
  }

  // Metric type
  std::string metric_type() { 
    return "Recall"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a >= b ? true : false;
  }

 protected:
  index_t true_positive_;
  index_t false_negative_;

 private:
  DISALLOW_COPY_AND_ASSIGN(RecallMetric);
};

//------------------------------------------------------------------------------
// F1 Score is the weighted average of Precision and Recall. 
// Therefore, this score takes both false positives and false 
// negatives into account. Intuitively it is not as easy to understand 
// as accuracy, but F1 is usually more useful than accuracy, especially 
// if you have an uneven class distribution
//------------------------------------------------------------------------------
class F1Metric : public Metric {
 public:
  // Constructor and Destructor
  F1Metric() 
   : total_example_(0), 
     true_positive_(0),
     true_negative_(0) { }
  ~F1Metric() { }

  // Accumulate counters in one thread
  static void f1_accum_thread(const std::vector<real_t>* Y,
                              const std::vector<real_t>* pred,
                              index_t* true_pos,
                              index_t* true_neg,
                              size_t start_idx,
                              size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *true_pos = 0;
    *true_neg = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      real_t p_label = (*pred)[i] > 0 ? 1 : -1;
      real_t r_label = (*Y)[i] > 0 ? 1 : -1;
      if (p_label > 0 && r_label > 0) {
        (*true_pos)++;
      } else if (p_label < 0 && r_label < 0) {
        (*true_neg)++;
      }
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    total_example_ += Y.size();
    // multi-thread training
    std::vector<index_t> sum_1(threadNumber_, 0);
    std::vector<index_t> sum_2(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(f1_accum_thread,
                               &Y,
                               &pred,
                               &(sum_1[i]),
                               &(sum_2[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum_1.size(); ++i) {
      true_positive_ += sum_1[i];
    }
    for (size_t i = 0; i < sum_2.size(); ++i) {
      true_negative_ += sum_2[i];
    }
  }

  // Reset counters
  void Reset() {
    true_positive_ = 0;
    true_negative_ = 0;
    total_example_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return (true_positive_ * 2.0) / 
           (total_example_ + true_positive_ - true_negative_);
  }

  // Metric type
  std::string metric_type() { 
    return "F1"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a >= b ? true : false;
  }

 protected:
  index_t total_example_;
  index_t true_positive_;
  index_t true_negative_;

 private:
  DISALLOW_COPY_AND_ASSIGN(F1Metric);
};

//------------------------------------------------------------------------------
// The area under the curve (often referred to as simply the AUC) is 
// equal to the probability that a classifier will rank a randomly chosen 
// positive instance higher than a randomly chosen negative one 
// (assuming 'positive' ranks higher than 'negative').
//------------------------------------------------------------------------------
class AUCMetric : public Metric {
 public:
  // Constructor and Destructor
  AUCMetric() {
    all_positive_number_.clear();
    all_negative_number_.clear();
    all_positive_number_.resize(kMaxBucketSize, 0);
    all_negative_number_.resize(kMaxBucketSize, 0);
  }
  ~AUCMetric() { }

  // Accumulate counters during the training.
  //
  // Counted on the calling thread. The histogram is a million buckets wide,
  // so giving each worker its own costs 8 MB of allocation and a
  // million-bucket reduction per worker, to spread a few tens of thousands of
  // increments -- the sharding was more work than the counting it parallelized.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
      // An exact sigmoid, not the approximate one: this bucketing is what
      // orders the predictions, so an error here reorders neighbours outright
      // rather than shifting a score slightly. A confident prediction
      // saturates it to 1.0, which has to clamp to the top bucket -- wrapping
      // would file the most confident positives under the lowest score.
      real_t sigmoid_score = 1.0f / (1.0f + std::exp(-pred[i]));
      index_t bkt_id = std::min(index_t(sigmoid_score * kMaxBucketSize),
                                kMaxBucketSize - 1);
      if (Y[i] > 0) {
        all_positive_number_[bkt_id] += 1;
      } else {
        all_negative_number_[bkt_id] += 1;
      }
    }
  }

  // Reset counters
  void Reset() {
    all_positive_number_.clear();
    all_negative_number_.clear();
    all_positive_number_.resize(kMaxBucketSize, 0);
    all_negative_number_.resize(kMaxBucketSize, 0);
  }

  // Return AUC
  real_t GetMetric() {
    return CalcAUC(all_positive_number_, 
                   all_negative_number_);
  }

  // Metric type
  std::string metric_type() {
    return "AUC";
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a >= b ? true : false;
  }

 protected:
  std::vector<index_t> all_positive_number_;
  std::vector<index_t> all_negative_number_;

  // The histograms are a million buckets wide, so taking them by value copied
  // eight megabytes per call -- and then read the members anyway, which is why
  // nobody noticed.
  real_t CalcAUC(const std::vector<index_t>& positive_vec,
                 const std::vector<index_t>& negative_vec) {
    CHECK_EQ(positive_vec.size(), negative_vec.size());
    long long positive_sum = 0;
    long long negative_sum= 0;
    long long pre_positive_sum = 0.0;
    double auc = 0.0;
    for (index_t i = 0; i < kMaxBucketSize; ++i) {
      pre_positive_sum = positive_sum;
      positive_sum += positive_vec[i];
      negative_sum += negative_vec[i];
      auc += (pre_positive_sum + positive_sum) *
             (double)(negative_vec[i]) * 1.0 / 2;
    }
    // With one class absent there is no pair to rank, and the ratio below is
    // 0/0. Report the value a coin flip would earn rather than a NaN, which
    // compares false against everything and would tell early stopping that no
    // epoch ever improved.
    if (positive_sum == 0 || negative_sum == 0) {
      return 0.5;
    }
    return 1.0 - auc / (double)(positive_sum * negative_sum);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(AUCMetric);
};

 /*********************************************************
  *  For regression                                       *
  *********************************************************/

inline real_t abs(real_t a) { return a > 0 ? a : -a; }

//------------------------------------------------------------------------------
// MAE (Mean Absolute Error) is a measure of difference 
// between two continuous variables. Assume X and Y are 
// variables of paired observations that express the same phenomenon. 
//------------------------------------------------------------------------------
class MAEMetric : public Metric {
 public:
  // Constructor and Destructor
  MAEMetric() 
   : error_(0),
     total_example_(0) { }
  ~MAEMetric() { }

  // Accumulate counters in one thread
  static void mae_accum_thread(const std::vector<real_t>* Y,
                               const std::vector<real_t>* pred,
                               real_t* error,
                               size_t start_idx,
                               size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *error = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      (*error) += abs((*Y)[i] - (*pred)[i]);
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    total_example_ += Y.size();
    // multi-thread training
    std::vector<real_t> sum(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(mae_accum_thread,
                               &Y,
                               &pred,
                               &(sum[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum.size(); ++i) {
      error_ += sum[i];
    }
  }

  // Reset counters
  void Reset() {
    error_ = 0;
    total_example_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return error_ / total_example_;
  }

  // Metric type
  std::string metric_type() { 
    return "MAE"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a <= b ? true : false;
  }

 protected:
  real_t error_;
  index_t total_example_;

 private:
  DISALLOW_COPY_AND_ASSIGN(MAEMetric);
};

//------------------------------------------------------------------------------
// The mean absolute percentage error (MAPE), also known as mean absolute 
// percentage deviation (MAPD), is a measure of prediction accuracy of a 
// forecasting method in statistics, for example in trend estimation. 
//------------------------------------------------------------------------------
class MAPEMetric : public Metric {
 public:
  // Constructor and Destructor
  MAPEMetric() 
   : error_(0),
     total_example_(0) { }
  ~MAPEMetric() { }

  // Accumulate counters in one thread
  static void mae_accum_thread(const std::vector<real_t>* Y,
                               const std::vector<real_t>* pred,
                               real_t* error,
                               size_t start_idx,
                               size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *error = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      (*error) += abs((*Y)[i]-(*pred)[i]) / (*Y)[i];
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    total_example_ += Y.size();
    // multi-thread training
    std::vector<real_t> sum(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(mae_accum_thread,
                               &Y,
                               &pred,
                               &(sum[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum.size(); ++i) {
      error_ += sum[i];
    }
  }

  // Reset counters
  void Reset() {
    error_ = 0;
    total_example_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return error_ / total_example_;
  }

  // Metric type
  std::string metric_type() { 
    return "MAPE"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a <= b ? true : false;
  }

 protected:
  real_t error_;
  index_t total_example_;

 private:
  DISALLOW_COPY_AND_ASSIGN(MAPEMetric);
};

//------------------------------------------------------------------------------
// The root-mean-square deviation (RMSD) or root-mean-square error (RMSE)
// is a frequently used measure of the differences between values 
// (sample and population values) predicted by a model or an estimator 
// and the values actually observed. 
//------------------------------------------------------------------------------
class RMSDMetric : public Metric {
 public:
  // Constructor and Destructor
  RMSDMetric() 
   : error_(0),
     total_example_(0) { }
  ~RMSDMetric() { }

  // Accumulate counters in one thread
  static void rmsd_accum_thread(const std::vector<real_t>* Y,
                                const std::vector<real_t>* pred,
                                real_t* error,
                                size_t start_idx,
                                size_t end_idx) {
    CHECK_GE(end_idx, start_idx);
    *error = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      real_t tmp = (*Y)[i] - (*pred)[i];
      (*error) += tmp * tmp;
    }
  }

  // Accumulate counters during the training.
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_EQ(Y.size(), pred.size());
    total_example_ += Y.size();
    // multi-thread training
    std::vector<real_t> sum(threadNumber_, 0);
    for (int i = 0; i < threadNumber_; ++i) {
      size_t start_idx = getStart(pred.size(), threadNumber_, i);
      size_t end_idx = getEnd(pred.size(), threadNumber_, i);
      pool_->enqueue(std::bind(rmsd_accum_thread,
                               &Y,
                               &pred,
                               &(sum[i]),
                               start_idx,
                               end_idx));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber_);
    for (size_t i = 0; i < sum.size(); ++i) {
      error_ += sum[i];
    }
  }

  // Reset counters
  void Reset() {
    error_ = 0;
    total_example_ = 0;
  }

  // Return Accuracy
  real_t GetMetric() {
    return sqrt(error_ / total_example_);
  }

  // Metric type
  std::string metric_type() { 
    return "RMSD"; 
  }

  // Compare two metric value
  bool cmp(const real_t a, const real_t b) {
    return a <= b ? true : false;
  }

 protected:
  real_t error_;
  index_t total_example_;

 private:
  DISALLOW_COPY_AND_ASSIGN(RMSDMetric);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_metric_registry, Metric);

#define REGISTER_METRIC(format_name, metric_name)           \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_metric_registry,                               \
      Metric,                                               \
      format_name,                                          \
      metric_name)

#define CREATE_METRIC(format_name)                          \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_metric_registry,                               \
      format_name)

}  // namespace xLearn

#endif  // XLEARN_LOSS_METRIC_H_
