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
This file provides Float4, a vector of four real_t built on Google Highway.
*/

#ifndef XLEARN_BASE_SIMD_H_
#define XLEARN_BASE_SIMD_H_

#include "hwy/highway.h"

#include "src/base/math.h"

namespace xLearn {

namespace hn = hwy::HWY_NAMESPACE;

//------------------------------------------------------------------------------
// Four packed real_t, held in one vector register. The width matches kAlign,
// so a Float4 covers exactly one aligned step of a latent factor vector:
//
//    Float4 s = Float4::Load(s_ptr);
//    s = s + Float4::Load(w_ptr) * Float4::Broadcast(v);
//    s.Store(s_ptr);
//
// The width is fixed at four rather than left to Highway to choose, because
// kAlign is baked into the serialized model: get_aligned_k() pads K to a
// multiple of it. FFM goes further and interleaves the per-coordinate
// optimizer state in blocks of that size, so a wider vector there would
// change the model format. FM does not -- it keeps each of w, wg and z in one
// flat run of aligned_k -- so FM could run wider whenever aligned_k allows.
//
// Load and Store both require kAlignByte alignment.
//------------------------------------------------------------------------------
class Float4 {
 public:
  typedef hn::FixedTag<real_t, 4> Tag;
  typedef hn::Vec<Tag> Register;

  Float4() { }
  explicit Float4(Register reg) : reg_(reg) { }

  static inline Float4 Zero() {
    return Float4(hn::Zero(Tag()));
  }

  static inline Float4 Broadcast(real_t x) {
    return Float4(hn::Set(Tag(), x));
  }

  static inline Float4 Load(const real_t* ptr) {
    return Float4(hn::Load(Tag(), ptr));
  }

  inline void Store(real_t* ptr) const {
    hn::Store(reg_, Tag(), ptr);
  }

  // Sum of the four lanes.
  inline real_t Sum() const {
    return hn::ReduceSum(Tag(), reg_);
  }

  inline Register reg() const { return reg_; }

 private:
  Register reg_;
};

inline Float4 operator+(Float4 a, Float4 b) {
  return Float4(hn::Add(a.reg(), b.reg()));
}

inline Float4 operator-(Float4 a, Float4 b) {
  return Float4(hn::Sub(a.reg(), b.reg()));
}

inline Float4 operator*(Float4 a, Float4 b) {
  return Float4(hn::Mul(a.reg(), b.reg()));
}

inline Float4 operator/(Float4 a, Float4 b) {
  return Float4(hn::Div(a.reg(), b.reg()));
}

inline Float4 Sqrt(Float4 x) {
  return Float4(hn::Sqrt(x.reg()));
}

// a * b + c. Written as one op because the compiler will not fuse a multiply
// and an add that reach it as two separate intrinsics, so spelling out the
// multiply-add is the only way to get the fused instruction where one exists.
inline Float4 MulAdd(Float4 a, Float4 b, Float4 c) {
  return Float4(hn::MulAdd(a.reg(), b.reg(), c.reg()));
}

// c - a * b
inline Float4 NegMulAdd(Float4 a, Float4 b, Float4 c) {
  return Float4(hn::NegMulAdd(a.reg(), b.reg(), c.reg()));
}

// Approximate 1/sqrt(x), straight from the hardware estimate: 12 bits from
// _mm_rsqrt_ps, 8 from the NEON vrsqrte. That is the same order of error as
// the InvSqrt() in math.h, which already scales the adagrad step for the
// linear terms in these very functions.
inline Float4 RSqrt(Float4 x) {
  return Float4(hn::ApproximateReciprocalSqrt(x.reg()));
}

inline Float4 Abs(Float4 x) {
  return Float4(hn::Abs(x.reg()));
}

// The magnitude of a, carrying the sign of b.
inline Float4 CopySign(Float4 a, Float4 b) {
  return Float4(hn::CopySign(a.reg(), b.reg()));
}

//------------------------------------------------------------------------------
// Which lanes a comparison selected. Kept apart from Float4 because the two
// are distinct register files on some targets.
//------------------------------------------------------------------------------
class Float4Mask {
 public:
  typedef hn::Mask<Float4::Tag> Register;

  explicit Float4Mask(Register reg) : reg_(reg) { }

  inline Register reg() const { return reg_; }

 private:
  Register reg_;
};

inline Float4Mask operator<=(Float4 a, Float4 b) {
  return Float4Mask(hn::Le(a.reg(), b.reg()));
}

// Zero in the lanes the mask selected, x in the rest.
inline Float4 IfThenZeroElse(Float4Mask mask, Float4 x) {
  return Float4(hn::IfThenZeroElse(mask.reg(), x.reg()));
}

} // namespace xLearn

#endif // XLEARN_BASE_SIMD_H_
