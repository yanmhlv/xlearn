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
This file provides Vec, a vector of real_t built on Google Highway.
*/

#ifndef XLEARN_BASE_SIMD_H_
#define XLEARN_BASE_SIMD_H_

#include "hwy/highway.h"

#include "src/base/math.h"

namespace xLearn {

namespace hn = hwy::HWY_NAMESPACE;

//------------------------------------------------------------------------------
// A short run of packed real_t, held in one vector register:
//
//    Vec<4> s = Vec<4>::Load(s_ptr);
//    s = s + Vec<4>::Load(w_ptr) * Vec<4>::Broadcast(v);
//    s.Store(s_ptr);
//
// kMaxLanes is a request, not a promise. It resolves through Highway's
// CappedTag to min(kMaxLanes, what this target has), so Vec<8> is eight lanes
// on AVX2 and four on NEON from the same source. Step loops by Lanes() rather
// than by kMaxLanes and the capped case stays correct instead of merely
// compiling; Lanes() is constexpr, so nothing is paid for asking.
//
// A caller picks the width from aligned_k, not from the model format. kAlign
// pads K to a multiple of four and is baked into the serialized model, so the
// width has to divide whatever that padding produced -- which is why the
// choice is per call rather than a constant here.
//
// Load and Store are unaligned. Requiring alignment would tie the width to
// whatever the allocators happen to guarantee -- param_v_ is aligned to
// kAlignByte and FM's scratch is a std::vector, both of which give exactly the
// 16 bytes four lanes need and no more, so a wider vector would fault rather
// than merely run slower. On every target this code compiles for, an unaligned
// load that does not straddle a cache line costs nothing.
//------------------------------------------------------------------------------
template <int kMaxLanes>
class Vec {
 public:
  typedef hn::CappedTag<real_t, kMaxLanes> Tag;
  typedef hn::Vec<Tag> Register;

  // How many real_t this actually holds, after the cap.
  static constexpr index_t Lanes() {
    return static_cast<index_t>(Tag().MaxLanes());
  }

  Vec() { }
  explicit Vec(Register reg) : reg_(reg) { }

  static inline Vec Zero() {
    return Vec(hn::Zero(Tag()));
  }

  static inline Vec Broadcast(real_t x) {
    return Vec(hn::Set(Tag(), x));
  }

  static inline Vec Load(const real_t* ptr) {
    return Vec(hn::LoadU(Tag(), ptr));
  }

  inline void Store(real_t* ptr) const {
    hn::StoreU(reg_, Tag(), ptr);
  }

  // Sum of the lanes.
  inline real_t Sum() const {
    return hn::ReduceSum(Tag(), reg_);
  }

  inline Register reg() const { return reg_; }

 private:
  Register reg_;
};

template <int N>
inline Vec<N> operator+(Vec<N> a, Vec<N> b) {
  return Vec<N>(hn::Add(a.reg(), b.reg()));
}

template <int N>
inline Vec<N> operator-(Vec<N> a, Vec<N> b) {
  return Vec<N>(hn::Sub(a.reg(), b.reg()));
}

template <int N>
inline Vec<N> operator*(Vec<N> a, Vec<N> b) {
  return Vec<N>(hn::Mul(a.reg(), b.reg()));
}

template <int N>
inline Vec<N> operator/(Vec<N> a, Vec<N> b) {
  return Vec<N>(hn::Div(a.reg(), b.reg()));
}

template <int N>
inline Vec<N> Sqrt(Vec<N> x) {
  return Vec<N>(hn::Sqrt(x.reg()));
}

// a * b + c. Written as one op because the compiler will not fuse a multiply
// and an add that reach it as two separate intrinsics, so spelling out the
// multiply-add is the only way to get the fused instruction where one exists.
template <int N>
inline Vec<N> MulAdd(Vec<N> a, Vec<N> b, Vec<N> c) {
  return Vec<N>(hn::MulAdd(a.reg(), b.reg(), c.reg()));
}

// c - a * b
template <int N>
inline Vec<N> NegMulAdd(Vec<N> a, Vec<N> b, Vec<N> c) {
  return Vec<N>(hn::NegMulAdd(a.reg(), b.reg(), c.reg()));
}

// Approximate 1/sqrt(x), straight from the hardware estimate: 12 bits from
// _mm_rsqrt_ps, 8 from the NEON vrsqrte. That is the same order of error as
// the InvSqrt() in math.h, which already scales the adagrad step for the
// linear terms in these very functions.
template <int N>
inline Vec<N> RSqrt(Vec<N> x) {
  return Vec<N>(hn::ApproximateReciprocalSqrt(x.reg()));
}

template <int N>
inline Vec<N> Abs(Vec<N> x) {
  return Vec<N>(hn::Abs(x.reg()));
}

// The magnitude of a, carrying the sign of b.
template <int N>
inline Vec<N> CopySign(Vec<N> a, Vec<N> b) {
  return Vec<N>(hn::CopySign(a.reg(), b.reg()));
}

//------------------------------------------------------------------------------
// Which lanes a comparison selected. Kept apart from Vec because the two are
// distinct register files on some targets.
//------------------------------------------------------------------------------
template <int kMaxLanes>
class VecMask {
 public:
  typedef hn::Mask<typename Vec<kMaxLanes>::Tag> Register;

  explicit VecMask(Register reg) : reg_(reg) { }

  inline Register reg() const { return reg_; }

 private:
  Register reg_;
};

template <int N>
inline VecMask<N> operator<=(Vec<N> a, Vec<N> b) {
  return VecMask<N>(hn::Le(a.reg(), b.reg()));
}

// Zero in the lanes the mask selected, x in the rest.
template <int N>
inline Vec<N> IfThenZeroElse(VecMask<N> mask, Vec<N> x) {
  return Vec<N>(hn::IfThenZeroElse(mask.reg(), x.reg()));
}

} // namespace xLearn

#endif // XLEARN_BASE_SIMD_H_
