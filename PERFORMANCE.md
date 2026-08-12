# Performance decisions

Why the hot paths are shaped the way they are, and what was tried and thrown
away. Each record states the problem, the decision, the measurement that
settled it, and what it costs.

This is a record of decisions, not a tutorial. Where a decision is load-bearing
in a way a reader of the code would not guess, the reasoning is also in a
comment at the site; this file is the index.

## Terms

Used consistently below, and in the code:

- **Example** — one labelled observation: a set of feature values plus the
  target the model is asked to reproduce.
- **Feature** — one input dimension, identified by a zero-based feature id.
  Absent features are omitted rather than stored as zero.
- **Field** — a group of mutually exclusive features that came from the same
  original categorical variable. Only FFM models read fields; LR and FM leave
  every field id at zero.
- **Node** — one non-zero entry of an Example: a feature id, its value, and its
  field id. A DMatrix stores these as parallel columns rather than as a struct,
  and omits a column the data never varies.
- **Row** — the slot an Example occupies in a DMatrix. A Row is a position; the
  Example is what sits there.
- **DMatrix** — a contiguous run of Examples held in memory, with their labels
  and normalizers.
- **Row Reference** (`RowRef`) — a borrowed view of one Example's Nodes: the
  columns and the length they share, owning nothing and outliving nothing. An
  Example may legitimately have zero Nodes while still carrying a label.

---

## How anything here was measured

Every number below comes from the same harness, and the harness is part of the
result: several plausible-looking measurements in this project were wrong
before it settled.

- **Per-epoch time is a slope, not a total.** Wall time is measured at 2 epochs
  and at 6, and the difference is divided by 4. File load, parse, model
  allocation and process startup appear in both and cancel. Timing a single run
  charges parsing to whichever build parses faster, which is not the thing under
  test.
- **The `.bin` cache is warmed first.** The first run of a text file writes a
  binary cache beside it; the second reads it. Differencing an uncached run
  against a cached one reports *negative* instruction counts. A throwaway run
  precedes every measurement.
- **Builds are interleaved, not run in blocks**, so a burst of load on the host
  lands on both. Best of 3 (5 or 7 where the effect is near the noise floor).
- **One pinned P-core** (`--cpuset-cpus=0,1` on an i5-13500), single thread.
  Thread-scaling effects are real but they are a different experiment.
- **Each build gets its own copy of the data**, because they disagree about the
  layout of the `.bin` cache written next to it.
- **Quality is scored outside the engine.** AUC and logloss are computed from
  the written prediction file by an independent script, so a change to the
  engine's own metric code cannot flatter it.

Two calibrations worth keeping:

- **The noise floor is about ±2.5%** on the standard dataset and about **±5%**
  on the 148k-feature one. Anything smaller needs more repetitions or is not a
  result. The cheapest way to establish it is a configuration where both builds
  run identical code — for the parameter-prefetch work, LR was below the gate
  and served as exactly that control.
- **The seed-to-seed AUC band**, over five seeds: FM ftrl ±0.0002, FFM ftrl
  ±0.0012. SGD is looser, about ±0.004, because it is the most order-sensitive.
  A quality claim smaller than the band for its configuration is noise.

---

# Accepted

## 1. Store Nodes as columns, and elide a column the data never varies

**Status:** accepted.

**Problem.** A Row was a separately heap-allocated `std::vector<Node>`, and a
`Node` was 12 bytes: feature id, feature value, and a field id that libsvm data
never fills and that LR and FM never read. At 11 Nodes per Row that is 132 B of
nodes plus a 24 B vector object, a ~16 B heap header and an 8 B pointer:
**~173 B/Row**, matching the 52 MB arena measured at 300k Rows.

Hardware counters said LR was memory-bound and not compute-bound — 61.5
instructions per Node against the Rust port's 81.9, but **2.2x the wall time**,
IPC 0.92 against 2.69, and 126.7 LLC misses per 1000 Nodes against 0.64. It was
retiring 25% fewer instructions and taking twice as long to do it.

A Row-count sweep confirmed the mechanism: the gap tracked the arena crossing
the 24 MB LLC.

```
  20,000 rows | arena ~  3 MB | 1.19x slower than the reference
  60,000 rows | arena ~ 10 MB | 1.29x
 200,000 rows | arena ~ 34 MB | 1.89x
 700,000 rows | arena ~120 MB | 2.23x
```

**Decision.** Compressed sparse row: an offset array cutting parallel feature
columns into Rows. A column whose entry would be identical for every Node is
not stored at all and `RowRef` returns the implied value through an accessor —
libsvm data carries no fields, one-hot data carries no values but 1.0.

**Result.** ~44 B/Row on libsvm data, ~1/4 of the old footprint. Peak RSS
65 MB → 26 MB for LR and FM, 67 → 41 MB for FFM.

**Consequences.** `AddNode` loses random access by Row id — a caller filling
Rows out of order must buffer or sort first. A `RowRef` does not survive the
next `AddNode`, where a `SparseRow*` did; nothing today interleaves building and
reading, but new callers must be checked. Growing a column toward gigabytes
reallocates and copies, transiently needing ~1.5-2x the final size, which
`Reserve(rows, nodes)` avoids wherever the size is known in advance.

## 2. Interleave feature and field ids in one array

**Status:** accepted, against the obvious alternative.

**Problem.** Given the decision to store columns, the natural layout is one
array per column: feats, fields, vals.

**Decision.** Feature and field ids share one array, feature followed by field,
with a stride of 1 when the data carries no fields.

**Measurement.** Separating them was **12-14% slower** even though it moved
strictly fewer bytes: nothing reads a field without also reading its feature,
so splitting turned FFM's one random access per Row into two, into arrays far
enough apart to miss separately.

**Consequences.** `RowRef::feat` and `field` multiply by a stride instead of
indexing directly. The stride is loop-invariant and hoists.

## 3. Carry Row metadata as an array, not as an index

**Status:** accepted.

**Problem.** A caller visiting Rows in an order of its own needs, per Row, the
label, the normalizer, and where the Row's Nodes begin and end.

**Decision.** `RowMeta` — begin, len, label, norm — gathered into one array that
the caller permutes and walks.

**Measurement.** The alternative, permuting an array of indices, costs four
random accesses per Row: into `Y`, into `norm`, and into both ends of `offset`.
A gathered `RowMeta` array is read straight through and leaves only the Nodes
random. On a shuffled FFM epoch, **174 ms against 115 ms**.

**Consequences.** 16 B/Row of metadata is copied by the shuffle instead of 4 B
of index — see decision 12, where that cost is paid down.

## 4. Shuffle the traversal, not the data

**Status:** accepted.

**Problem.** Shuffling materialised a second DMatrix by copying every sampled
Row into it, once per epoch. Under CSR that is impossible to do by pointer and
ruinous to do by value: it would move every Node of every Row, every epoch.

**Decision.** The permutation moves out of the data and into the traversal.
`Reader::Rows()` returns the order to visit; the gradient loops map through it.

**Consequences.** A trap worth stating because it is the tempting design: the
order must **not** live inside `DMatrix` behind `Row(i)`. `Y[i]` and `norm[i]`
are read directly at call sites and would silently not be permuted.

`Predict` deliberately takes no order — `pred[i]` must stay aligned with `Y[i]`
for `Evaluate` and for the output file. Shuffling only ever mattered for the
SGD update sequence.

Side effect: the validation reader was previously shuffled too, pointlessly. It
is now traversed in natural order. Every metric is order-invariant, so this is
quality-equivalent.

## 5. Carry the shuffle generator across epochs

**Status:** accepted. **This is a quality fix, not a performance one.**

**Problem.** `ShuffleOrder(order_, seed_ + 1)` constructed a fresh `std::mt19937`
from a fixed seed at every epoch, so **every epoch walked the Rows in the same
permutation** — close to not shuffling at all.

**Decision.** One `std::mt19937` per Reader, seeded once, drawn from thereafter.

**Measurement.** LR sgd held-out AUC **0.63015 → 0.68146**. That is two orders
of magnitude outside the ±0.004 seed band for sgd. FM and FFM sgd move less
because their latent terms dominate.

## 6. Google Highway for SIMD, with the width a per-call parameter

**Status:** accepted. Two decisions in one area — which SIMD layer, and how wide.

### 6a. Why a portable SIMD library rather than intrinsics

**Problem.** The latent kernels are the whole cost of FM and FFM and they have
to vectorize. Three ways to get there, and CI builds all three of ubuntu
(x86-64), macOS (arm64/NEON) and Windows (MSVC), so none of them is optional
work:

- **Raw intrinsics.** `_mm256_fmadd_ps` and `vfmaq_f32` are different functions
  with different headers and different availability, so every kernel becomes two
  or three `#ifdef` branches that must be kept in step. The kernels here are
  already the subtlest code in the tree; tripling them is how the branches
  silently diverge.
- **Auto-vectorization alone.** It does not reliably produce a *fused*
  multiply-add, and the fusion is the point: a multiply and an add that reach
  the compiler as two separate operations are not fused, so spelling out
  `MulAdd` is the only way to get the instruction where one exists. It also
  cannot be relied on for the reduction and select patterns FTRL needs.
- **`std::experimental::simd`.** Not available across all three compilers at the
  standard this project targets.

**Decision.** [Google Highway](https://github.com/google/highway), pinned to
1.4.0 and fetched by CMake's `FetchContent` at configure time — no submodule, no
vendored copy, nothing for a user to install first.

It is wrapped in a thin `Vec<N>` in `src/base/simd.h` rather than used directly.
That wrapper is the load-bearing part: it keeps `hwy` out of the twenty-odd
kernel loops that would otherwise name it, so the operator set the kernels see
(`+ - * /`, `MulAdd`, `NegMulAdd`, `Sqrt`, `RSqrt`, `Abs`, `CopySign`,
`IfThenZeroElse`) is small, and swapping the layer underneath would be one file.
It is also what makes 6b expressible at all — the width becomes a template
parameter on our type rather than a different Highway tag at every call site.

**Consequences.** A build-time network fetch, which is why `HWY_ENABLE_TESTS`,
`EXAMPLES`, `CONTRIB` and `INSTALL` are all forced off — only the library
itself is built. Highway is compiled with its own flags, deliberately, because
it chooses its instruction sets itself; our `xlearn_flags` target is declared
after it for that reason.

Highway's **static** dispatch is what is used, not `HWY_DYNAMIC_DISPATCH`. That
is a deliberate trade with one visible cost, in 6b.

### 6b. Why the width is chosen per call

**Problem.** `Float4` fixed the width at four lanes because `kAlign` is baked
into the serialized model. FFM went further and interleaved optimizer state
every four floats, so a wider vector there would have changed the model format.
`FFMScore::CalcScore` disassembled to `ymm=0, xmm=60` on a machine with 256-bit
FMA.

**Decision.** `Vec<N>` over Highway's `CappedTag`, which resolves to
`min(N, hardware lanes)` — eight on AVX2 and four on NEON from one source, no
`#ifdef`. Loops step by `Lanes()`, not by a literal, so a capped width stays
*correct* rather than merely compiling. The width is chosen per call from
`aligned_k`; `kAlign` and the model file are untouched.

**Consequences.** Load and Store became unaligned. That is not a concession, it
is the point: aligned loads tie the width to whatever the allocators happen to
guarantee, and both `param_v_` (`posix_memalign` to 16) and FM's scratch
(`std::vector`) give exactly the 16 bytes four lanes need and no more — so at
eight lanes the aligned load **faults** rather than merely running slower. On
every target this compiles for, an unaligned load that does not straddle a cache
line is free.

The pip wheel sets `XLEARN_NATIVE_ARCH = "OFF"` and compiles to baseline
x86-64, where Highway caps the request back to four. Widening therefore benefits
source builds only. Everything else in this document is ISA-independent.

## 7. FM: accumulate the pairwise term in one pass, in registers

**Status:** accepted.

**Problem.** `CalcScore` walked the Row's latent blocks twice — once to build
the sum `s`, once in `LatentSum` recomputing `w * val` it already had.

**Decision.** One walk, folding each block against the `s` built from the blocks
before it. The pairwise total stays in registers rather than getting a scratch
plane of its own.

**Consequences, one earned the hard way.** The first version of this collapsed
the original's four independent accumulator chains into one and **regressed 20%
on a wide K** — `s` already carries a loop-carried dependency through memory,
and adding a second one put it on the critical path. The fix is `kChains`
independent accumulators, dispatched on how many vector blocks the width
actually leaves: eight lanes on AVX2 leave k=16 with two blocks where four lanes
leave it with four, and asking for chains the width cannot fill puts every block
back on one chain — which is where the dependency was in the first place.

This form is also **more numerically robust**, not less: the textbook
`0.5 * (s² - Σx²)` cancels catastrophically at large latent magnitudes and
overflows to `inf - inf` past `sqrt(FLT_MAX)`. The pair-at-a-time form has no
such range.

## 8. Fuse scoring and the gradient where there is something to share

**Status:** accepted, **conditionally** — and the condition is the interesting
part.

**Problem.** The training loop calls `CalcScore` then `CalcGrad` on the same
Row. FM rebuilds an identical `s` in `CalcGrad` over parameters the linear
update cannot have touched. FFM re-resolves the Row into terms.

**Decision.** A `Step()` on the `Score` base whose default body is today's pair,
plus `PrefersFusedStep()` asked **once per batch, not once per Row**, because
the answer decides which loop the Loss runs.

**Measurement.** FM and FFM say yes — FFM's duplicated row resolution alone was
**7%** of an epoch. LR says no, and it is not a wash: routing a score function
with nothing to share through `Step` puts an indirect call it cannot inline
around a body that does very little, which measured **17% on LR**.

**Consequences.** `Step` takes a raw function pointer and a context, not a
`std::function` — it runs once per Example and must not allocate.

## 9. FFM: whole planes instead of interleaved optimizer state

**Status:** accepted. **Breaks the model format.**

**Problem.** FFM stored each (feature, field) block as
`[w x4][wg x4][w x4][wg x4]…`, interleaving optimizer state every `kAlign`
floats. A score-only pass loaded every cache line for data it never read, and
the width was pinned at four forever because widening changes the serialized
layout.

**Decision.** Whole planes: `[w x aligned_k][wg x aligned_k][z x aligned_k]`,
which is what FM already did.

**Result.** Scoring touches half the cache lines under adagrad and a third under
FTRL, and decision 6 becomes applicable to FFM.

**Consequences, and the sharp edge.** At `aligned_k == 4` a single block means
interleaved and planar are **byte-identical**. Convenient — k ≤ 4 models are
unaffected — and dangerous, because the k=8 case is the only one that breaks, so
a test at k=4 will not catch it. The format gate is tested at k=8 specifically.
See decision 10; these two must land together.

## 10. Version both on-disk formats

**Status:** accepted. Not a performance change; it is what makes decision 9 safe.

**Problem.** Neither format had a magic word or a version.

`Model::Serialize` wrote `param_v_` as one opaque blob whose length is
recomputed on load from the feature, field, K and auxiliary counts. **A planar
file and an interleaved file are therefore the same length** — an old checkpoint
would have loaded with no error and read every latent coordinate from the wrong
slot, gradient cache as weights. Silent corruption, reachable through `-pre`,
which skips `Initialize()` and takes the layout from the file.

The `.bin` data cache was worse: `hash_binary` read two hashes positionally and
checked them against the **source text file**, so a cache re-encoded by a
different build, sitting beside an unchanged `.txt`, passed and was read back as
garbage.

**Decision.** Magic word and format version at the head of both, ahead of the
hashes so an old cache fails on its first eight bytes rather than being read as
a header.

**The two react differently on mismatch, deliberately.** A `.bin` is derived
from text and keyed by content hash, so an unknown version is treated exactly
like "no cache present" and the text is re-parsed. A `.model` cannot be
regenerated from anything, so it reports and stops.

One trap: `ReadDataFromDisk` returns short reads silently, so the header reader
must zero-initialise and check the returned count against `sizeof`. The old
`hash_binary` compared uninitialised bytes on a truncated file.

## 11. Compile the latent plane length as a constant

**Status:** accepted.

**Problem.** With the plane length a runtime value, the innermost loop over
vector blocks runs **once** at k=8 — and profiling showed roughly 35% of
`FFMScore::CalcScore` going to the compare, the branch and the pointer bumps
around that single block.

**Decision.** Kernels take the plane length as a template constant where the
caller knows it (`AK`), zero where it does not. The dispatcher picks 8, 4, or
runtime. Almost every model is trained at k ≤ 8.

**Measurement.**

```
FFM  k=8   adagrad -9.7%   ftrl -11.7%   sgd -15.6%
FFM  k=4   adagrad -11.0%  ftrl  -8.2%   sgd -14.0%
FM   k=8   adagrad -8.4%   ftrl  -8.6%   sgd -11.3%
```

Bit-identical AUC in every case.

**Consequences.** Four instantiations per kernel. The dispatch lives in one thin
wrapper per kernel rather than at every call site, so the call sites stay a
choice of width and nothing else — FM and FFM use the same shape.

## 12. Fetch each Row's Nodes before it is scored

**Status:** accepted. **The single largest win in this document.**

**Problem.** A shuffled epoch reads its `RowMeta` in order and its Nodes at
random. Profiling LR showed 58.8% of `CalcScore` on the load of a feature id.
That access has nowhere to hide: the hardware prefetcher cannot predict a
shuffled walk.

**Decision.** Four Rows ahead, issue a prefetch for the next Row's columns. One
is not enough to cover the miss; much further and the line is evicted before use.

**Measurement.**

```
LR   adagrad -28.2%   ftrl -23.9%   sgd -30.8%
FM   adagrad -13.0%   ftrl -10.6%   sgd -16.9%
FFM  adagrad -10.6%   ftrl  -6.8%   sgd -14.2%
```

Bit-identical AUC everywhere — it is a hint, and the epoch trains identically
with it removed.

## 13. Fetch a large model's parameters two Rows ahead

**Status:** accepted, **gated by model size**.

**Problem.** The second stage of the same pipeline. Which parameters a Row reads
is a function of its feature ids, so that fetch cannot be issued until the ids
themselves have arrived — hence a nearer distance than decision 12.

**Decision.** `Score::PrefetchParams`, called at distance 2, only when the
parameter arrays exceed 8 MB. Unlike the Node columns, which are always larger
than any cache, the parameters may be small enough to stay resident — and then
the hints are pure cost.

**Measurement.** On a 148k-feature model:

```
FM  adagrad  -17.5%   (9.5 MB of latent factors, above the gate)
FM  ftrl      -7.1%   (14.2 MB, above)
FM  sgd       +0.2%   (5.3 MB, below the gate — no prefetch issued)
```

On a 742k-feature LR model, ftrl (8.9 MB) **-20.3%**, while adagrad (5.9 MB) and
sgd (3.0 MB) stayed below the gate and unchanged. On the standard 3710-feature
dataset every configuration is within ±1.5%: the gate correctly costs nothing.

**Consequences.** The threshold sits near the cache a core keeps to itself. It
is a single constant, and the right value is a property of the machine, not of
the model.

## 14. FTRL: select the weight instead of branching on the sign of z

**Status:** accepted. **Largest single-kernel win.**

**Problem.** LR ftrl was the worst configuration in the tree — 2.29x the Rust
port's time, where every other configuration sat between 1.14x and 1.49x.
Profiling `calc_grad_ftrl` put 17% on one `vdivss`, 11.5% across two `vsqrtss`,
and a further ~13% on a pair of `vcomiss`/`ja` — GCC had compiled the sign test
into a **real branch**.

**Decision.** Compute both arms and select. `std::copysign(1.0f, z)` for the
sign, `std::fabs(z) <= lambda_1` for the soft threshold.

**Measurement.** LR ftrl **-38.6%**, FM ftrl **-26.5%**, FFM ftrl **-5.7%**,
AUC bit-identical.

**Why it is so large.** Which way `z` falls is a coin toss the branch predictor
cannot learn, so it mispredicts about half the time. The mispredict costs more
than the divide the branch was there to skip.

**Consequences.** The divide now always executes. That is the trade, and it is
overwhelmingly favourable: with `lambda_1 = 0` the branch was almost never
taken anyway.

## 15. FTRL: one linear update, held in registers

**Status:** accepted.

**Problem.** The scalar FTRL linear update was copied verbatim into
`linear_score.cc`, `fm_score.cc` and `ffm_score.cc`. Each took `w`, `n` and `z`
as **references into the parameter array**, where they alias one another as far
as the compiler can prove — so the store to `z` forced a reload of `n` and a
second `std::sqrt` of a sum the line above already had. Three square roots per
feature where two suffice.

**Decision.** One `ftrl_update` on the `Score` base, reading the three slots into
locals and writing them back once.

**Measurement.** LR ftrl -5.4%, FM ftrl -2.1%. Bit-identical: the operations and
their order are unchanged, only the redundant reload is gone.

**Note on the SIMD kernels.** The same triple-sqrt appears in the vector FTRL
kernels and costs nothing there, because the values are locals rather than
memory references and GCC common-subexpression-eliminates the third. Aliasing
was the whole story.

## 16. FTRL: start z at zero

**Status:** accepted. **A math fix, not a performance one.**

**Problem.** `Model::set_value()` initialised **every** auxiliary plane to 1.0,
including FTRL's `z`, which must start at 0.

With `z = 1`, the first update to any weight sets it to roughly
`-1 / ((beta + sqrt(n))/alpha + lambda_2)` ≈ -0.15 **regardless of the
gradient** — so the random latent initialisation is discarded on first touch and
the linear weights acquire a systematic negative bias. This affected LR, FM and
FFM alike under `-p ftrl`.

**Measurement.** FFM ftrl 0.69314 → 0.69419 with **non-overlapping** before/after
ranges over five seeds.

## 17. Run the shuffle's draws ahead of its swaps

**Status:** accepted, marginally.

**Problem.** Once the epoch itself got fast, `std::shuffle` over
`std::vector<RowMeta>` was **17% of an LR sgd epoch**. It moves 16 B/Row at
random over 4.8 MB, waiting on DRAM for every swap.

**Decision.** Which partner a step swaps with depends only on the step, never on
the array, so the draws can run ahead of the swaps — a small ring of pending
draws, each prefetched.

**Measurement.** LR adagrad -5.8%, ftrl -2.9%, sgd -3.3% (best of 7; at 3
repetitions this sat inside the noise band and needed re-measuring before it
could be called a result).

**Consequences.** The permutation differs from `std::shuffle`'s, so AUC moves
within the order-noise band. That makes subsequent A/B comparisons on sgd
noisier, which is a real cost for a 3-6% gain. Kept, but it is the closest call
in this document.

## 18. AUC: count on the calling thread, and stop copying the histogram

**Status:** accepted.

**Problem.** Two problems in one class. `Accumulate` constructed a
million-bucket `Info` **per thread on every call** — 160 MB of zeroed allocation
per validation pass at 20 threads, plus a million-bucket reduction per worker,
to spread a few tens of thousands of increments. And `CalcAUC` took both
histograms **by value** — 8 MB of copying per call — then ignored the arguments
and read the members anyway, which is why nobody noticed.

**Decision.** Count on the calling thread; take the histograms by const
reference and actually read them.

**Also fixed.** The bucketing now uses an exact sigmoid rather than the
approximate one — this bucketing is what *orders* the predictions, so an error
here reorders neighbours outright rather than shifting a score slightly — and
saturation clamps to the top bucket instead of wrapping, which had been filing
the most confident positives under the lowest score. A single-class input
returns 0.5 rather than NaN, which compares false against everything and told
early stopping that no epoch ever improved.

---

# Rejected

Each of these was implemented and measured. They are recorded because they all
look like good ideas, and the next person will think of them.

## R1. Clang instead of GCC

**Slower on every single configuration**, +4% to +19%, same source, same flags,
same `-march=native`:

```
LR   adagrad +10.1%   ftrl +19.4%   sgd +16.5%
FM   adagrad  +4.5%   ftrl  +9.1%   sgd  +6.8%
FFM  adagrad  +9.8%   ftrl  +4.2%   sgd +10.9%
```

Worth re-testing on a compiler bump; it is one Docker build. As of clang 19 on
Debian trixie, GCC wins.

## R2. `-fno-math-errno`

**No effect** (within ±2.5%). The hypothesis was that `std::sqrt` carried a
libm call for errno semantics. GCC already emits an inline `sqrtss` and puts the
errno path behind a cold, perfectly-predicted branch. Reverted.

## R3. Parameter prefetch for FFM

**Flat, twice**, on a 104 MB model — and marginally negative on ftrl. A
feature's latent blocks for every field span eleven cache lines on production
data, so fetching the first buys nothing, and the pair loop then reads them at
scattered field offsets. Implemented for LR and FM (decision 13); the FFM
override is deliberately absent, and the header says so.

## R4. splitmix64 for the shuffle

A small stateless generator looks like the cheap choice against mt19937 and is
not: mt19937 generates 624 words at a time, so a draw costs a few cycles of
buffer read, where splitmix64 pays the full latency of its dependent multiplies
on every one. Took the shuffle from **7.3% to 10.8%** of an LR epoch.

## R5. Four accumulator chains over pairs in FFM's score kernel

The score kernel rotates four accumulators over the coordinate `d`, which leaves
a short K with only one chain. Spreading them over *pairs* instead is the
obvious next move.

**It is architecture-dependent, and x86 loses.** It needs four pairs of weight
pointers live at once, which x86-64's sixteen vector registers cannot hold:
**260 ms against 190** at k=8, from register spill. On arm64, with thirty-two
registers, the same change is **30% faster**. Kept the d-rotation, which is
within 2% of the best at both k=8 and k=16 on x86.

## R6. Separate arrays for feature and field ids

See decision 2. **12-14% slower** despite moving fewer bytes.

## R7. Shrinking the FFM Row encoding to fit the LLC

**A hypothesis disproved by its own experiment**, recorded because the reasoning
was persuasive and wrong.

FFM's LLC miss rate looked high, suggesting the Row arena should be shrunk to
fit. Growing the arena from 2.5 MB to 30 MB raised LLC misses **27-fold** —
3.43 to 91.13 per 1000 Nodes — while cycles per Node stayed **flat**, 234.2 to
237.1. Those are streaming misses fully covered by the hardware prefetcher, and
shrinking the encoding would have bought nothing. FFM is instruction-bound; its
wins came from decisions 9 and 11.

## R8. Sharding the AUC histogram across threads

See decision 18. The sharding was more work than the counting it parallelized.

---

## Where this leaves the engine

Against the engine as it stood before this work (`6c97797`), on 300k Examples ×
11 Nodes, single thread:

| | k=8 | k=4 | peak RSS |
|---|---|---|---|
| LR | -51% to -62% | -52% to -60% | 0.40x |
| FM | -52% to -59% | -53% to -59% | 0.40x |
| FFM | -31% to -42% | -24% to -40% | 0.61x |

Against an independent Rust port of the same algorithms on the same data, the
C++ is now faster in 7 of 9 configurations at k=8 and 8 of 9 at k=4, on 0.58x
the memory for LR and FM and 0.78-0.83x for FFM. The one remaining loss is FFM
adagrad at 1.10x, whose kernel profiles as evenly spread across its vector ops
with no stall left — that gap is codegen, not structure.

## Known headroom

- **FFM adagrad**, the one configuration still behind the reference.
- **The `unit` specialization.** One-hot data with normalization off multiplies
  every pair by `1.0 * 1.0 * 1.0`; specializing that away deletes the per-pair
  value loads and two scalar multiplies. Not implemented here, and not visible
  on the benchmark set, which runs with normalization on.
- **Tiled shuffling.** Keeping the shuffled walk near-sequential, rather than
  uniformly random, trades a little independence for locality.
- **Duplicate (feature, field) Nodes in one Example.** The FFM pair loop reads
  both halves and then writes both, losing one of the two updates when they are
  the same block. A correctness nuance, not a performance one, and not currently
  handled.
