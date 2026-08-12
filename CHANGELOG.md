# Changelog

Notable changes to this fork of xLearn. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Why a change was made, and what was measured, is in
[PERFORMANCE.md](PERFORMANCE.md) for the hot paths and in comments at the site
for everything else.

## [Unreleased]

Training is 1.3-2.6x faster and uses 1.6-2.5x less memory than the previous
release, at equal or better held-out quality. Two on-disk formats changed; see
**Breaking** below before upgrading.

### Breaking

- **`.model` checkpoints trained at k > 4 are no longer readable.** FFM latent
  blocks are now stored as whole planes rather than interleaving optimizer state
  every four floats. An old and a new file of the same shape are the same
  length, so the previous format — which carried no magic bytes and no version —
  would have loaded one as the other and read every latent coordinate from the
  wrong slot. Checkpoints now carry a magic word and a version and an
  unrecognised file is refused with a message naming the fix. Retrain, or keep
  the old binary to serve the old model. Models trained at k <= 4 are unaffected:
  at that width the two layouts are byte-identical.
- **`-pre` is the path this matters on.** It skips `Initialize()` and takes the
  model shape from the file, so it is how a stale checkpoint reaches the new
  kernels.
- **`Loss::Initialize()` lost its `batch_size` parameter** and
  `Loss::CalcGrad()` gained a traversal-order argument. Both are internal C++
  interfaces; the CLI and the Python API are unchanged.

### Migration

- **`.bin` data caches need no action.** A cache is derived from a text file and
  keyed by its content hash, so an unrecognised version is treated as "no cache
  present" and the text is re-parsed. Stale caches are silently regenerated.
- **Model files must be retrained** if they were trained at k > 4. There is no
  converter.

### Added

- `PERFORMANCE.md`, recording each optimization, its measurement, and the eight
  approaches that were implemented, measured, and rejected.
- Magic word and format version at the head of both the `.model` checkpoint and
  the `.bin` data cache.
- `Reader::Rows()`, exposing the order Examples should be visited in, so a
  shuffled epoch permutes an index rather than copying every Example it visits.
- `Score::Step()`, scoring and applying the gradient in one call, so an
  implementation can keep what the two halves share. Guarded by
  `Score::PrefersFusedStep()`, asked once per batch — FM and FFM opt in, LR
  deliberately does not.
- `Score::PrefetchParams()`, and Example-ahead prefetching in the gradient
  loops. Both are hints; an epoch trains bit-identically without them.
- `DMatrix::Reserve()`, so callers that know the final size — the binary cache
  and the Python arrays — avoid repeatedly reallocating a multi-gigabyte column.
- `Model::Initialize()` takes a seed, so a run can be repeated exactly and a
  family of runs can differ.

### Changed

- **Examples are stored as columns** rather than as a separately heap-allocated
  vector of 12-byte nodes per Example, and a column whose entries never vary is
  not stored at all. ~173 bytes per Example becomes ~44 on libsvm data.
- **FFM latent blocks are planar.** Scoring reads half the cache lines it used
  to under adagrad and a third under FTRL.
- **The SIMD width is chosen per call** from the padded latent length rather
  than fixed at four lanes, resolving through Highway to eight on AVX2 and four
  on NEON from one source. The serialized model format is unchanged by this.
  Note that the pip wheel compiles to baseline x86-64 by deliberate choice, so
  the widening reaches source builds only.
- **The latent plane length is compiled as a constant** for the k <= 8 models
  that are almost all of them, removing the loop machinery around a block loop
  that runs once.
- **FM accumulates the pairwise term in one pass**, in registers. Also more
  numerically robust than the previous `0.5 * (s^2 - sum x^2)`, which cancels
  catastrophically at large latent magnitudes and overflows past
  `sqrt(FLT_MAX)`.
- **The FTRL linear update is shared by all three score functions**, holds a
  feature's slots in registers rather than in aliasing references, and selects
  the new weight rather than branching on the sign of `z`.
- The shuffle runs its draws ahead of its swaps, and is no longer re-seeded
  identically at every epoch (see **Fixed**).
- The validation reader is no longer shuffled. Every metric is order-invariant,
  so this is quality-equivalent.
- `Model::GetAuxiliarySize()` returns `index_t` rather than `real_t`, so offset
  expressions stop being float multiplies narrowed back to integers.
- AUC bucketing uses an exact sigmoid rather than the approximate one, and
  saturates to the top bucket instead of wrapping.
- `get_user_name()` falls back to `"unknown"` instead of constructing a
  `std::string` from a null pointer.

### Fixed

- **Every epoch shuffled Examples into the same order.** The generator was
  reconstructed from a fixed seed at each epoch, which is close to not shuffling
  at all. Held-out AUC for LR under sgd improves from 0.63015 to 0.68146.
- **FTRL's `z` accumulator was initialised to 1.0 instead of 0.0**, so the first
  update to any weight drove it to roughly -0.15 regardless of the gradient,
  discarding the random initialisation and giving the linear weights a
  systematic negative bias. Affected LR, FM and FFM alike under `-p ftrl`.
  Held-out AUC for FFM under ftrl improves from 0.69314 to 0.69419, with
  non-overlapping ranges across five seeds.
- **The `.bin` cache was validated against the source text file, not against its
  own layout**, so a cache written by a build with a different row encoding,
  sitting beside an unchanged `.txt`, passed the check and was read back as
  garbage.
- **`Reader::block_` was freed twice**, and on the binary path freed while
  uninitialised: allocated with `malloc`, released with `delete []`, then
  released again by `Clear()`. It is now owned by a `scoped_array`.
- **`DMatrix::Reset()` called `STLDeleteElementsAndClear` from inside its own
  loop**, running out of bounds after the first iteration.
- **`FromDMReader` double-freed the caller's matrix**, because `data_samples_`
  aliased a matrix owned by the caller and both destructors ran.
- **`GetMiniBatch` leaked the row `AddRow()` allocated** before overwriting the
  slot.
- **AUC allocated a million-bucket histogram per thread per evaluation** — 160
  MB of zeroed allocation per validation pass at 20 threads — to spread a few
  tens of thousands of increments, then passed both histograms **by value** to a
  function that ignored the arguments and read the members anyway.
- **AUC returned NaN when one class was absent.** NaN compares false against
  everything, so it told early stopping that no epoch had ever improved. It now
  returns 0.5, the value a coin flip earns.
- **The binary aborted at startup whenever neither `USER` nor `USERNAME` was
  set** — the normal case under Docker, CI, cron and systemd — by constructing a
  `std::string` from a null `getenv`.

### Removed

- The distributed-computation path: `Loss::CalcGradDist` had a commented-out
  body, and `DMatrix::Compress`, `DMatrix::GetMiniBatch` and `feature_map`
  existed only to serve it.
- `struct Node`, `SparseRow`, and the per-Example heap allocation they implied.
- Four unused scratch pointers on `FFMScore` and three on `FMScore`.
