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
- **Every `.model` checkpoint from an earlier build is refused, and the format
  version is now 2.** The FM normalization fix below changes what a stored FM
  latent weight means without changing where it sits, so an old file loads with
  nothing to fail on and scores differently than it did when it was validated.
  That is the same silent wrongness the k > 4 layout change produced, and it
  gets the same refusal rather than a note in a changelog nobody reads at
  prediction time. Linear and FFM checkpoints are scored identically by this
  release and are refused only because the version is a single number for the
  whole file; retraining them changes nothing but the version they carry.
- **A tuned `-lambda_2` no longer means what it did.** With the double
  application removed, `lambda_2` reaches the weights only through ftrl's
  proximal denominator, beside `(beta + sqrt(n)) / alpha`. That term grows with
  the number of updates a coordinate has seen and is order 100 by the end of a
  run, so the useful range for `lambda_2` moved up by roughly four orders of
  magnitude. A value carried over from an earlier build is not a weaker
  regularizer than it was — it is no regularizer at all.

### Migration

- **`.bin` data caches need no action.** A cache is derived from a text file and
  keyed by its content hash, so an unrecognised version is treated as "no cache
  present" and the text is re-parsed. Stale caches are silently regenerated.
- **Model files must all be retrained.** There is no converter. See the format
  version note above for why linear and FFM files are included.
- **Retune `-lambda_2` if you had tuned it**, upward by about four orders of
  magnitude. On 300k Examples of synthetic CTR data the held-out optimum moved
  from `0.0002` to `10` for FFM and from the `0.00002` default to `5`-`10` for
  FM. Left at the default this release simply trains without L2 on the ftrl
  path, which is a defensible default but a different one than before.
- **Lower `-r` for `fm` under `-p sgd`.** Correcting the FM normalizer makes the
  pairwise term larger by a factor of `||x||^2`, and plain sgd is the one
  optimizer that does not adapt its own step to compensate. On the same data the
  held-out optimum moved from `0.2` to about `0.05`; left at `0.2` an FM sgd run
  overshoots and loses about 0.023 AUC. `adagrad` and `ftrl` rescale themselves
  and both improve at their existing settings, and `adagrad` is the default.

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

#### Dependencies

xLearn previously vendored nothing and linked nothing. It now has one runtime
dependency and two test-only ones, all fetched by CMake's `FetchContent` at
configure time and pinned to a tag — nothing to install, vendor, or carry as a
submodule.

- **[Google Highway](https://github.com/google/highway) 1.4.0**, the portable
  SIMD layer behind `src/base/simd.h`, and the only one linked into the shipped
  binaries. Chosen over per-ISA intrinsics, which would have made each kernel
  two or three `#ifdef` branches to keep in step, and over plain
  auto-vectorization, which does not reliably fuse a multiply and an add that
  reach it as separate operations. `HWY_ENABLE_TESTS`, `EXAMPLES`, `CONTRIB` and
  `INSTALL` are forced off, so only the library is built.
- **[GoogleTest](https://github.com/google/googletest) 1.18.0** and
  **[Google Benchmark](https://github.com/google/benchmark) 1.9.5**, fetched
  only when `XLEARN_BUILD_TESTS=ON`.

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

- **FTRL applied `lambda_2` twice, over-regularizing every model trained with
  it.** The proximal step already carries L2 in the weight denominator
  (`(beta + sqrt(n))/alpha + lambda_2`), and the gradient handed to it added
  `lambda_2 * weight` on top. The gradient fed to an FTRL update is the loss
  gradient alone, so with no loss gradient at all a nonzero weight still moved
  `n` and `z` as though an example had arrived, and was then penalized again on
  the way out. Affected the linear and bias terms of all three score functions
  and the FM and FFM latent kernels; the bias alone was already correct. The
  penalty grows as `lambda_2` is tuned up, so the symptom was that raising it
  hurt more than it should. `lambda_2` defaults to `0.00002`, so every default
  FTRL run was affected. How much it hurt is easiest to see at the top of the
  range: from `lambda_2 = 0.002` up, the old FM and FFM held-out AUC lands on
  the linear model's number exactly, because the latent factors are driven to
  zero and the model quietly degenerates to a GLM. Measured on 300k Examples of
  synthetic CTR data at k = 8, with each build at its own best `lambda_2`, FM
  ftrl goes from 0.7085 to 0.7500 AUC and 0.5495 to 0.5288 logloss. FFM ftrl is
  a wash on ranking and better calibrated: 0.6995 to 0.6984 AUC, 0.5653 to
  0.5595 logloss.
- **FM applied the row normalizer once per factor instead of once per pair**, so
  a pairwise term carried `norm` twice and was weighted by `1/||x||^4` where
  normalization calls for `1/||x||^2`. The linear term beside it already used
  `sqrt(norm)`, and FFM already applied exactly one `norm` per pair, so FM was
  the only one of the three that disagreed — by a factor of `||x||^-2`, varying
  per row. Score and gradient shared the error, so training converged cleanly on
  a differently-scaled objective and nothing reported a problem. Affects FM
  whenever `norm` is on, which is the default. An FM model trained by an earlier
  build scores differently under this one, which is why the checkpoint version is
  bumped above rather than left to a warning. Held-out numbers move, and on 300k
  Examples of synthetic CTR data at k = 8 they move up wherever the optimizer
  adapts its own step: FM adagrad gains 0.019 AUC at stock settings and FM ftrl
  0.042 at a tuned `lambda_2`. FM sgd is the exception and needs `-r` retuned;
  see **Migration**. Linear and FFM are unaffected by this fix, and their sgd and
  adagrad runs are bit-identical across the change.
- **A `.bin` cache truncated inside its last vector was read back as zeros
  rather than refused.** `ReadVectorFromFile` sized the vector from the length
  field and then discarded the payload read's return value, so the tail that
  never arrived was value-initialised. `norm` is the last vector written, and
  every check on the body compares sizes drawn from length fields written before
  the truncation point — so that one vector had no check that could fire. At
  `norm = 0` a row contributes nothing to the score and produces no feature
  gradient: the affected rows trained as if empty, with only the bias moving.
  Reads that must consume a full record now fail on a short one, and both
  `DMatrix::Serialize` and `Model::Serialize` write under a pending name and
  rename onto the final one, so a run killed mid-write leaves no file for the
  next run to pick up.
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
