# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

xLearn trains linear, factorization machine (FM) and field-aware factorization
machine (FFM) models on large-scale sparse data. C++23 core, two command line
tools, and a nanobind Python extension.

## Commands

`Taskfile.yml` is the entry point; `task --list` is the discovery surface. It
wraps CMake rather than replacing it — CMake is still the build system, and the
tasks exist to encode the arguments and working directories that are easy to get
wrong.

```sh
task build                            # configure and build; PRESET=dev|debug|dist
task test                             # every test
task test-one NAME=fm_score_test      # one test binary
task test-case PACKAGE=score NAME=fm_score_test CASE=FMScoreTest.calc_score
task bench NAME=score_bench FILTER=ffm
task test-python                      # install the extension, run the Python tests
task test-python PYTHON=/path/to/python3.12   # pick the interpreter
task clean                            # remove one preset's build tree
```

`PRESET` defaults to `dev` and selects the build tree for every C++ task:
`task test PRESET=debug`. The Python tasks ignore it — pip drives its own build
directory from `pyproject.toml`.

Presets are `dev` (Release, `-march=native`, tests on), `debug` (assertable) and
`dist` (no tests, no native arch — portable binaries).

The tasks are thin, and CI calls the underlying commands directly rather than
installing Task:

```sh
cmake --preset dev && cmake --build --preset dev -j
ctest --preset dev
```

Two arguments the tasks exist to get right, worth knowing when running the
commands by hand:

- **`ctest -R` is an unanchored regex.** `-R fm_score_test` also runs
  `ffm_score_test`; `test-one` anchors it to `^...$`.
- **A test binary must run from its own directory.** Tests write scratch files
  under fixed names into the working directory, which is why one binary is one
  ctest entry rather than one entry per case. `test-case` sets `dir:` for this.

Test binaries land in `build/<preset>/test/<package>/`, benchmarks in
`build/<preset>/bench/`. Benchmark names are lowercase and slash-separated
(`ffm/ftrl/grad/k=16`), so `FILTER` is matched against that shape.

`xlearn_train` and `xlearn_predict` land at the top of the build tree, with the
criteo sample data and `run_example.sh` staged beside them (`demo/CMakeLists.txt`).

`compile_commands.json` is written to `build/<preset>/` — point clangd at it
with `--compile-commands-dir=build/dev` or a symlink at the project root.

### Python

`task test-python` runs the whole loop. By hand it is:

```sh
python -m pip install '.[sklearn]' pandas   # scikit-build-core drives cmake
python -m unittest discover -s python-package/test -p 'test_data_conversion.py'
cd demo/classification/criteo_ctr && python ../../../python-package/test/test_python.py
```

`test_data_conversion.py` imports scipy and sklearn, so the `sklearn` extra is
not optional for the tests. `PYTHON` defaults to `python3`; set it when that is
not a 3.12+ interpreter, which `requires-python` demands. The criteo test writes
`*.bin`, `model.out` and `output.txt` beside its data — untracked, and worth
deleting after a manual run.

`test_python.py` reads its dataset from the working directory, so it must run
from `demo/classification/criteo_ctr`. `pyproject.toml` forces
`XLEARN_BUILD_PYTHON=ON`, tests off and native arch off; the extension builds
against the stable ABI from 3.12, so one abi3 wheel covers every supported
Python.

## Architecture

A run is `Solver` (`src/solver/solver.h`) constructing everything and holding it
for the run's duration:

```
Solver -> Checker         validates HyperParam, rejects before any work
       -> Reader          Inmem | Ondisk (a Block at a time) | FromDM (Python)
            -> Parser     libsvm | libffm | csv, detected from the file
            -> DMatrix    Examples, Labels, Norms
       -> Trainer         epochs, early stop, cross-validation folds
            -> Loss       cross-entropy | squared; fans rows over the ThreadPool
                 -> Score linear | fm | ffm; reads and updates the Model
       -> Metric          reported, never optimized against
       -> Model           linear term, latent factors, bias, gradient caches
```

`Inference` (`src/solver/inference.h`) is the prediction counterpart of `Trainer`.

### Packages

One static library per `src/<dir>`, declared by `xlearn_add_library(<name>)` and
reachable everywhere as `xlearn::<name>`. Headers are included by their path
from the project root (`#include "src/score/fm_score.h"`). Dependencies come
from `FetchContent` at pinned tags — nothing is vendored or installed.

`src/base` is the substrate: `simd.h` (Highway wrapper), `thread_pool.h`,
`class_register.h`, logging, file and string helpers.

### The class registry

Polymorphic families are constructed from strings, not switch statements
(`src/base/class_register.h`). A subclass calls `REGISTER_SCORE("fm", FMScore)`
in its `.cc`; `solver.cc` calls `CREATE_SCORE(hyper_param_.score_func)`. The
registered string *is* the hyper-parameter value the user types.

Adding a variant is usually **three** edits, not two: the subclass, its
`REGISTER_*`, and `src/solver/checker.cc`, which validates the name before the
registry ever sees it. Metrics (`-x`) and optimizers (`-o`) are checked
against string whitelists there; score and loss are not named on the command
line at all but derived together from the `-s <0-5>` task switch, so a fourth
model family means extending that switch and its help text.

Registries: `REGISTER_SCORE`, `REGISTER_LOSS`, `REGISTER_METRIC`,
`REGISTER_PARSER`, `REGISTER_READER`.

### Threading

`Loss::CalcGrad` slices the rows across `ThreadPool` and calls `Sync`. With
`lock_free` off it runs one slice; on, it runs `threadNumber_` slices that
update shared model weights without synchronization, accepting lost updates for
throughput.

## Invariants that span files

Get these wrong and the failure is silent, not a compile error.

- **`RowRef` is borrowed and short-lived.** It points into `DMatrix` columns
  that grow, so the next `AddNode` invalidates it. `AddNode` also has no random
  access by row id — a caller filling rows out of order must buffer or sort.
- **Columns the data never varies are elided.** libsvm data stores no field
  column, one-hot data stores no value column; `RowRef` accessors return the
  implied value. Never read the underlying arrays directly.
- **Feature and field ids share one interleaved array** with a stride of 1 or 2.
  `RowRef::feat` and `RowRef::field` multiply by that stride.
- **SIMD width is chosen per call from `aligned_k`, never fixed.** `kAlign = 4`
  padding of K is baked into the serialized model, so the width has to divide
  whatever the padding produced. `Vec<N>` loads and stores are unaligned by
  design — see the header comment in `src/base/simd.h` before changing either.
- **`Score::OptType` is resolved once in `Initialize`**, not matched per row.
- **The fused `Step()` path is opt-in per score function.** `PrefersFusedStep()`
  returns true only for FM; routing LR through it measured 17% slower.

### Serialized formats

Both carry a magic and a version, and both are versioned deliberately:

- `.model` — `kModelVersion` in `src/data/model_parameters.h`. Bumping it
  invalidates every user's checkpoint; say so in `CHANGELOG.md`.
- `.bin` data cache — `kDMatrixVersion` in `src/data/data_structure.h`. Caches
  regenerate on their own, but the version must still be bumped: the hashes
  guarding the cache are hashes of the *text* file, so a changed row layout
  would otherwise be read back as this layout's rows.

Any change to `DMatrix` serialization or `Model` serialization bumps the
matching constant.

## Vocabulary

`CONTEXT.md` is the project's glossary — Example, Row, Node, RowRef, DMatrix,
Block, Score, Prediction, Metric, Model Family, Gradient Cache, and the rest,
each with an explicit *Avoid* list. Use those terms in code, identifiers,
commit messages and docs, and prefer them over the synonyms a paper or another
library would use.

## Performance work

`PERFORMANCE.md` records every optimization that was kept and the eight that
were implemented, measured and rejected — each with the problem, the decision,
the measurement, and what it costs. Read the relevant record before reshaping a
hot path; several of the obvious-looking alternatives are already in the
rejected half.

Its harness section is binding on any speed or quality claim made here:
per-epoch time is a slope (wall time at 2 epochs and at 6, differenced and
divided by 4), the `.bin` cache is warmed by a throwaway run first, builds are
interleaved rather than run in blocks, one pinned core and a single thread, and
quality is scored outside the engine. The noise floor is about ±2.5% (±5% on
the 148k-feature dataset); a smaller difference is not a result. A new
optimization gets a new record.

The decision-site comments in the hot paths are part of that record —
`PERFORMANCE.md` is the index into them. Do not strip them when editing nearby.
