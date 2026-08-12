<img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "400"/>

[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](./LICENCE)
[![Project Status](https://img.shields.io/badge/version-0.4.4-green.svg)]()

## What is xLearn?

xLearn is a ***high performance***, ***easy-to-use***, and ***scalable*** machine learning package that contains linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM), all of which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data. Many real world datasets deal with high dimensional sparse feature vectors like a recommendation system where the number of categories and users is on the order of millions. In that case, if you are the user of liblinear, libfm, and libffm, now xLearn is your another better choice.

[Get Started! (English)](http://xlearn-doc.readthedocs.io/en/latest/index.html)

[Get Started! (中文)](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html)

### Performance

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/speed.png" width = "800"/>

xLearn is developed by high-performance C++ code with careful design and optimizations. Our system is designed to maximize CPU and memory utilization, provide cache-aware computation, and support lock-free learning. By combining these insights, xLearn is 5x-13x faster compared to similar systems.

**This fork goes further.** Examples are stored as compressed sparse columns instead of a heap-allocated vector of nodes each, the SIMD width is chosen per call rather than fixed at four lanes, FFM latent blocks are planar, and each Example's features and a large model's parameters are fetched ahead of the Example that needs them. Measured against the previous release on 300k Examples of synthetic CTR data, one pinned core, single thread:

| model | time, k=8 | time, k=4 | peak memory |
|---|---|---|---|
| LR | −51% to −62% | −52% to −60% | 0.40x |
| FM | −52% to −59% | −53% to −59% | 0.40x |
| FFM | −31% to −42% | −24% to −40% | 0.61x |

Held-out AUC is equal or better in 8 of 9 model/optimizer combinations, and two long-standing correctness bugs — a shuffle that produced the same order every epoch, and an FTRL accumulator initialised to the wrong value — are fixed along the way.

[PERFORMANCE.md](PERFORMANCE.md) records each optimization, the measurement that settled it, and the eight approaches that were implemented, measured and rejected. [CHANGELOG.md](CHANGELOG.md) has the full list of changes.

### Ease-of-use

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/code.png" width = "600"/>

Users can just clone the code and compile it by using cmake — the one runtime dependency, [Google Highway](https://github.com/google/highway), is fetched at configure time, so there is nothing to install first. Also, xLearn supports very simple Python and CLI interface for data scientists, and it also offers many useful features that have been widely used in machine learning and data mining competitions, such as cross-validation, early-stop, etc.

#### Dependencies

| | | |
|---|---|---|
| [Google Highway](https://github.com/google/highway) 1.4.0 | portable SIMD behind `src/base/simd.h` | always |
| [GoogleTest](https://github.com/google/googletest) 1.18.0 | unit tests | `XLEARN_BUILD_TESTS=ON` |
| [Google Benchmark](https://github.com/google/benchmark) 1.9.5 | kernel microbenchmarks | `XLEARN_BUILD_TESTS=ON` |

All three are fetched by CMake's `FetchContent` at configure time and pinned to a tag; none needs to be installed, vendored, or carried as a submodule. Highway is the only one linked into the shipped binaries, and it is header-and-static-library only. Why it is used at all, rather than raw intrinsics or plain auto-vectorization, is in [PERFORMANCE.md](PERFORMANCE.md).

### Scalability

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/scalability.png" width = "650"/>

xLearn can be used for solving large-scale machine learning problems. xLearn supports out-of-core training, which can handle very large data (TB) by just leveraging the disk of a PC.

## How to Contribute

xLearn has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

 * Please contribute if you find any bug in xLearn.
 * Contribute new features you want to see in xLearn.
 * Contribute to the tests to make it more reliable.
 * Contribute to the documents to make it clearer for everyone.
 * Contribute to the examples to share your experience with other users.
 * Open issue if you met problems during development.

Note that, please post iusse and contribution in *English* so that everyone can get help from them.

## What's New

 - 2026-08-12 Training is 1.3-2.6x faster and uses 1.6-2.5x less memory. Main update:

    * Examples stored as compressed sparse columns, with columns the data never varies elided
    * Per-call SIMD width, planar FFM latent blocks, and Example-ahead prefetching
    * Fixed a shuffle that produced the same order every epoch, and FTRL's `z` initialisation
    * Versioned the `.model` and `.bin` formats, which had neither magic bytes nor a version

    **Breaking:** `.model` checkpoints trained at k > 4 must be retrained — see
    [CHANGELOG.md](CHANGELOG.md). `.bin` data caches regenerate on their own.
    [PERFORMANCE.md](PERFORMANCE.md) explains what was optimized and what was tried
    and rejected.

 - 2019-10-13 [Andrew Kane](https://github.com/ankane) add [Ruby bindings](https://github.com/ankane/xlearn) for xLearn!

 - 2019-4-25 xLearn 0.4.4 version release. Main update:

    * Support Python DMatrix
    * Better Windows support
    * Fix bugs in previous version

 - 2019-3-25 xLearn 0.4.3 version release. Main update:
    * Fix bugs in previous version

 - 2019-3-12 xLearn 0.4.2 version release. Main update:
    * Release Windows version of xLearn

 - 2019-1-30 xLearn 0.4.1 version release. Main update:
    * More flexible data reader

 - 2018-11-22 xLearn 0.4.0 version release. Main update:

    * Fix bugs in previous version
    * Add online learning for xLearn

 - 2018-11-10 xLearn 0.3.8 version release. Main update:

    * Fix bugs in previous version.
    * Update early-stop mechanism.

 - 2018-11-08. xLearn gets 2000 star! Congs!

 - 2018-10-29 xLearn 0.3.7 version release. Main update:

    * Add incremental Reader, which can save 50% memory cost.

 - 2018-10-22 xLearn 0.3.5 version release. Main update:

    * Fix bugs in 0.3.4.

 - 2018-10-21 xLearn 0.3.4 version release. Main update:

    * Fix bugs in on-disk training.
    * Support new file format.

 - 2018-10-14 xLearn 0.3.3 version release. Main update:

    * Fix segmentation fault in prediction task.
    * Update early-stop meachnism.

 - 2018-09-21 xLearn 0.3.2 version release. Main update:

    * Fix bugs in previous version
    * New TXT format for model output

 - 2018-09-08 xLearn uses the new logo:

 <img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "300"/>

 - 2018-09-07 The [Chinese document](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html) is available now!

 - 2018-03-08 xLearn 0.3.0 version release. Main update:

    * Fix bugs in previous version
    * Solved the memory leak problem for on-disk learning
    * Support TXT model checkpoint
    * Support Scikit-Learn API

 - 2017-12-18 xLearn 0.2.0 version release. Main update:

    * Fix bugs in previous version
    * Support pip installation
    * New Documents
    * Faster FTRL algorithm

 - 2017-11-24 The first version (0.1.0) of xLearn release !
