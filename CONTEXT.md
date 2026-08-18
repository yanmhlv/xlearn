# xLearn

A library for training and applying linear, factorization machine, and
field-aware factorization machine models on large-scale sparse data, with a
C++ core, a command-line interface, and a Python binding.

## Language

### The data

**Example**:
One labelled observation — a set of feature values plus the target value the
model is asked to reproduce.
_Avoid_: sample, instance, record, row, case

**Row**:
The storage slot an Example occupies inside a DMatrix. Row is a position; the
Example is what sits there.

**Feature**:
One measurable input dimension, identified by a zero-based feature id.

**Feature Value**:
The magnitude a Feature takes in a given Example. Absent features are omitted
rather than stored as zero.

**Field**:
A group of mutually exclusive Features that came from the same original
categorical variable. Only FFM models use fields; linear and FM models leave
every field id at zero.
_Avoid_: group, category, namespace

**Label**:
The target value of an Example — `+1`/`0`/`-1` for classification, an arbitrary
real for regression. A dataset used only for prediction has no labels.
_Avoid_: y, target, ground truth

**Node**:
One non-zero entry of an Example — a feature id, its feature value, and its
field id. A DMatrix stores these as parallel columns rather than as a struct,
and omits a column the data never varies.

**Row Reference**:
A borrowed view of the Nodes belonging to one Example: the columns and the
length they share, owning nothing and outliving nothing. An Example may
legitimately have zero Nodes while still carrying a Label.
_Avoid_: sparse row, row view, slice

**DMatrix**:
A contiguous run of Examples held in memory, with their Labels and Norms. The
same structure serves as a whole dataset, a Block streamed off disk, and the
Python caller's input.
_Avoid_: dataset (a DMatrix may hold only part of one), matrix

**Norm**:
A per-Example scaling factor applied to that Example's contribution during
scoring, so that Examples with many Features do not dominate those with few.
Referred to as instance-wise normalization, which is the term of art.

**Block**:
The bounded slice of a data file read into memory at one time, sized in
megabytes. Blocks are what make on-disk training possible.
_Avoid_: chunk, buffer, batch

**Data Format**:
The on-disk text encoding of a dataset — `libsvm`, `libffm`, or `csv`. Detected
from the file rather than declared.

**Binary Cache**:
A serialized DMatrix written beside a text dataset, keyed by a hash of that
dataset, and reused on the next run to skip parsing.

### The model

**Model**:
The complete learned state of a training run: the Linear Term, the Latent
Factors, the Bias, their Gradient Caches, and the Model Family and Loss
Function they were trained under.
_Avoid_: weights, parameters, estimator

**Model Family**:
Which functional form the Model takes — `linear`, `fm`, or `ffm`. Chosen once
and baked into the Model; it cannot be changed after initialization.
_Avoid_: model type, task type, algorithm

**Linear Term**:
The per-Feature weight vector shared by every Model Family.
_Avoid_: w, coefficients

**Latent Factor**:
The learned vector attached to a Feature that gives it its interaction with
other Features. FM stores one per Feature; FFM stores one per Feature per
Field. Linear models have none.
_Avoid_: v, embedding, factor matrix

**Latent Dimension**:
The length of a single Latent Factor. Padded up to a multiple of the SIMD width
before it is stored.
_Avoid_: K, rank, num_K

**Bias**:
The single intercept term added to every Score.

**Gradient Cache**:
The per-parameter optimizer state stored alongside each weight — the
accumulated squared gradient for AdaGrad, or the accumulator and per-coordinate
counter for FTRL. It is part of the Model and travels with it.
_Avoid_: auxiliary, aux, momentum

**Checkpoint**:
A Model serialized to disk. The binary form round-trips exactly; the text form
is for inspection only.

**Pre-trained Model**:
A Checkpoint loaded at the start of a training run so that training continues
from it rather than from a random initialization.
_Avoid_: warm start, online learning

### Training and evaluation

**Score**:
The raw real-valued output of applying a Model to one Example, before any
squashing or thresholding.
_Avoid_: prediction (that is the transformed value), output, logit, and
especially *score* in the sense of "how good the model is" — that is a Metric

**Prediction**:
What is reported to the caller for one Example: the Score, optionally passed
through a sigmoid or reduced to a sign.

**Loss Function**:
The quantity training minimizes — `cross-entropy` for classification,
`squared` for regression. Its choice is what makes a run a classification or a
regression run.
_Avoid_: objective, cost, error function, task

**Metric**:
A measure of Model quality computed from Predictions and Labels, reported to
the user and never optimized against. Also what early stopping watches.
_Avoid_: score, evaluation function

**Optimizer**:
The rule that turns a gradient into a weight update — `sgd`, `adagrad`, or
`ftrl`. Its choice determines the size of the Gradient Cache.
_Avoid_: updater, solver, optimization method

**Epoch**:
One pass over every Example in the training set.

**Early Stop**:
Ending training at the epoch whose Metric on the validation set was best, once
a fixed number of subsequent epochs have failed to beat it, and rolling the
Model back to that epoch's weights.

**Cross-Validation**:
Training over a dataset split into folds, each fold serving in turn as the
validation set. Produces reported Metrics, not a saved Model.

**Lock-Free**:
Updating shared Model weights from several threads without synchronization,
accepting lost updates in exchange for throughput.

### Doing the work

**Solver**:
The entry point that turns a set of Hyper-Parameters into a completed training
or prediction run, owning every other collaborator for its duration.
_Avoid_: session, engine, context, driver

**Hyper-Parameters**:
Everything the caller chooses before a run: Model Family, Loss Function,
Optimizer settings, file paths, and run modes.
_Avoid_: config, options, params, settings

**Checker**:
Validates Hyper-Parameters — whether from the command line or from the Python
binding — and rejects the run before any work begins.
_Avoid_: validator

**Reader**:
Supplies successive DMatrices from one data source, and knows how that source
is traversed: entirely in memory, streamed off disk a Block at a time, or
handed over directly by the Python caller.
_Avoid_: loader, iterator, data source

**Parser**:
Turns a Block of raw text in one Data Format into Examples appended to a
DMatrix.
_Avoid_: decoder, tokenizer

**Trainer**:
Runs Epochs against a Model until training ends, whether by exhausting the
epoch count, by Early Stop, or by completing every Cross-Validation fold.

**Predictor**:
Applies a Model to a dataset that needs no training and emits the Predictions.
_Avoid_: inferencer, scorer
