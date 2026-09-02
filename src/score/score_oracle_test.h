#ifndef XLEARN_SCORE_SCORE_ORACLE_TEST_H_
#define XLEARN_SCORE_SCORE_ORACLE_TEST_H_

#include <cmath>
#include <random>
#include <vector>

#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"

namespace xLearn {
namespace oracle {

struct Node {
  index_t feat;
  index_t field;
  real_t val;
};

struct Row {
  std::vector<Node> nodes;

  RowBuffer Buffer() const {
    RowBuffer buf;
    for (const Node& n : nodes) {
      buf.Add(n.feat, n.val, n.field);
    }
    return buf;
  }
};

inline Row MakeRow(index_t num_feat, index_t num_field, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<real_t> value(-2.0, 2.0);
  Row row;
  for (index_t i = 0; i < num_feat; ++i) {
    row.nodes.push_back({i, i % num_field, value(gen)});
  }
  return row;
}

inline void FillDistinct(Model& model, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<real_t> value(-0.5, 0.5);
  real_t* w = model.GetParameter_w();
  for (index_t i = 0; i < model.GetNumParameter_w(); ++i) {
    w[i] = value(gen);
  }
  real_t* v = model.GetParameter_v();
  for (index_t i = 0; i < model.GetNumParameter_v(); ++i) {
    v[i] = value(gen);
  }
  model.GetParameter_b()[0] = value(gen);
}

// The padding between num_K and aligned_k is never read by a correct kernel,
// so leaving it random would let a kernel that reads past num_K disagree with
// the oracle for a reason the oracle cannot express. Zeroed, the two agree
// exactly when the kernel respects the boundary.
inline void ZeroLatentPadding(Model& model) {
  const index_t k = model.GetNumK();
  const index_t aligned_k = model.get_aligned_k();
  if (k == aligned_k) return;
  const index_t aux = model.GetAuxiliarySize();
  real_t* v = model.GetParameter_v();
  const index_t planes = model.GetNumParameter_v() / aligned_k;
  for (index_t p = 0; p < planes; ++p) {
    if (p % aux != 0) continue;
    for (index_t d = k; d < aligned_k; ++d) {
      v[p * aligned_k + d] = 0.0;
    }
  }
}

inline double LinearAndBias(const Row& row, Model& model, double norm) {
  const double sqrt_norm = std::sqrt(norm);
  const index_t aux = model.GetAuxiliarySize();
  const index_t num_feat = model.GetNumFeature();
  const real_t* w = model.GetParameter_w();
  double sum = 0.0;
  for (const Node& n : row.nodes) {
    if (n.feat >= num_feat) continue;
    sum += double(n.val) * double(w[n.feat * aux]) * sqrt_norm;
  }
  return sum + double(model.GetParameter_b()[0]);
}

// An FFM pair reads each feature's latent block for the *other* feature's
// field: v[i][field_j] against v[j][field_i]. Written from that definition
// rather than from the kernel, so a kernel that swapped the two, or dropped
// the field offset entirely, disagrees here.
inline double FFMScoreOf(const Row& row, Model& model, double norm) {
  const index_t k = model.GetNumK();
  const index_t aligned_k = model.get_aligned_k();
  const index_t aux = model.GetAuxiliarySize();
  const index_t num_feat = model.GetNumFeature();
  const index_t num_field = model.GetNumField();
  const real_t* v = model.GetParameter_v();
  const index_t block = num_field * aux * aligned_k;

  double sum = 0.0;
  for (size_t i = 0; i < row.nodes.size(); ++i) {
    const Node& a = row.nodes[i];
    if (a.feat >= num_feat || a.field >= num_field) continue;
    for (size_t j = i + 1; j < row.nodes.size(); ++j) {
      const Node& b = row.nodes[j];
      if (b.feat >= num_feat || b.field >= num_field) continue;
      const real_t* wa = v + a.feat * block + b.field * aux * aligned_k;
      const real_t* wb = v + b.feat * block + a.field * aux * aligned_k;
      double dot = 0.0;
      for (index_t d = 0; d < k; ++d) {
        dot += double(wa[d]) * double(wb[d]);
      }
      sum += dot * double(a.val) * double(b.val) * norm;
    }
  }
  return sum + LinearAndBias(row, model, norm);
}

// FM keeps one latent block per feature. The normalizer goes on each factor
// as sqrt(norm), so a pair carries exactly one norm -- the same convention
// the linear term uses.
inline double FMScoreOf(const Row& row, Model& model, double norm) {
  const index_t k = model.GetNumK();
  const index_t aligned_k = model.get_aligned_k();
  const index_t aux = model.GetAuxiliarySize();
  const index_t num_feat = model.GetNumFeature();
  const real_t* v = model.GetParameter_v();

  double sum = 0.0;
  for (size_t i = 0; i < row.nodes.size(); ++i) {
    const Node& a = row.nodes[i];
    if (a.feat >= num_feat) continue;
    for (size_t j = i + 1; j < row.nodes.size(); ++j) {
      const Node& b = row.nodes[j];
      if (b.feat >= num_feat) continue;
      const real_t* wa = v + a.feat * aux * aligned_k;
      const real_t* wb = v + b.feat * aux * aligned_k;
      double dot = 0.0;
      for (index_t d = 0; d < k; ++d) {
        dot += double(wa[d]) * double(wb[d]);
      }
      sum += dot * double(a.val) * double(b.val) * norm;
    }
  }
  return sum + LinearAndBias(row, model, norm);
}

// Float accumulation over k*pairs terms drifts from the double oracle, and
// the kernel reorders that sum across lanes and chains by design. Scale the
// tolerance with the work rather than asserting a fixed epsilon.
inline double Tolerance(double magnitude, index_t terms) {
  return 1e-5 * terms + 1e-4 * std::fabs(magnitude);
}

}  // namespace oracle
}  // namespace xLearn

#endif  // XLEARN_SCORE_SCORE_ORACLE_TEST_H_
