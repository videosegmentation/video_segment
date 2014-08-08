// Copyright (c) 2010-2014, The Video Segmentation Project
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the The Video Segmentation Project nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ---


#include "base/base_impl.h"
#include "segmentation/histograms.h"

namespace segmentation {

// Converts scalar bin index to 3 dim index. Used to scale ColorHistograms efficiently.
class ColorHistogramIndexLUT {
 public:
  // Pass number of bins in each dimension.
  ColorHistogramIndexLUT(int bin_x, int bin_y, int bin_z) :
    bin_x_(bin_x),
    bin_y_(bin_y),
    bin_z_(bin_z) {
    const int total_bins = bin_x * bin_y * bin_z;
    lut_.resize(total_bins);
    int idx = 0;
    for (int x = 0; x < bin_x; ++x) {
      for (int y = 0; y < bin_y; ++y) {
        for (int z = 0; z < bin_z; ++z, ++idx) {
          lut_[idx] = std::make_tuple(x, y, z);
        }
      }
    }
  }

  const std::tuple<int, int, int>& Ind2Sub(int index) const {
    return lut_[index];
  }

  int get_bin_x() const { return bin_x_; }
  int get_bin_y() const { return bin_y_; }
  int get_bin_z() const { return bin_z_; }

private:
  const int bin_x_;
  const int bin_y_;
  const int bin_z_;
  std::vector<std::tuple<int, int, int> > lut_;
};

// Factory class to create ColorHistogramIndexLUT (only one LUT per bin configuration).
class ColorHistogramIndexLUTFactory {
 public:
  // Default accessor for clients.
  static ColorHistogramIndexLUTFactory& Instance() { return instance_; }

  // Returns actual LUT for a bin configuration
  const ColorHistogramIndexLUT& GetLUT(int bin_x, int bin_y, int bin_z) {
    // Find table.
    for (const auto& table : tables_) {
      if (table->get_bin_x() == bin_x &&
          table->get_bin_y() == bin_y &&
          table->get_bin_z() == bin_z) {
        return *table;
      }
    }

    // Not found, insert.
    std::unique_ptr<ColorHistogramIndexLUT> new_lut(
        new ColorHistogramIndexLUT(bin_x, bin_y, bin_z));
    tables_.push_back(std::move(new_lut));
    return *tables_.back();
  }

 private:
  ColorHistogramIndexLUTFactory() = default;
  ColorHistogramIndexLUTFactory(const ColorHistogramIndexLUTFactory&) = delete;
  ColorHistogramIndexLUTFactory& operator=(const ColorHistogramIndexLUTFactory&) = delete;
  static ColorHistogramIndexLUTFactory instance_;

  vector<std::unique_ptr<ColorHistogramIndexLUT> > tables_;
};

ColorHistogramIndexLUTFactory ColorHistogramIndexLUTFactory::instance_;

// Color histogram implementation.
ColorHistogram::ColorHistogram(int lum_bins, int color_bins, bool sparse)
  : lum_bins_(lum_bins),
    color_bins_(color_bins),
    sq_color_bins_(color_bins * color_bins),
    total_bins_(lum_bins * color_bins * color_bins),
    is_sparse_(sparse) {
  if (sparse) {
    // Anticipate 10% load.
    sparse_bins_ = HashBins(total_bins_ / 10);
  } else {
    bins_.resize(total_bins_, 0);
  }
}

ColorHistogram ColorHistogram::EmptyCopy() const {
  ColorHistogram copy(lum_bins_, color_bins_, is_sparse_);
  copy.weight_sum_ = weight_sum_;
  copy.mean_ = mean_;
  copy.var_ = var_;
  copy.is_normalized_ = is_normalized_;
  return copy;
}

void ColorHistogram::AddPixel(const uint8_t* pixel) {
  DCHECK(!is_normalized_) << "Cannot add to histogram after normalization.";
  // Compute 3D bin position and increment.
  const int bin = PixelValueToBin(pixel);
  if (is_sparse_) {
    ++sparse_bins_[bin];
  } else {
    ++bins_[bin];
  }

  weight_sum_ += 1.0;
}

void ColorHistogram::AddValueInterpolated(float x_bin,
                                          float y_bin,
                                          float z_bin,
                                          float weight) {
  DCHECK(!is_normalized_) << "Cannot add to histogram after normalization.";

  // Get integer locations.
  const int int_x = x_bin;
  const int int_y = y_bin;
  const int int_z = z_bin;

  const float dx = x_bin - (float)int_x;
  const float dy = y_bin - (float)int_y;
  const float dz = z_bin - (float)int_z;

  // The bins each value falls between.
  const int int_x_bins[2] = { int_x, int_x + (dx >= 1e-6f) };
  const int int_y_bins[2] = { int_y, int_y + (dy >= 1e-6f) };
  const int int_z_bins[2] = { int_z, int_z + (dz >= 1e-6f) };

  // Corresponding interpolation weights.
  const float dx_vals[2] = { 1.0f - dx, dx };
  const float dy_vals[2] = { 1.0f - dy, dy };
  const float dz_vals[2] = { 1.0f - dz, dz };

  if (is_sparse_) {
    for (int x = 0; x < 2; ++x) {
      const int slice_bin = int_x_bins[x] * sq_color_bins_;
      for (int y = 0; y < 2; ++y) {
        const int row_bin = slice_bin + int_y_bins[y] * color_bins_;
        for (int z = 0; z < 2; ++z) {
          const int bin = row_bin + int_z_bins[z];
          const float value = dx_vals[x] * dy_vals[y] * dz_vals[z] * weight;
          sparse_bins_[bin] += value;
        }
      }
    }
  } else {
    for (int x = 0; x < 2; ++x) {
      const int slice_bin = int_x_bins[x] * sq_color_bins_;
      for (int y = 0; y < 2; ++y) {
        const int row_bin = slice_bin + int_y_bins[y] * color_bins_;
        for (int z = 0; z < 2; ++z) {
          const int bin = row_bin + int_z_bins[z];
          const float value = dx_vals[x] * dy_vals[y] * dz_vals[z] * weight;
          bins_[bin] += value;
        }
      }
    }
  }

  weight_sum_ += weight;
}

void ColorHistogram::AddPixelInterpolated(const uint8_t* pixel, float weight) {
  AddValueInterpolated((float)pixel[0] * (1.0f / 255.f) * (lum_bins_ - 1),
                       (float)pixel[1] * (1.0f / 255.f) * (color_bins_ - 1),
                       (float)pixel[2] * (1.0f / 255.f) * (color_bins_ - 1),
                       weight);
}

void ColorHistogram::AddPixelValuesInterpolated(float lum,
                                                float color_1,
                                                float color_2,
                                                float weight) {
  AddValueInterpolated(lum * (1.0f / 255.f) * (lum_bins_ - 1),
                       color_1 * (1.0f / 255.f) * (color_bins_ - 1),
                       color_2 * (1.0f / 255.f) * (color_bins_ - 1),
                       weight);
}

ColorHistogram ColorHistogram::ScaleHistogram(const vector<float>& gain) const {
  const ColorHistogramIndexLUT& lut =
    ColorHistogramIndexLUTFactory::Instance().GetLUT(
      lum_bins_, color_bins_, color_bins_);
  ColorHistogram result = EmptyCopy();
  if (!IsSparse()) {
    for (int i = 0; i < total_bins_; ++i) {
      const float value = bins_[i];
      if (value) {
        const std::tuple<int, int, int>& idx_3d = lut.Ind2Sub(i);
        const float bin_lum = std::min(lum_bins_ - 1.f, std::get<0>(idx_3d) * gain[0]);
        const float bin_col1 = std::min(color_bins_ - 1.f, std::get<1>(idx_3d) * gain[1]);
        const float bin_col2 = std::min(color_bins_ - 1.f, std::get<2>(idx_3d) * gain[2]);
        result.AddValueInterpolated(bin_lum, bin_col1, bin_col2, value);
      }
    }
  } else {
    for (const auto& bin : sparse_bins_) {
      const std::tuple<int, int, int>& idx_3d = lut.Ind2Sub(bin.first);
      const float bin_lum = std::min(lum_bins_ - 1.f, std::get<0>(idx_3d) * gain[0]);
      const float bin_col1 = std::min(color_bins_ - 1.f, std::get<1>(idx_3d) * gain[1]);
      const float bin_col2 = std::min(color_bins_ - 1.f, std::get<2>(idx_3d) * gain[2]);
      result.AddValueInterpolated(bin_lum, bin_col1, bin_col2, bin.second);
    }
  }
  DCHECK_LT(fabs(WeightSum() - result.WeightSum()), 1e-3f);

  return result;
}

void ColorHistogram::ConvertToSparse() {
  if (IsSparse()) {
    DLOG(WARNING) << "Conversion to sparse histogram of already sparse histogram "
                     "requested. Ignored.";
    return;
  }

  // Anticipate 10% load.
  sparse_bins_ = HashBins(total_bins_ / 10);
  for (int bin_idx = 0; bin_idx < total_bins_; ++bin_idx) {
    const float value = bins_[bin_idx];
    if (value != 0) {
      sparse_bins_[bin_idx] = value;
    }
  }

  // Free memory.
  bins_ = vector<float>();
  is_sparse_ = true;
}

void ColorHistogram::MergeWithHistogram(const ColorHistogram& rhs) {
  DCHECK(is_sparse_ == rhs.is_sparse_) << "Sparsity differs.";
  DCHECK(is_normalized_ == rhs.is_normalized_) << "Normalization differs.";

  const double n = weight_sum_ + rhs.weight_sum_;
  if (n == 0) {
    return;
  }

  // Weighted merge for normalized histograms.
  const float n_l = weight_sum_ / n;
  const float n_r = rhs.weight_sum_ / n;

  // New weight_sum equals sum of both.
  weight_sum_ = n;

  double weighted_bin_sum = 0;
  if (!IsSparse()) {
    if (IsNormalized()) {
      for (int i = 0; i < total_bins_; ++i) {
        bins_[i] = bins_[i] * n_l + rhs.bins_[i] * n_r;
        weighted_bin_sum += bins_[i];
      }

      // Re-Normalize.
      const float denom = 1.0f / weighted_bin_sum;
      for (float& bin : bins_) {
        bin *= denom;
      }
    } else {
      for (int i = 0; i < total_bins_; ++i) {
        bins_[i] += rhs.bins_[i];
      }
    }
  } else {
    // Sparse version.
    if (IsNormalized()) {
      for (auto& bin : sparse_bins_) {
        const auto rhs_bin_iter = rhs.sparse_bins_.find(bin.first);
        if (rhs_bin_iter != rhs.sparse_bins_.end()) {
          bin.second = bin.second * n_l + rhs_bin_iter->second * n_r;
        } else {
          bin.second *= n_l;
        }
        weighted_bin_sum += bin.second;
      }

      // Process rhs bins that we might have missed.
      for (const auto& rhs_bin : rhs.sparse_bins_) {
        const auto bin_iter = sparse_bins_.find(rhs_bin.first);
        if (bin_iter == sparse_bins_.end()) {
          weighted_bin_sum += (
            (sparse_bins_[rhs_bin.first] = rhs_bin.second * n_r));
        }
      }

      // Normalize.
      const float denom = 1.0f / weighted_bin_sum;
      for (auto& bin : sparse_bins_) {
        bin.second *= denom;
      }
    } else {
      for (auto& bin : sparse_bins_) {
        const auto rhs_bin_iter = rhs.sparse_bins_.find(bin.first);
        if (rhs_bin_iter != rhs.sparse_bins_.end()) {
          bin.second += rhs_bin_iter->second;
        }
      }

      // Process rhs bins that we might have missed.
      for (const auto& rhs_bin : rhs.sparse_bins_) {
        const auto bin_iter = sparse_bins_.find(rhs_bin.first);
        if (bin_iter == sparse_bins_.end()) {
          sparse_bins_.insert(rhs_bin);
        }
      }
    }
  }
}

void ColorHistogram::NormalizeToOne() {
  if (IsNormalized()) {
    DLOG(WARNING) << "Normalization of normalized histogram requested. Ignored.";
  }

  is_normalized_ = true;

  if (weight_sum_ == 0) {
    return;
  }

  const float denom = 1.0f / weight_sum_;
  if (!IsSparse()) {
    for (auto& bin : bins_) {
      bin *= denom;
    }
  } else {
    for (auto& bin : sparse_bins_) {
      bin.second *= denom;
    }
  }
}

double ColorHistogram::GenericDistance(
    const ColorHistogram& rhs,
    std::function<float(float,float)> fun) const {
  DCHECK(IsSparse() == rhs.IsSparse());
  double sum = 0;
  if (!IsSparse()) {
    for (int i = 0; i < total_bins_; ++i) {
      sum += fun(bins_[i], rhs.bins_[i]);
    }
  } else {
    // Sparse processing.
    for (const auto& bin : sparse_bins_) {
      const auto& rhs_bin_iter = rhs.sparse_bins_.find(bin.first);
      sum += fun(bin.second, rhs_bin_iter != rhs.sparse_bins_.end() ?
                             rhs_bin_iter->second : 0.0f);
    }

    // Process rhs bins that we might have missed.
    for (const auto& rhs_bin : rhs.sparse_bins_) {
      const auto& bin_iter = sparse_bins_.find(rhs_bin.first);
      if (bin_iter == sparse_bins_.end()) {
        sum += fun(0, rhs_bin.second);
      }
    }
  }

  return sum;
}

float ColorHistogram::ChiSquareDist(const ColorHistogram& rhs) const {
  DCHECK(IsNormalized() && rhs.IsNormalized());
  return 0.5 * GenericDistance(rhs, [](float a, float b) -> float {
    const float add = a + b;
    if (fabs(add) > 1e-12) {
      const float sub = a - b;
      return sub * sub / add;
    } else {
      return 0.0f;
    }
  });
}

float ColorHistogram::KLDivergence(const ColorHistogram& rhs) const {
  DCHECK(IsNormalized() && rhs.IsNormalized());
  const double eps = 1e-10;
  return 0.5 * GenericDistance(rhs, [eps](float a, float b) -> float {
    const double ratio = (a + eps) / (b + eps);
    return a * std::log(ratio) + b * std::log(1.0 / ratio);
  });
}


float ColorHistogram::JSDivergence(const ColorHistogram& rhs) const {
  DCHECK(IsNormalized() && rhs.IsNormalized());
  const double eps = 1e-10;
  return 0.5 * GenericDistance(rhs, [eps](float a, float b) -> float {
    const double inv_mean = 1.0 / ((a + b) * 0.5 + eps);
    const double ratio_a = (a + eps) * inv_mean;
    const double ratio_b = (b + eps) * inv_mean;
    return a * std::log(ratio_a) + b * std::log(ratio_b);
  });
}


float ColorHistogram::L2Dist(const ColorHistogram& rhs) const {
  return std::sqrt(GenericDistance(rhs, [](float a, float b) -> float {
    const float diff = a - b;
    return diff * diff;
  }));
}

void ColorHistogram::ComputeMeanAndVariance() {
  DCHECK(IsSparse()) << "Implemented for sparse histograms only.";
  DCHECK(IsNormalized()) << "Implemented for normalized histograms only.";

  mean_.fill(0);
  var_.fill(0);

  const ColorHistogramIndexLUT& lut = ColorHistogramIndexLUTFactory::Instance().GetLUT(
    lum_bins_, color_bins_, color_bins_);
  // Iteration uses simplification that all vals sum to one.
  for (const auto& bin : sparse_bins_) {
    const std::tuple<int, int, int>& idx_3d = lut.Ind2Sub(bin.first);
    const float val = bin.second;

    mean_[0] += std::get<0>(idx_3d) * val;
    mean_[1] += std::get<1>(idx_3d) * val;
    mean_[2] += std::get<2>(idx_3d) * val;

    var_[0] += std::get<0>(idx_3d) * std::get<0>(idx_3d) * val;
    var_[1] += std::get<1>(idx_3d) * std::get<1>(idx_3d) * val;
    var_[2] += std::get<2>(idx_3d) * std::get<2>(idx_3d) * val;
  }

  var_[0] -= mean_[0] * mean_[0];
  var_[1] -= mean_[1] * mean_[1];
  var_[2] -= mean_[2] * mean_[2];
}

VectorHistogram::VectorHistogram(int angle_bins) : num_bins_(angle_bins),
  num_vectors_(0) {
  bins_.resize(num_bins_, 0);
}

float VectorHistogram::NormAngle(float x, float y) {
  // ensure angle \in (0, 1)
  return atan2(y, x) / (2.0 * M_PI + 1e-4) + 0.5;
}

void VectorHistogram::AddVector(float x, float y) {
  bins_[NormAngle(x, y) * num_bins_] += hypot(x, y);
  ++num_vectors_;
}

void VectorHistogram::IncrementBin(int bin_num) {
  ++bins_[bin_num];
  ++num_vectors_;
}

void VectorHistogram::AddVectorInterpolated(float x, float y) {
  // ensure angle \in (0, 1)
  float bin = NormAngle(x, y) * num_bins_;
  float magn = hypot(x, y);
  float int_bin = floor(bin);

  const float bin_center[3] = { int_bin - 1.0f + 0.5f,
                                int_bin + 0.5f,
                                int_bin + 1.0f + 0.5f
                              };

  const int bin_idx[3] = { ((int)int_bin - 1 + num_bins_) % num_bins_,
                            (int)int_bin,
                           ((int)int_bin + 1) % num_bins_
                         };

  // Loop over.
  for (int i = 0; i < 3; ++i) {
    const float da = fabs(bin_center[i] - bin);
    if (da < 1) {
      bins_[bin_idx[i]] += (1 - da) * magn;
    }
  }

  ++num_vectors_;
}

void VectorHistogram::IncrementBinInterpolated(float bin) {
  const int bin_idx = bin;
  const float dx = bin - (float)bin_idx;
  bins_[bin_idx] += 1.0f - dx;
  if (bin_idx + 1 < num_bins_) {
    bins_[bin_idx + 1] += dx;
  } else {
    bins_[0] += dx;
  }

  ++num_vectors_;
}

void VectorHistogram::MergeWithHistogram(const VectorHistogram& rhs) {
  float n_l = num_vectors_;
  float n_r = rhs.num_vectors_;
  if (n_l + n_r > 0) {
    float n = 1.0f / (n_l + n_r);

    for (int i = 0; i < num_bins_; ++i) {
      bins_[i] = (bins_[i] * n_l + rhs.bins_[i] * n_r) * n;
    }

    num_vectors_ += rhs.num_vectors_;
    NormalizeToOne();
  }
}

float VectorHistogram::ChiSquareDist(const VectorHistogram& rhs) const {
  const float* pl = bins_.data();
  const float* pr = rhs.bins_.data();

  float add, sub;
  float sum = 0;

  for (int i = 0; i < num_bins_; ++i, ++pl, ++pr) {
    add = *pl + *pr;
    if (add) {
      sub = *pl - *pr;
      sum += sub * sub / add;
    }
  }

  return 0.5 * sum;
}

float VectorHistogram::L2Dist(const VectorHistogram& rhs) const {
  const float* pl = bins_.data();
  const float* pr = rhs.bins_.data();

  float diff;
  float sum = 0;

  for (int i = 0; i < num_bins_; ++i, ++pl, ++pr) {
    diff = *pl - *pr;
    sum += diff * diff;
  }

  return std::sqrt(sum);
}

void VectorHistogram::Normalize() {
  if (num_vectors_ > 0) {
    float weight = 1.0f / num_vectors_;
    for (int i = 0; i < num_bins_; ++i) {
      bins_[i] *= weight;
    }
  }
}

void VectorHistogram::NormalizeToOne() {
  float sum = 0;
  for (int i = 0; i < num_bins_; ++i) {
    sum += bins_[i];
  }

  if (sum > 0) {
    sum = 1.0 / sum;

    for (int i = 0; i < num_bins_; ++i) {
      bins_[i] *= sum;
    }
  }
}

}  // namespace segmentation.
