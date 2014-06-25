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

#ifndef VIDEO_SEGMENT_SEGMENTATION_HISTOGRAMS_H__
#define VIDEO_SEGMENT_SEGMENTATION_HISTOGRAMS_H__

#include "base/base.h"

#include <array>

namespace segmentation {

// Region's color histogram for Lab color space. Supports dense as well as sparse
// representation. Dense histograms are more efficient to be populated and can be
// converted to sparse representation (irreversible).
//
// Usage example:
//   ColorHistogram color_hist;
//   for ( <some pixels > ) {
//     const uint8_t* pixel = <pointer to LAB uint8_t data>
//     color_hist.AddPixel(pixel)   OR
//     color_hist.AddPixelInterpolated(pixel, weight)    <- per pixel weight.
//   }
//
//   // Save some memory.
//   color_hist.ConvertToSparse();
//   // Ensure L1 norm is 1
//   color_hist.NormalizeToOne();
//
//   ColorHistogram color_hist2;
//
//   // Determine distance between histograms
//   const float distance = color_hist.ChiSquareDist(color_hist2);
class ColorHistogram {
public:
  // Creates new ColorHistogram of dimension:
  // lum_bins x color_bins x color_bins.
  ColorHistogram(int lum_bins, int color_bins, bool sparse = false);
  ColorHistogram(const ColorHistogram& rhs) = default;
  ~ColorHistogram() = default;

  // Adds pixel with to a single bin (rounding towards zero computation).
  void AddPixel(const uint8_t* pix);
  void AddPixelInterpolated(const uint8_t* pix, float weight = 1.0f);

  // Same as above direct values (within bounds [0, 255], not checked).
  void AddPixelValuesInterpolated(float lum,
                                  float color_1,
                                  float color_2,
                                  float weight = 1.0f);

  // Generalized version, no bound checking is performed.
  void AddValueInterpolated(float x_bin, float y_bin, float z_bin, float weight);

  // Non-reversible operation - frees memory.
  void ConvertToSparse();
  bool IsSparse() const { return is_sparse_; }

  // Merges two histograms. Merging is supported for both dense and sparse as well as
  // normalized and unnormalized histograms. However, both histograms need to be of the
  // same type, e.g. both sparse and normalized.
  void MergeWithHistogram(const ColorHistogram& rhs);

  // Normalizes weights such that their L1 norm is 1.
  // Note: This does not change weight_sum_ to support merging normalize histograms.
  // IMPORTANT: After normalization to one no further items should be added via AddPixel.
  // This is only checked in debug mode.
  void NormalizeToOne();

  bool IsNormalized() const { return is_normalized_; }

  // Returns scaled histogram, i.e. scaling each bin location by corresponding
  // gain (one for each channel). Effectively shifts the histogram bins.
  ColorHistogram ScaleHistogram(const std::vector<float>& gain) const;

  float L2Dist(const ColorHistogram& rhs) const;

  // These distance should be evaluated only on normalized histograms.
  // (checked in debug mode).
  float ChiSquareDist(const ColorHistogram& rhs) const;
  float KLDivergence(const ColorHistogram& rhs) const;
  float JSDivergence(const ColorHistogram& rhs) const;

  // Should be called on sparse, normalized histograms.
  void ComputeMeanAndVariance();

  int LuminanceBins() const { return lum_bins_; }
  int ColorBins() const { return color_bins_; }
  float WeightSum() const { return weight_sum_; }

  // Returns number of hash bins that are not zero.
  int NumSparseEntries() const { return sparse_bins_.size(); }

  // Call ComputeMeanAndVariance, before querying mean and variance
  float LuminanceMean() const { return mean_[0]; }
  float ColorMeanA() const { return mean_[1]; }
  float ColorMeanB() const { return mean_[2]; }

  // Trace of covariance matrix equals
  // L1 norm of radius of ellipse described by covariance matrix.
  float L1CovarianceRadius() const { return var_[0] + var_[1] + var_[2]; }

  // Non-assignable.
  ColorHistogram& operator=(const ColorHistogram&) = delete;

private:
  // Returns copy of the histogram without copying any data.
  ColorHistogram EmptyCopy() const;

  // Generic distance implementation for above distance. fun must return per
  // bin-pair distance.
  double GenericDistance(const ColorHistogram& rhs,
                         std::function<float(float,float)> fun) const;

  const int lum_bins_ = 0;
  const int color_bins_ = 0;
  const int total_bins_ = 0;

  // Total weight sum across all *added* elements.
  // Note: If Normalize to one is called the sum of all elements does not necessary equal
  // weight_sum.
  double weight_sum_ = 0.0;

  std::array<float, 3> mean_{ {0.f, 0.f, 0.f} };
  std::array<float, 3> var_{ {0.f, 0.f, 0.f} };

  bool is_sparse_ = false;
  bool is_normalized_ = false;

  // Maps bins index to value.
  typedef std::unordered_map<int, float> HashBins;
  HashBins sparse_bins_;
  std::vector<float> bins_;
};

// This is basically a 1D histogram with wrap around.
class VectorHistogram {
public:
  VectorHistogram(int angle_bins);
  VectorHistogram(const VectorHistogram& rhs) = default;
  ~VectorHistogram() = default;

  void AddVector(float x, float y);
  void IncrementBin(int bin_num);

  // Adds vector with wrap around.
  void AddVectorInterpolated(float x, float y);
  void IncrementBinInterpolated(float bin);
  void MergeWithHistogram(const VectorHistogram& rhs);

  // Normalizes by the number of vectors.
  void Normalize();
  // Normalizes L1 norm of histogram to one.
  void NormalizeToOne();

  float ChiSquareDist(const VectorHistogram& rhs) const;
  float L2Dist(const VectorHistogram& rhs) const;

  int NumVectors() const { return num_vectors_; }
  int NumBins() const { return num_bins_; }
  const float* BinValues() const { return bins_.data(); }

private:
  // Returns normalized angle in (0, 1) for passed (x, y) vector.
  float NormAngle(float x, float y);

  std::vector<float> bins_;
  const int num_bins_;
  int num_vectors_;
};

// TODO(grundman): We need a true 2D histogram for optical flow here. Would fix that
// flower garden sequence, heh? use log for magnitude. Check with HOG video paper.

}  // namespace segmentation.

#endif  // VIDEO_SEGMENT_SEGMENTATION_HISTOGRAMS_H__
