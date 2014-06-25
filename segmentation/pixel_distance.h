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

#ifndef VIDEO_SEGMENT_SEGMENTATION_PIXEL_DISTANCE_H__
#define VIDEO_SEGMENT_SEGMENTATION_PIXEL_DISTANCE_H__

#include "base/base.h"
#include <type_traits>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

namespace segmentation {

// Defines several types of templated distances, descriptors and traits that interact
// with each other:
// a) PixelDistanceFunction<type, channels>   (derived classes)
//    // Implements distance function between two pixel pointers ptr_1, ptr_2/
//    // Returns normalized distance in [0, 1].
//    inline float operator()(const T* ptr_1, const T* ptr_2)
//    
// b) PixelDescriptor<type, descriptor_size, input_size) 
//    // Implements storing descriptors inside RegionNodes via
//    static void SetDescriptor(const T* img_ptr, float* descriptor)
//    // Here img_ptr is valid for input_size elements
//    // and descriptor for descriptor_size elements
//
// c) [Spatial|TemporalDistance]
//    Aggregates PixelDistanceFunction and PixelDescriptor over an image represented
//    by cv::Mat. Spatial distances only evaluate *within* and image, whereas
//    temporal distances evaluate across images.
//    Each distance implements:
//    // Common interface for fast pixel distance evaluation in scanline order.
//    // // Sets position of current pixel location.
//    // void MoveAnchorTo(int x, int y);
//    // // Advances in scanline order.
//    // void IncrementAnchor();
//    // // Same for test pixel location.
//    // void MoveTestAnchorTo(int x, int y);
//    // void IncrementTestAnchor();
//    // Returns pixel feature distance between anchor and test-anchor displaced by
//    // dx, dy.
//    // dx, dy. Constraints:
//    // For spatial distance dx is in {-1, 0, 1} and dy in {0, 1}.
//    // For temporal distance both dx and dy are in {-1, 0, 1}.
//    // float PixelDistance(int dx, int dy);

//    // In addition spatial distance needs to supply:
//    void SetPixelDescriptor(float* descriptor)
//    // and
//    static constexpr int descriptor_size();
//    // where descriptor_size() returns the corresponding templated size from
//    // PixelDescriptor.
//
// d) Trait classes: Describe how descriptors are merged and how their distances
//    are computed. Note: Descriptor distance do not affect the order in which regions
//    are merged (this is solely taken care of by above PixelDistanceFunction), but
//    they determine if two regions considered for merging (via low edge cost) really
//    will be merged based on descriptor similarity. Similarily, regions with equal
//    constraints might be split up if their DescriptorDistance is sufficiently
//    dis-similar.
//    The interface is as follows:
//    (for details see segmentation_graph.h)
//
//    // Note: The region_descriptor_size is different from the pixel_descriptor_size.
//    // During construction of the graph a temporary PixelDescriptor is formed
//    // based on the passed templated version.
//    // This temporary PixelDescriptor is then converted to a RegionDescriptor via
//    // below's InitializeDescriptor function.
//    DescriptorTraits<region_descriptor_size, pixel_descriptor_size>  (derived classes)
//
//
//
//    // Returns thresholds for merge and split decisions (in normalized domain [0, 1]).
//    float MergeDistanceThreshold()
//    float SplitDistanceThreshold()
//
//    // Distance between two descriptors (lhs, rhs). In addition current edge distance
//    // between the two descriptors is passed. Should return distance in [0, 1].
//    float DescriptorDistance(const float* lhs,
//                             const float* rhs,
//                             float edge_distance);
//
//    // Merge two descriptors during region merge.
//    void MergeDescriptor(int lhs_size,             // size of source lhs region
//                         const float* lhs,         // lhs descriptor
//                         int rhs_size,             // size of source/dest. rhs region
//                         float* rhs_dst) const;    // rhs descriptor (destination)
//
//    // Copies pixel_descriptor to region_descriptor and performs additional
//    // initializations.
//    void InitializeDescriptor(const float* pixel_descriptor,
//                              float* region_descriptor) const;
//
//    // Called if for two regions descriptor distance is above MergeDistanceThreshold.
//    // In that case return value of this function is used to determine if the merge test
//    // is considered to be failed. Once a region's merge test has failed, it is not
//    // considered for merges again!
//    bool FlagMergeTestFailed(const float* lhs,
//                             bool lhs_above_min_region_sz,
//                             const float* rhs,
//                             bool rhs_above_min_region_sz) const;


////// PixelDistanceFunction
////////////////////////////

// Each distance must be derived from PixelDistanceFunction and templated with the
// expected DataType of the underlying cv::Mat and the stride between neighboring pixels
// (in sizeof(DataType) pixels).
template <class DataType, int channels>
struct PixelDistanceFunction {
  typedef DataType data_type;
  static constexpr int stride () { return channels; };
};

struct ColorDiff3L1 : public PixelDistanceFunction<float, 3> {
  inline float operator()(const float* ptr_1, const float* ptr_2) {
    const float diff_1 = ptr_1[0] - ptr_2[0];
    const float diff_2 = ptr_1[1] - ptr_2[1];
    const float diff_3 = ptr_1[2] - ptr_2[2];
    return (fabs(diff_1) + fabs(diff_2) + fabs(diff_3)) * (1.0f / 3.0f);
  }
};

struct ColorDiff3L2 : public PixelDistanceFunction<float, 3> {
  inline float operator()(const float* ptr_1, const float* ptr_2) {
    const float diff_1 = ptr_1[0] - ptr_2[0];
    const float diff_2 = ptr_1[1] - ptr_2[1];
    const float diff_3 = ptr_1[2] - ptr_2[2];
    return sqrt((diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3) * (1.0f / 3.0f));
  }
};

// TODO(grundman): Usage example / motivation.
struct GradientDiffL2 : public PixelDistanceFunction<float, 2> {
  inline float operator()(const float* ptr_1, const float* ptr_2) {
    const float diff_1 = ptr_1[0] - ptr_2[0];
    const float diff_2 = ptr_1[1] - ptr_2[1];
    return std::min<float>(1.0f,
                           10.0f * sqrt((diff_1 * diff_1 + diff_2 * diff_2) * (0.5f)));
  }
};

struct GradientDiffL1 : public PixelDistanceFunction<float, 2> {
  inline float operator()(const float* ptr_1, const float* ptr_2) {
    const float diff_1 = ptr_1[0] - ptr_2[0];
    const float diff_2 = ptr_1[1] - ptr_2[1];
    return (fabs(diff_1) + fabs(diff_2)) * 0.5f;
  }
};


////// PixelDescriptor
//////////////////////

// Each pixel descriptor must be derived from PixelDescriptorBase and implement
template<class DataType, int kDescriptorSize, int kInputSize = kDescriptorSize>
struct PixelDescriptorBase {
  typedef DataType data_type;
  static constexpr int descriptor_size() { return kDescriptorSize; }
  static constexpr int input_size() { return kInputSize; }
};

// Pixel descriptors for SpatialCvMatDistance.
struct ColorPixelDescriptor : public PixelDescriptorBase<float, 3> {
  static void SetDescriptor(const float* img_ptr, float *descriptor) {
    descriptor[0] = img_ptr[0];
    descriptor[1] = img_ptr[1];
    descriptor[2] = img_ptr[2];
  }
};

struct GradientPixelDescriptor : public PixelDescriptorBase<float, 2> {
  static void SetDescriptor(const float* img_ptr, float *descriptor) {
    descriptor[0] = img_ptr[0];
    descriptor[1] = img_ptr[1];
  }
};


////// Spatial/Temporal cv::Mat distances
/////////////////////////////////////////


// NOTE: You need to ensure that each distance object is copyable
//       or implement a valid copy constructor.
// Every distance must be inherited from AbstractPixelDistance.
class AbstractPixelDistance {
 public:
  virtual ~AbstractPixelDistance() = 0;
};

// Can be parametrized by generic DistanceFunction's and 
// PixelDesciptors, see below for examples.
template <class DistanceFunction, class PixelDescriptor>
class SpatialDistance : public AbstractPixelDistance {
  static_assert(std::is_same<typename DistanceFunction::data_type,
                             typename PixelDescriptor::data_type>::value,
                "DistanceFunction and PixelDescriptor data_type mismatch");
 public:
  typedef typename DistanceFunction::data_type data_type;
  static constexpr int stride() { return DistanceFunction::stride(); };
  static constexpr int descriptor_size() { return PixelDescriptor::descriptor_size(); };
  typedef PixelDescriptor pixel_descriptor;
};


template <class DistanceFunction>
class TemporalDistance : public AbstractPixelDistance {
public:
  typedef typename DistanceFunction::data_type data_type;
  static constexpr int stride() { return DistanceFunction::stride(); };
};

template <class DistanceFunction, class PixelDescriptor>
class SpatialCvMatDistance : public SpatialDistance<DistanceFunction, PixelDescriptor> {
  using SpatialDistance<DistanceFunction, PixelDescriptor>::stride;
  using typename SpatialDistance<DistanceFunction, PixelDescriptor>::data_type;
 public:
  SpatialCvMatDistance(const cv::Mat& image,
                       const DistanceFunction& distance = DistanceFunction())
      : image_(image), distance_(distance) {
    CHECK_EQ(stride(), image.channels());
  }

  inline void MoveAnchorTo(int x, int y) {
    anchor_ptr_ = image_.ptr<data_type>(y) + stride() * x;
  }

  inline void IncrementAnchor() {
    anchor_ptr_ += stride();
  }

  inline void SetPixelDescriptor(float* descriptor) {
    PixelDescriptor::SetDescriptor(anchor_ptr_, descriptor);
  }

  inline void MoveTestAnchorTo(int x, int y) {
    test_ptr_[0] = image_.ptr<data_type>(y) + stride() * x;
    test_ptr_[1] = image_.ptr<data_type>(y + 1) + stride() * x;
  }

  inline void IncrementTestAnchor() {
    test_ptr_[0] += stride();
    test_ptr_[1] += stride();
  }

  inline float PixelDistance(int dx, int dy) {
    return distance_(anchor_ptr_, test_ptr_[dy] + stride() * dx);
  }

  // Debugging function returning underlying cv::Mat
  const cv::Mat& Image() const { return image_; }

 private:
  const cv::Mat image_;

  const data_type* anchor_ptr_;
  const data_type* test_ptr_[2];

  DistanceFunction distance_;

  static_assert(DistanceFunction::stride() == PixelDescriptor::input_size(),
                "DistanceFunction and PixelDescriptor number of channels mismatch");
};

// Can be parametrized by generic DistanceFunction's and 
// PixelDesciptors, see below for examples.
template <class DistanceFunction, class PixelDescriptor>
class SpatialCvMatDistanceGeneric :
  public SpatialDistance<DistanceFunction, PixelDescriptor> {
  using SpatialDistance<DistanceFunction, PixelDescriptor>::stride;
  using typename SpatialDistance<DistanceFunction, PixelDescriptor>::data_type;
  using typename SpatialDistance<DistanceFunction, PixelDescriptor>::descriptor_size;
 public:
  SpatialCvMatDistanceGeneric(const cv::Mat& image,
                              const cv::Mat& descriptor_image,
                              const DistanceFunction& distance = DistanceFunction())
      : image_(image), descriptor_image_(descriptor_image), distance_(distance) { }

  inline void MoveAnchorTo(int x, int y) {
    anchor_ptr_ = image_.ptr<data_type>(y) + stride() * x;
    descriptor_ptr_ = descriptor_image_.ptr<data_type>(y) + descriptor_size() * x;
  }

  inline void IncrementAnchor() {
    anchor_ptr_ += stride();
    descriptor_ptr_ += descriptor_size();
  }

  inline void SetPixelDescriptor(float* descriptor) {
    PixelDescriptor::SetDescriptor(descriptor_ptr_, descriptor);
  }

  inline void MoveTestAnchorTo(int x, int y) {
    test_ptr_[0] = image_.ptr<data_type>(y) + stride() * x;
    test_ptr_[1] = image_.ptr<data_type>(y + 1) + stride() * x;
  }

  inline void IncrementTestAnchor() {
    test_ptr_[0] += stride();
    test_ptr_[1] += stride();
  }

  inline float PixelDistance(int dx, int dy) {
    return distance_(anchor_ptr_, test_ptr_[dy] + stride() * dx);
  }

  // Debugging function returning underlying cv::Mat
  const cv::Mat& Image() const { return image_; }

 private:
  const cv::Mat image_;
  const cv::Mat descriptor_image_;

  const data_type* anchor_ptr_;
  const data_type* descriptor_ptr_;
  const data_type* test_ptr_[2];

  DistanceFunction distance_;
};

template <class DistanceFunction>
class TemporalCvMatDistance : public TemporalDistance<DistanceFunction> {
  using TemporalDistance<DistanceFunction>::stride;
  using typename TemporalDistance<DistanceFunction>::data_type;
public:
  TemporalCvMatDistance(const cv::Mat& curr_image,
                        const cv::Mat& prev_image,
                        const DistanceFunction& distance = DistanceFunction())
  : curr_image_(curr_image), prev_image_(prev_image), distance_(distance) {
    CHECK_EQ(stride(), curr_image.channels());   // Unnamed enum -> int conversion.
    CHECK_EQ(stride(), prev_image.channels());
    CHECK_EQ(curr_image.size(), prev_image.size());
   }

  inline void MoveAnchorTo(int x, int y) {
    anchor_ptr_ = curr_image_.ptr<data_type>(y) + stride() * x;
  }

  inline void IncrementAnchor() {
    anchor_ptr_ += stride();
  }

  inline void MoveTestAnchorTo(int x, int y) {
    test_ptr_[0] = prev_image_.ptr<data_type>(y - 1) + stride() * x;
    test_ptr_[1] = prev_image_.ptr<data_type>(y) + stride() * x;
    test_ptr_[2] = prev_image_.ptr<data_type>(y + 1) + stride() * x;
  }

  inline void IncrementTestAnchor() {
    test_ptr_[0] += stride();
    test_ptr_[1] += stride();
    test_ptr_[2] += stride();
  }

  inline float PixelDistance(int dx, int dy) {
    return distance_(anchor_ptr_, test_ptr_[dy + 1] + stride() * dx);
  }

  const cv::Mat& CurrImage() const { return curr_image_; }
  const cv::Mat& PrevImage() const { return prev_image_; }

private:
  const cv::Mat curr_image_;
  const cv::Mat prev_image_;

  const data_type* anchor_ptr_;
  const data_type* test_ptr_[3];

  DistanceFunction distance_;
};

// Returns constant value for each edge.
class ConstantPixelDistance : public AbstractPixelDistance {
 public:
  static constexpr int descriptor_size() { return 0; }

  ConstantPixelDistance(float value) : value_(value) {}

  inline void MoveAnchorTo(int, int) {
  }

  inline void IncrementAnchor() {
  }

  inline void MoveTestAnchorTo(int, int) {
  }

  inline void IncrementTestAnchor() {
  }

  inline float PixelDistance(int, int) {
    return value_;
  }

  inline void SetPixelDescriptor(float* descriptor) {
  }

 private:
   const float value_;
};

/// Concrete instantiations for ColorDiffs and PixelDescriptors
///////////////////////////////////////////////////////////////

typedef SpatialCvMatDistance<ColorDiff3L1, ColorPixelDescriptor>
SpatialCvMatDistance3L1;

typedef SpatialCvMatDistance<ColorDiff3L2, ColorPixelDescriptor>
SpatialCvMatDistance3L2;

typedef SpatialCvMatDistanceGeneric<ColorDiff3L1, ColorPixelDescriptor>
SpatialCvMatDistanceGeneric3L1;

typedef SpatialCvMatDistanceGeneric<ColorDiff3L2, ColorPixelDescriptor>
SpatialCvMatDistanceGeneric3L2;

typedef SpatialCvMatDistance<GradientDiffL2, GradientPixelDescriptor>
    SpatialGradientDistanceL2;

typedef SpatialCvMatDistance<GradientDiffL1, GradientPixelDescriptor>
    SpatialGradientDistanceL1;

typedef TemporalCvMatDistance<ColorDiff3L1> TemporalCvMatDistance3L1;
typedef TemporalCvMatDistance<ColorDiff3L2> TemporalCvMatDistance3L2;


// Segmentation Trait classes.
///////////////////////////////

template <int kRegionDescriptorSize, int kPixelDescriptorSize>
class DescriptorTraitsBase {
 public:
  static constexpr int region_descriptor_size() {
    return kRegionDescriptorSize;
  }

  static constexpr int pixel_descriptor_size() {
    return kPixelDescriptorSize;
  }
};

class ColorMeanDescriptorTraits : public DescriptorTraitsBase<3, 3> {
 public:
  float MergeDistanceThreshold() const { return 0.05f; }
  float SplitDistanceThreshold() const { return 0.15f; }

  ColorMeanDescriptorTraits(float force_merge_weight = 0.0f)
      : force_merge_weight_(force_merge_weight) {
  }

  inline float DescriptorDistance(const float* lhs,
                                  const float* rhs,
                                  float edge_distance) const {
    const float diff_1 = lhs[0] - rhs[0];
    const float diff_2 = lhs[1] - rhs[1];
    const float diff_3 = lhs[2] - rhs[2];
    const float dist =
        sqrt((diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3) * (1.0f / 3.0f));

    if (edge_distance < force_merge_weight_ && dist < 0.2) {
      return 0.0f;
    } else {
      return dist;
    }
  }

  inline void MergeDescriptor(int lhs_size,
                              const float* lhs,
                              int rhs_size,
                              float* rhs_dst) const {
    const float denom = 1.0f / (lhs_size + rhs_size);
    const float a = lhs_size * denom;
    const float b = rhs_size * denom;
    rhs_dst[0] = a * lhs[0] + b * rhs_dst[0];
    rhs_dst[1] = a * lhs[1] + b * rhs_dst[1];
    rhs_dst[2] = a * lhs[2] + b * rhs_dst[2];
  }

  inline void InitializeDescriptor(const float* pixel_descriptor,
                                   float* region_descriptor) const {
    region_descriptor[0] = pixel_descriptor[0];
    region_descriptor[1] = pixel_descriptor[1];
    region_descriptor[2] = pixel_descriptor[2];
  }

  inline bool FlagMergeTestFailed(const float* lhs,
                                  bool lhs_above_min_region_sz,
                                  const float* rhs,
                                  bool rhs_above_min_region_sz) const {
    return lhs_above_min_region_sz || rhs_above_min_region_sz;
  }
 private:
  float force_merge_weight_;
};

class GradientMeanDescriptorTraits : public DescriptorTraitsBase<2, 2> {
 public:
  float MergeDistanceThreshold() const { return 0.1f; }
  float SplitDistanceThreshold() const { return 0.15f; }

  inline float DescriptorDistance(const float* lhs,
                                  const float* rhs,
                                  float edge_distance) const {
    const float diff_1 = lhs[0] - rhs[0];
    const float diff_2 = lhs[1] - rhs[1];
    return std::min<float>(1.0f,
                           10.0f * sqrt((diff_1 * diff_1 + diff_2 * diff_2) * 0.5f));
  }

  inline void MergeDescriptor(int lhs_size,
                              const float* lhs,
                              int rhs_size,
                              float* rhs_dst) const {
    const float denom = 1.0f / (lhs_size + rhs_size);
    const float a = lhs_size * denom;
    const float b = rhs_size * denom;
    rhs_dst[0] = a * lhs[0] + b * rhs_dst[0];
    rhs_dst[1] = a * lhs[1] + b * rhs_dst[1];
  }

  inline void InitializeDescriptor(const float* pixel_descriptor,
                                   float* region_descriptor) const {
    const float sign = pixel_descriptor[0] < 0 ? -1.0f : 1.0f;
    region_descriptor[0] = pixel_descriptor[0] * sign;
    region_descriptor[1] = pixel_descriptor[1] * sign;
  }

  inline bool FlagMergeTestFailed(const float* lhs,
                                  bool lhs_above_min_region_sz,
                                  const float* rhs,
                                  bool rhs_above_min_region_sz) const {
    return lhs_above_min_region_sz || rhs_above_min_region_sz;
  }
};


// NOTE: Variance of small homogenous tubes is close to zero, so this serves more as an
// example than a usuable implementation.
class ColorMeanVarianceDescriptorTraits : public DescriptorTraitsBase<6, 3> {
 public:
  float MergeDistanceThreshold() const { return 0.1f; }
  float SplitDistanceThreshold() const { return 0.75f; }

  inline float DescriptorDistance(const float* lhs, const float* rhs) const {
    // Bhattacharya distance.
    const float mean_var[3] = { std::max(1e-4f, (0.5f * (lhs[3] + rhs[3]))),
                                std::max(1e-4f, (0.5f * (lhs[4] + rhs[4]))),
                                std::max(1e-4f, (0.5f * (lhs[5] + rhs[5]))) };

    const float diff[3] = { lhs[0] - rhs[0],
                            lhs[1] - rhs[1],
                            lhs[2] - rhs[2] };

    // 3rd z-score covers 99.8% of all distances -> 3 * 3 = 9 max for each term.
    const float dist = sqrt(diff[0] * diff[0] / mean_var[0] +
                            diff[1] * diff[1] / mean_var[1] +
                            diff[2] * diff[2] / mean_var[2]) * 0.2;

    return std::min(1.0f, dist);
  }

  inline void MergeDescriptor(int lhs_size,
                              const float* lhs,
                              int rhs_size,
                              float* rhs_dst) const {
    const float denom = 1.0f / (lhs_size + rhs_size);
    const float a = lhs_size * denom;
    const float b = rhs_size * denom;
    const float lhs_sq_mean[3] = { lhs[0] * lhs[0],
                                   lhs[1] * lhs[1],
                                   lhs[2] * lhs[2] };

    const float rhs_sq_mean[3] = { rhs_dst[0] * rhs_dst[0],
                                   rhs_dst[1] * rhs_dst[1],
                                   rhs_dst[2] * rhs_dst[2] };

    // Update mean of merged region.
    rhs_dst[0] = a * lhs[0] + b * rhs_dst[0];
    rhs_dst[1] = a * lhs[1] + b * rhs_dst[1];
    rhs_dst[2] = a * lhs[2] + b * rhs_dst[2];

    // Update variance of merged region.
    // Variance merge rule:
    // n_1 * (v_1 + mean_1^2) + n_2 * (v_2 + mean_2^2) / (n_1 + n_2) - new_mean^2
    rhs_dst[3] = a * (lhs[3] + lhs_sq_mean[0]) + b * (rhs_dst[3] + rhs_sq_mean[0])
        - rhs_dst[0] * rhs_dst[0];
    rhs_dst[4] = a * (lhs[4] + lhs_sq_mean[1]) + b * (rhs_dst[4] + rhs_sq_mean[1])
        - rhs_dst[1] * rhs_dst[1];
    rhs_dst[5] = a * (lhs[5] + lhs_sq_mean[2]) + b * (rhs_dst[5] + rhs_sq_mean[2])
        - rhs_dst[2] * rhs_dst[2];
  }

  inline void InitializeDescriptor(const float* pixel_descriptor,
                                   float* region_descriptor) const {
    region_descriptor[0] = pixel_descriptor[0];
    region_descriptor[1] = pixel_descriptor[1];
    region_descriptor[2] = pixel_descriptor[2];
    // Initialize with zero variance.
    // TODO(grundman): Shouldn't that be infinity?
    region_descriptor[3] = 0.0f;
    region_descriptor[4] = 0.0f;
    region_descriptor[5] = 0.0f;
  }

  inline bool FlagMergeTestFailed(const float* lhs,
                                  bool lhs_above_min_region_sz,
                                  const float* rhs,
                                  bool rhs_above_min_region_sz) const {
     return lhs_above_min_region_sz || rhs_above_min_region_sz;
   }
};


/// Aggregators to combine several distances, descriptor and traits.
/////////////////////////////////////////////////////////////////////

// Pixel distance aggregators.
//////////////////////////////
template<class Distance1, class Distance2, class DistanceAggregator>
class AggregatedDistance : public AbstractPixelDistance {
 public:
  AggregatedDistance(const Distance1& dist_1,
                     const Distance2& dist_2,
                     const DistanceAggregator& aggregator)
    : dist_1_(dist_1), dist_2_(dist_2), aggregator_(aggregator) {
  }

  virtual ~AggregatedDistance() { }

  inline void MoveAnchorTo(int x, int y) {
    dist_1_.MoveAnchorTo(x, y);
    dist_2_.MoveAnchorTo(x, y);
  }

  inline void IncrementAnchor() {
    dist_1_.IncrementAnchor();
    dist_2_.IncrementAnchor();
  }

  inline void MoveTestAnchorTo(int x, int y) {
    dist_1_.MoveTestAnchorTo(x ,y);
    dist_2_.MoveTestAnchorTo(x, y);
  }

  inline void IncrementTestAnchor() {
    dist_1_.IncrementTestAnchor();
    dist_2_.IncrementTestAnchor();
  }

  inline float PixelDistance(int dx, int dy) {
    return aggregator_(dist_1_.PixelDistance(dx, dy),
                       dist_2_.PixelDistance(dx, dy));
  }

 protected:
  Distance1 dist_1_;
  Distance2 dist_2_;
  DistanceAggregator aggregator_;
};

// Same as above, but use for spatial distances with descriptor support.
template<class Distance1, class Distance2, class DistanceAggregator>
class AggregatedSpatialDistance :
    public AggregatedDistance<Distance1, Distance2, DistanceAggregator> {
 static constexpr int kDescriptorSize1 = Distance1::descriptor_size();
 static constexpr int kDescriptorSize2 = Distance2::descriptor_size();
 public:
  AggregatedSpatialDistance(const Distance1& dist_1,
                            const Distance2& dist_2,
                            const DistanceAggregator& aggregator)
    : AggregatedDistance<Distance1, Distance2, DistanceAggregator>(
        dist_1, dist_2, aggregator) {
  }

  static constexpr int descriptor_size() {
    return kDescriptorSize1 + kDescriptorSize2;
  }

  inline void SetPixelDescriptor(float* descriptor) {
    this->dist_1_.SetPixelDescriptor(descriptor);
    this->dist_2_.SetPixelDescriptor(descriptor + kDescriptorSize1);
  }
};

// Linear weighting of distances.
// Note: Convex combination, i.e. weights should sum to one.
class LinearDistanceAggregator2 {
 public:
  LinearDistanceAggregator2(float weight_1, float weight_2)
      : weight_1_(weight_1), weight_2_(weight_2) {
  }
  float operator()(float dist_1, float dist_2) const {
    return weight_1_ * dist_1 + weight_2_ * dist_2;
  }
 private:
  float weight_1_;
  float weight_2_;
};

// Simply multiplies distances.
class IndependentDistanceAggregator2 {
 public:
  IndependentDistanceAggregator2() { }
  float operator()(float dist_1, float dist_2) const {
    return 1.0f - (1.0f - dist_1) * (1.0f - dist_2);
  }
};

// Evaluates to sqrt(dist_1 * dist_1 + dist_2 * dist_2) / sqrt(2.0)
class SqrtAggregator {
 public:
  SqrtAggregator() { }
  float operator()(float dist_1, float dist_2) const {
    return sqrt(dist_1 * dist_1 + dist_2 * dist_2) * .70711f;   // 1.f / sqrt(2.0)
  }
};


///// Trait aggregators
////////////////////////////

template <class Trait1, class Trait2, class DistanceAggregator>
class AggregatedDescriptorTraits :
  public DescriptorTraitsBase<Trait1::region_descriptor_size() +
                              Trait2::region_descriptor_size(),
                              Trait1::pixel_descriptor_size() +
                              Trait2::pixel_descriptor_size()> {
 public:
  AggregatedDescriptorTraits(const Trait1& trait_1,
                             const Trait2& trait_2,
                             const DistanceAggregator& aggregator)
      : trait_1_(trait_1), trait_2_(trait_2), aggregator_(aggregator) {
  }

  float MergeDistanceThreshold() const {
    return aggregator_.MergeDistanceThreshold(trait_1_.MergeDistanceThreshold(),
                                              trait_2_.MergeDistanceThreshold());
  }

  float SplitDistanceThreshold() const {
    return aggregator_.SplitDistanceThreshold(trait_1_.SplitDistanceThreshold(),
                                              trait_2_.SplitDistanceThreshold());
  }

  inline float DescriptorDistance(const float* lhs,
                                  const float* rhs,
                                  float edge_distance) const {
    return aggregator_.DescriptorDistance(
        trait_1_.DescriptorDistance(lhs, rhs, edge_distance),
        trait_2_.DescriptorDistance(lhs + kDescriptorOffset,
                                    rhs + kDescriptorOffset,
                                    edge_distance));
  }

  inline void MergeDescriptor(int lhs_size,
                              const float* lhs,
                              int rhs_size,
                              float* rhs_dst) const {
    trait_1_.MergeDescriptor(lhs_size, lhs, rhs_size, rhs_dst);
    trait_2_.MergeDescriptor(lhs_size,
                             lhs + kDescriptorOffset,
                             rhs_size,
                             rhs_dst + kDescriptorOffset);
  }

  inline void InitializeDescriptor(const float* pixel_descriptor,
                                   float* region_descriptor) const {
    trait_1_.InitializeDescriptor(pixel_descriptor, region_descriptor);
    trait_2_.InitializeDescriptor(pixel_descriptor + kInputOffset,
                                  region_descriptor + kDescriptorOffset);
  }

  inline bool FlagMergeTestFailed(const float* lhs,
                                  bool lhs_above_min_region_sz,
                                  const float* rhs,
                                  bool rhs_above_min_region_sz) const {
    return trait_1_.FlagMergeTestFailed(lhs,
                                        lhs_above_min_region_sz,
                                        rhs,
                                        rhs_above_min_region_sz) ||
           trait_2_.FlagMergeTestFailed(lhs,
                                        lhs_above_min_region_sz,
                                        rhs,
                                        rhs_above_min_region_sz);
  }

 private:
  Trait1 trait_1_;
  Trait2 trait_2_;
  DistanceAggregator aggregator_;

  static constexpr int kDescriptorOffset = Trait1::region_descriptor_size();
  static constexpr int kInputOffset = Trait1::pixel_descriptor_size();
};


}  // namespace segmentation.

#endif  // VIDEO_SEGMENT_SEGMENTATION_PIXEL_DISTANCE_H__
