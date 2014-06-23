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

#ifndef REGION_DESCRIPTOR_H__
#define REGION_DESCRIPTOR_H__

#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "base/base.h"
#include "histograms.h"
#include "segment_util/segmentation.pb.h"

#define SPARSE_HISTS 1

namespace cv {
  class Mat;
}

namespace segmentation {

// Generic framework to define region descriptors / features for hierarchical
// segmentation stage.
// Per region descriptor several classes are involved:
// 1.) A descriptor extractor derived from RegionDescriptorExtractor. The descriptors
//     extractor's task is to pre-compute any information necessary for extracting
//     descriptors during AddOverSegmentation.
//     The actual form of this extraction is completely up to the user.
// 2.) The extractor defines a abstract interface to create corresponding
//     RegionDescriptor's. During graph creation each RegionDescriptor is passed
//     the oversegmented regions rasterization and the extractor which is used to add
//     to build up the descriptor.
//     Further, each descriptor must support hierarchical merging functions to allow
//     the creation of descriptors of super-regions.
//     After each hierarchical stage concludes descriptors can be updated viaL
// 3.) An descriptor updater. The updater is initialized by passing all created regions
//     after a stage completes. Then each descriptor is updated by passing the updater.
// 4.) A region distance. For each descriptor type the distances are evaluated and
//     a collected in a vector. The distance describes how to merge/convert this
//     vector into one distance.
// 5.) An optional protobuffer that extends RegionFeatures in segmentation.proto.
//     Each RegionDescriptor is called to add itself via an extension to RegionFeatures
//     which is used save descriptors to file, e.g. to perform learning later on.

class RegionDescriptorExtractor;
class RegionDescriptorUpdater;
class SegmentationDesc_Rasterization;
class RegionInformation;
typedef std::vector<std::unique_ptr<RegionInformation>> RegionInfoList;
class Segmentation;

// Abstract base class for a descriptor. Implement for each descriptor.
class RegionDescriptor : public TypedType {
 public:
  RegionDescriptor(const std::type_info* type) : TypedType(type), parent_region_(0) { }
  RegionDescriptor(const RegionDescriptor&) = default;

  virtual ~RegionDescriptor() = 0;

  // Called exactly once per frame.
  virtual void AddFeatures(const SegmentationDesc_Rasterization& raster,
                           const RegionDescriptorExtractor& extractor,
                           int frame_number) = 0;

  // Called by framework to determine similarity between two descriptors.
  virtual float RegionDistance(const RegionDescriptor& rhs) const = 0;

  // Descriptors are spatio-temporal and multiple frames are used to compile
  // a desciptor. This function is called when all frames have been added.
  // This function can be called multiple times!
  virtual void PopulatingDescriptorFinished() {}

  // During hierarchical segmentation RegionDescriptors are merged and cloned.
  // It is guaranteed that both functions will only be called after
  // PopulatingDescriptorFinished() has been issued.
  virtual void MergeWithDescriptor(const RegionDescriptor& rhs) = 0;

  // Clones a descriptor using CloneImpl();
  RegionDescriptor* Clone() const;

  // Descriptors can optionally have a corresponding feature defined as extension to
  // RegionFeatures in the segmentation.proto. This function is called on output if
  // requested by user to save descriptors to file.
  virtual void AddToRegionFeatures(RegionFeatures* features) const { }

  // Called before each hierarchical segmentation iteration. Can be used to
  // update/initialize descriptor. By default no-op.
  virtual void UpdateDescriptor(const RegionDescriptorUpdater& updater) { }

 protected:
  // Parent region for the descriptor. Set by the Segmentation object.
  const RegionInformation* ParentRegion() const { return parent_region_; }

 private:
  // Actual clone method to be implemented by derived classes.
  virtual RegionDescriptor* CloneImpl() const = 0;

  // Called by Segmentation object.
  void SetParent(const RegionInformation* parent) { parent_region_ = parent; }
  friend class RegionInformation;

  // Points to the actual region this descriptor is attached to.
  const RegionInformation* parent_region_;
};

// Generic base class to start the extraction process.
class RegionDescriptorExtractor : public TypedType {
 public:
  RegionDescriptorExtractor(const std::type_info* type) : TypedType(type) { }
  RegionDescriptorExtractor(const RegionDescriptorExtractor&) = default;
  virtual ~RegionDescriptorExtractor() = 0;

  // Reimplement to return corresponding descriptor.
  virtual std::unique_ptr<RegionDescriptor> CreateDescriptor() const = 0;
};

// We have mutliple descriptors per region.
typedef std::vector<std::shared_ptr<RegionDescriptorExtractor>> DescriptorExtractorList;

// Updater for region descriptors.
class RegionDescriptorUpdater : public TypedType {
 public:
  RegionDescriptorUpdater(const std::type_info* type) : TypedType(type) { }
  virtual ~RegionDescriptorUpdater() {}

  // Called after every hierarchical stage completes with segmentation result.
  virtual void InitializeUpdate(const RegionInfoList& region_list) = 0;

 private:
  // Specifies actual RegionDescriptorUpdater type.
  const std::type_info* type_;
};

// We maintain one updater per descriptor type.
typedef std::vector<std::shared_ptr<RegionDescriptorUpdater>> DescriptorUpdaterList;

// Use for non-mutable RegionDescriptors.
class NonMutableUpdater : public RegionDescriptorUpdater {
 public:
   NonMutableUpdater() : RegionDescriptorUpdater(&typeid(*this)) { }
  // No-op.
  virtual void InitializeUpdate(const RegionInfoList& region_list) override { }
};

// Convenience function.
// Returns a list non mutable updaters, effectively ignoring any update calls.
DescriptorUpdaterList NonMutableUpdaters(int num_updaters);

// Generic region distance.
// As argument a list of per-region descriptor distances is passed.
// Region distance is expected to derive a normalized distance in [0, 1] from it.
class RegionDistance {
 public:
  virtual float Evaluate(const std::vector<float>& descriptor_distances) const = 0;

  // Feature Dimension / Number of descriptors this distance is defined for.
  virtual int NumDescriptors() const = 0;

  virtual ~RegionDistance() {}
};



///////// Specific distance implementations
////////////////////////////////////////////

// Zero if all distances are close to zero, one if at least one descriptor distance
// is one (even if remaining ones are zero).
template<int dimension>
class SquaredORDistance : public RegionDistance {
 public:
  virtual float Evaluate(const std::vector<float>& descriptor_distances) const override {
    DCHECK_LE(dimension, descriptor_distances.size());
    float result = 1.0f;
    for (int i = 0; i < dimension; ++i) {
      result *= (1.0f - descriptor_distances[i]);
    }
    result = 1.0f - result;
    return result * result;
  }

  virtual int NumDescriptors() const { return dimension; }
  virtual ~SquaredORDistance() {}
};

// Same as above but expects additional SmallRegionPenalizer as last descriptor.
// Penalizes small regions by scaling computed region distance by
// RegionSizePenalizer::RegionDistance.
template<int dimension>
class SquaredORDistanceSizePenalized : public RegionDistance {
 public:
  virtual float Evaluate(const std::vector<float>& descriptor_distances) const {
    DCHECK_EQ(dimension + 1, descriptor_distances.size());
    const float base_distance = or_distance_.Evaluate(descriptor_distances);
    const float size_scale = descriptor_distances.back();
    return std::max(0.f, std::min(1.f, base_distance * size_scale));
  }

  virtual int NumDescriptors() const { return dimension + 1; }
  virtual ~SquaredORDistanceSizePenalized() { }

 private:
  SquaredORDistance<dimension> or_distance_;
};


// Concrete implementation for a variety of different RegionDescriptor's / Extractors.
//////////////////////////////////////////////////////////////////////////////////////
class AppearanceDescriptor3D : public RegionDescriptor {
 public:
  AppearanceDescriptor3D(int luminance_bins, int color_bins);
  virtual void AddFeatures(const SegmentationDesc_Rasterization& raster,
                           const RegionDescriptorExtractor& extractor,
                           int frame_number) override;

  virtual float RegionDistance(const RegionDescriptor& rhs) const override;
  virtual void PopulatingDescriptorFinished() override;
  virtual void MergeWithDescriptor(const RegionDescriptor& rhs) override;
  virtual RegionDescriptor* CloneImpl() const override;
  virtual void AddToRegionFeatures(RegionFeatures* features) const override;

 private:
  AppearanceDescriptor3D(const AppearanceDescriptor3D& rhs)
    : RegionDescriptor(rhs),
      is_populated_(rhs.is_populated_),
      color_histogram_(new ColorHistogram(*rhs.color_histogram_)) {
  }

  std::unique_ptr<ColorHistogram> color_histogram_;
  bool is_populated_ = false;
};

// Gain adaptive descriptor. To be finalized.
// Creates descriptor that compares regions within windows of specified window_size.
// Windows are gain and bias normalized.
class WindowedAppearanceDescriptor : public RegionDescriptor {
 public:
  WindowedAppearanceDescriptor(int window_size, int luminance_bins, int color_bins);
  virtual void AddFeatures(const SegmentationDesc_Rasterization& raster,
                           const RegionDescriptorExtractor& extractor,
                           int frame_number) override;

  virtual float RegionDistance(const RegionDescriptor& rhs) const override;
  virtual void PopulatingDescriptorFinished() override;
  virtual void MergeWithDescriptor(const RegionDescriptor& rhs) override;
  virtual RegionDescriptor* CloneImpl() const override;
  virtual void AddToRegionFeatures(RegionFeatures* features) const override;

 private:
  WindowedAppearanceDescriptor(const WindowedAppearanceDescriptor& rhs);

  // Returns 3 element gain change of frame w.r.t. anchor.
 static cv::Point3f GetGainChange(const cv::Point3f& anchor_average,
                                  const cv::Point3f& frame_average);

 // Returns true if gain_change is for any channel above threshold or below
 // 1.0 / threshold.
 static bool GainChangeAboveThreshold(const cv::Point3f& gain_change,
                                      float threshold);   // e.g. 1.1

  // Histogram and averages (in Lab color space) of anchor frame (first frame of window
  // boundary). Histograms are always calibrated w.r.t. first frame of a window and
  // therefore can be directly compared.
  struct CalibratedHistogram {
    CalibratedHistogram(const int lum_bins,
                        const int col_bins,
                        const cv::Point3f& mean_values_)
        : color_hist(lum_bins, col_bins, SPARSE_HISTS),
          mean_values(mean_values_) {
    }
    CalibratedHistogram(const CalibratedHistogram&) = default;

    ColorHistogram color_hist;
    cv::Point3f mean_values;
    bool is_populated = false;
  };

  int start_window() const { return start_window_; }
  int end_window() const { return start_window_ + windows_.size(); }

  // One histogram per window.
  typedef std::vector<std::unique_ptr<CalibratedHistogram>> WindowHistograms;
  WindowHistograms windows_;

  int window_size_;
  int start_window_ = -1;  // Start window index of first window in windows_;
  int compare_radius_ = 1;
  int luminance_bins_;
  int color_bins_;
};

// Used to build color histograms from LAB frames.
// Optionally uses windowed_descriptor if window_size > 0 (in this case pass previous
// extractor to the constructor).
class AppearanceExtractor : public RegionDescriptorExtractor {
 public:
  AppearanceExtractor(int luminance_bins,
                      int color_bins,
                      int window_size,
                      const cv::Mat& rgb_frame,
                      const AppearanceExtractor* prev_extractor);  // Optional.

  std::unique_ptr<RegionDescriptor> CreateDescriptor() const {
    if (window_size_ == 0) {
      return std::unique_ptr<RegionDescriptor>(new AppearanceDescriptor3D(
            luminance_bins_, color_bins_));
    } else {
      return std::unique_ptr<RegionDescriptor>(new WindowedAppearanceDescriptor(
            window_size_, luminance_bins_, color_bins_));
    }
  }

  const cv::Mat& LabFrame() const { return *lab_frame_; }

  // Returns average values for specified relative index, i.e. for frame:
  // frame_number - relative_index.
  const cv::Point3f& AverageValues(int relative_index) const {
    DCHECK_LT(relative_index, mean_values_.size());
    return mean_values_[relative_index];
  }

 private:
  int luminance_bins_;
  int color_bins_;

  std::unique_ptr<cv::Mat> lab_frame_;

  int window_size_ = 0;

  // Stores last window_size mean_values. Most recent mean values are at the front.
  std::vector<cv::Point3f> mean_values_;
};

// Feature-less descriptor. Updates on UpdateDescriptor.
class RegionSizePenalizer : public RegionDescriptor {
 public:
  RegionSizePenalizer(float penalizer)
      : RegionDescriptor(&typeid(*this)), penalizer_(penalizer) { }

  RegionSizePenalizer(float penalizer, float inv_av_region_size)
      : RegionSizePenalizer(penalizer) {
    inv_av_region_size_ = inv_av_region_size;
  }

  virtual void AddFeatures(const SegmentationDesc_Rasterization& raster,
                           const RegionDescriptorExtractor& extractor,
                           int frame_number) override { }

  virtual float RegionDistance(const RegionDescriptor& rhs) const override;
  virtual void PopulatingDescriptorFinished() override { }
  virtual void MergeWithDescriptor(const RegionDescriptor& rhs) override { }
  virtual RegionDescriptor* CloneImpl() const override {
    return new RegionSizePenalizer(*this);
  }

  virtual void AddToRegionFeatures(RegionFeatures* features) const override { }
  virtual void UpdateDescriptor(const RegionDescriptorUpdater& updater) override;

 private:
  RegionSizePenalizer(const RegionSizePenalizer&) = default;

  float penalizer_;
  float inv_av_region_size_ = 1.0f;   // Holds 1.0 / average region size
};

// Dummy extractor, simply returns RegionSizePenalizer.
class RegionSizePenalizerExtractor : public RegionDescriptorExtractor {
 public:
  RegionSizePenalizerExtractor(float penalizer)
      : RegionDescriptorExtractor(&typeid(*this)), penalizer_(penalizer) { }

  std::unique_ptr<RegionDescriptor> CreateDescriptor() const {
    return std::unique_ptr<RegionDescriptor>(new RegionSizePenalizer(penalizer_));
  }

 private:
  float penalizer_;
};

class RegionSizePenalizerUpdater : public RegionDescriptorUpdater {
 public:
  // Default to non-zero weight.
  RegionSizePenalizerUpdater() : RegionDescriptorUpdater(&typeid(*this)) { }

  // Determines average region size.
  virtual void InitializeUpdate(const RegionInfoList& region_list);
  float InvAverageRegionSize() const { return inv_av_region_size_; }
 private:
  float inv_av_region_size_ = 1.0f;
};


class FlowExtractor;
class FlowDescriptor : public RegionDescriptor {
 public:
  FlowDescriptor(int flow_bins);

  virtual void AddFeatures(const SegmentationDesc_Rasterization& raster,
                           const RegionDescriptorExtractor& extractor,
                           int frame_num) override;

  virtual float RegionDistance(const RegionDescriptor& rhs) const override;
  virtual void PopulatingDescriptorFinished() override;

  virtual void MergeWithDescriptor(const RegionDescriptor& rhs) override;
  virtual RegionDescriptor* CloneImpl() const override;

 private:
  FlowDescriptor(const FlowDescriptor& rhs);

  int start_frame() const { return start_frame_; }
  int end_frame() const { return start_frame_ + flow_histograms_.size(); }

  // A histogram per frame.
  std::vector<std::unique_ptr<VectorHistogram>> flow_histograms_;
  // Start frame for above vector.
  int start_frame_ = -1;

  int flow_bins_;
  bool is_populated_ = false;
};

class FlowExtractor : public RegionDescriptorExtractor {
 public:
  // Use if no flow is present at current frame.
  FlowExtractor(int flow_bins);
  FlowExtractor(int flow_bins, const cv::Mat& flow);

  std::unique_ptr<RegionDescriptor> CreateDescriptor() const {
    return std::unique_ptr<RegionDescriptor>(new FlowDescriptor(flow_bins_));
  }

  const cv::Mat& Flow() const { return flow_; }
  bool HasValidFlow() const { return valid_flow_; }

 private:
  int flow_bins_;
  bool valid_flow_ = false;   // Indicates if flow is present at current frame.
  const cv::Mat flow_;
};

// TODO(grundman): Fix texture and match descriptor.
/*

class MatchDescriptor : public RegionDescriptor {
 public:
  MatchDescriptor();
  void Initialize(int my_id);
  void AddMatch(int match_id, float strength);

  virtual float RegionDistance(const RegionDescriptor& rhs) const;

  virtual void MergeWithDescriptor(const RegionDescriptor& rhs);
  virtual RegionDescriptor* CloneImpl() const;

  virtual void OutputToAggregated(AggregatedDescriptor* agg_desc) const;
 private:
  int my_id_;

  // List of sorted (match_id, strength) tuples.
  struct MatchTuple {
    int match_id;
    float strength;
    bool operator<(const MatchTuple& rhs) const {
      return match_id < rhs.match_id;
    }
  };

  vector<MatchTuple> matches_;
  DECLARE_REGION_DESCRIPTOR(MatchDescriptor, MATCH_DESCRIPTOR);
};

class LBPDescriptor : public RegionDescriptor {
 public:
  LBPDescriptor();
  void Initialize(int frame_width, int frame_height);

  void AddSamples(const SegmentationDesc_Rasterization& raster,
                  const std::vector<std::shared_ptr<cv::Mat> >& lab_inputs);

  virtual float RegionDistance(const RegionDescriptor& rhs) const;

  virtual void PopulatingDescriptorFinished();
  virtual void MergeWithDescriptor(const RegionDescriptor& rhs);
  virtual RegionDescriptor* CloneImpl() const;

  virtual void OutputToAggregated(AggregatedDescriptor* agg_desc) const;
private:
  void AddLBP(const uchar* lab_ptr, int sample_id, int width_step);
  static int MapLBP(int lbp) {
    if (lbp_lut_.size() == 0) ComputeLBP_LUT(8); return lbp_lut_[lbp]; }

  void static ComputeLBP_LUT(int bits);
 private:
  int frame_width_;
  int frame_height_;

  // Maps 8 bit lbp to rotation invariant numbers from 0 .. 9.
  static std::vector<int> lbp_lut_;

  std::vector<std::shared_ptr<VectorHistogram> > lbp_histograms_;
  std::vector<std::shared_ptr<VectorHistogram> > var_histograms_;
};
   */

}  // namespace segmentation.

#endif  // REGION_DESCRIPTOR_H__

