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

#include "segmentation/region_descriptor.h"

#include <opencv2/imgproc/imgproc.hpp>

#include "base/base_impl.h"
#include "imagefilter/image_util.h"
#include "segment_util/segmentation_util.h"
#include "segmentation/segmentation_common.h"

namespace segmentation {

RegionDescriptorExtractor::~RegionDescriptorExtractor() { }
RegionDescriptor::~RegionDescriptor() { }

RegionDescriptor* RegionDescriptor::Clone() const {
  RegionDescriptor* cloned = CloneImpl();

  // Copy base class information.
  cloned->parent_region_ = parent_region_;
  return cloned;
}

DescriptorUpdaterList NonMutableUpdaters(int num_updaters) {
  DescriptorUpdaterList updaters;
  for (int n = 0; n < num_updaters; ++n) {
    updaters.push_back(std::shared_ptr<RegionDescriptorUpdater>(new NonMutableUpdater()));
  }
  return updaters;
}

AppearanceExtractor::AppearanceExtractor(
    int luminance_bins,
    int color_bins,
    int window_size,
    const cv::Mat& rgb_frame,
    const AppearanceExtractor* prev_extractor)
    : RegionDescriptorExtractor(&typeid(*this)),
      luminance_bins_(luminance_bins),
      color_bins_(color_bins),
      window_size_(window_size) {
  lab_frame_.reset(new cv::Mat(rgb_frame.rows,
                               rgb_frame.cols,
                               CV_8UC3));

  cv::cvtColor(rgb_frame, *lab_frame_, CV_BGR2Lab);

  if (window_size > 0) {
    if (prev_extractor) {
      mean_values_ = prev_extractor->mean_values_;
    }

    // Compute average per-channel intensity.
    cv::Scalar mean = cv::mean(*lab_frame_);
    mean_values_.insert(mean_values_.begin(), cv::Point3f(mean[0], mean[1], mean[2]));

    // Only store as many values as needed.
    if (mean_values_.size() > window_size) {
      mean_values_.resize(window_size);
    }
  }
}

AppearanceDescriptor3D::AppearanceDescriptor3D(int luminance_bins,
                                               int color_bins)
    : RegionDescriptor(&typeid(*this)) {
  color_histogram_.reset(new ColorHistogram(luminance_bins, color_bins, SPARSE_HISTS));
}

void AppearanceDescriptor3D::AddFeatures(const Rasterization& raster,
                                         const RegionDescriptorExtractor& extractor,
                                         int frame_number) {
  const AppearanceExtractor& appearance_extractor = extractor.As<AppearanceExtractor>();
  for (const auto& scan_inter : raster.scan_inter()) {
    const uint8_t* interval_ptr = appearance_extractor.LabFrame().ptr<uint8_t>(
        scan_inter.y()) +  3 * scan_inter.left_x();

    for (int x = scan_inter.left_x();
         x <= scan_inter.right_x();
         ++x, interval_ptr += 3) {
      color_histogram_->AddPixelInterpolated(interval_ptr);
    }
  }
}

float AppearanceDescriptor3D::RegionDistance(const RegionDescriptor& rhs_uncast) const {
  const auto& rhs = rhs_uncast.As<AppearanceDescriptor3D>();
  return color_histogram_->ChiSquareDist(*rhs.color_histogram_);
}

void AppearanceDescriptor3D::PopulatingDescriptorFinished() {
  if (!is_populated_) {
#ifndef SPARSE_HISTS
    color_histogram_->ConvertToSparse();
#endif
    color_histogram_->NormalizeToOne();
    is_populated_ = true;
  }
}

void AppearanceDescriptor3D::MergeWithDescriptor(const RegionDescriptor& rhs_uncast) {
  const auto& rhs = rhs_uncast.As<AppearanceDescriptor3D>();
  color_histogram_->MergeWithHistogram(*rhs.color_histogram_);
}

RegionDescriptor* AppearanceDescriptor3D::CloneImpl() const {
  return new AppearanceDescriptor3D(*this);
}

void AppearanceDescriptor3D::AddToRegionFeatures(RegionFeatures* descriptor) const {
}

WindowedAppearanceDescriptor::WindowedAppearanceDescriptor(int window_size,
                                                           int luminance_bins,
                                                           int color_bins)
    : RegionDescriptor(&typeid(*this)),
      window_size_(window_size),
      luminance_bins_(luminance_bins),
      color_bins_(color_bins) {
}

void WindowedAppearanceDescriptor::AddFeatures(const Rasterization& raster,
                                               const RegionDescriptorExtractor& extractor,
                                               int frame_number) {
  const auto& appearance_extractor = extractor.As<AppearanceExtractor>();
  const int global_window_idx = frame_number / window_size_;
  const int window_frame = frame_number % window_size_;

  if (start_window_ < 0) {
    start_window_ = global_window_idx;
  }

  const int window_idx = global_window_idx - start_window_;

  // Compact all previous windows and add new window.
  if (window_idx >= windows_.size()) {
    for (auto& window_ptr : windows_) {
      if (window_ptr == nullptr) {
         continue;
      }
      if (!window_ptr->is_populated) {
#if SPARSE_HISTS
        window_ptr->color_hist.ConvertToSparse();
#endif
        window_ptr->color_hist.NormalizeToOne();
        window_ptr->is_populated = true;
      }
    }

    windows_.resize(window_idx + 1);
  }

  if (windows_[window_idx] == nullptr) {
    windows_[window_idx].reset(
        new CalibratedHistogram(luminance_bins_,
                                color_bins_,
                                appearance_extractor.AverageValues(window_frame)));
  }

  ColorHistogram& histogram = windows_[window_idx]->color_hist;
  cv::Point3f gain_change = GetGainChange(windows_[window_idx]->mean_values,
                                          appearance_extractor.AverageValues(0));

  for (const auto& scan_inter : raster.scan_inter()) {
    const uint8_t* interval_ptr =
        appearance_extractor.LabFrame().ptr<uint8_t>(scan_inter.y()) +
        3 * scan_inter.left_x();

    for (int x = scan_inter.left_x();
         x <= scan_inter.right_x();
         ++x, interval_ptr += 3) {
      histogram.AddPixelValuesInterpolated(
          std::min(255.f, (float)interval_ptr[0] * gain_change.x),
          std::min(255.f, (float)interval_ptr[1] * gain_change.y),
          std::min(255.f, (float)interval_ptr[2] * gain_change.z));
    }
  }
}

float WindowedAppearanceDescriptor::RegionDistance(
    const RegionDescriptor& rhs_uncast) const {
  const WindowedAppearanceDescriptor& rhs = rhs_uncast.As<WindowedAppearanceDescriptor>();

  // Determine corresponding search interval.
  const int search_start = start_window();
  const int search_end = end_window();

  // Holds weighted average of per window distances (weighted by minimum histogram
  // weight, i.e. number of samples).
  double dist_sum = 0;
  double weight_sum = 0;

  for (int window_idx = search_start; window_idx < search_end; ++window_idx) {
    const int lhs_idx = window_idx - start_window_;
    // We allow for empty windows.
    if (windows_[lhs_idx] == nullptr) {
      LOG(WARNING) << "Empty window found, this can happen for small window sizes.";
      continue;
    }

    const ColorHistogram& curr_hist = windows_[lhs_idx]->color_hist;

    // Traverse comparison radius.
    double match_weight_sum = 0;
    double match_sum = 0;
    for (int match_window = window_idx - compare_radius_;
         match_window <= window_idx + compare_radius_;
         ++match_window) {
      const int rhs_idx = match_window - rhs.start_window_;
      if (rhs_idx < 0 || rhs_idx >= rhs.windows_.size()) {
        continue;
      }

      if (rhs.windows_[rhs_idx] == nullptr) {
        LOG(WARNING) << "Empty window found, this can happen for small window sizes.";
        continue;
      }

      ColorHistogram* match_hist = &rhs.windows_[rhs_idx]->color_hist;
      std::unique_ptr<ColorHistogram> gain_corrected_hist;
      if (match_window != window_idx) {
        // Build windowed gain corrected histogram, if needed.
        cv::Point3f gain = GetGainChange(windows_[lhs_idx]->mean_values,
                                         rhs.windows_[rhs_idx]->mean_values);
        if (GainChangeAboveThreshold(gain, 1.1f)) {
          gain_corrected_hist.reset(new ColorHistogram(
                rhs.windows_[rhs_idx]->color_hist.ScaleHistogram({gain.x, gain.y,
                                                                  gain.z})));
          match_hist = gain_corrected_hist.get();
        }
      }

      const float weight = std::min(curr_hist.WeightSum(), match_hist->WeightSum());
      const float dist = curr_hist.ChiSquareDist(*match_hist);
      DCHECK(dist >= 0.0f && dist < 1.0f + 1.0e-3f) << "Chi-Sq: " << dist;

      match_weight_sum += weight;
      match_sum += weight * dist;
    }

    dist_sum += match_sum;
    weight_sum += match_weight_sum;
  }

  if (weight_sum > 0) {
    const float result = dist_sum / weight_sum;
    DCHECK(result >= 0 && result < 1.0 + 1e-3) << "Distance: " << result;
    return result;
  } else {
    return 0.f;
  }
}

void WindowedAppearanceDescriptor::PopulatingDescriptorFinished() {
  for (auto& window : windows_) {
    if (window != nullptr && !window->is_populated) {
#if SPARSE_HISTS
      window->color_hist.ConvertToSparse();
#endif
      window->color_hist.NormalizeToOne();
      window->is_populated = true;
    }
  }
}

void WindowedAppearanceDescriptor::MergeWithDescriptor(
    const RegionDescriptor& rhs_uncast) {
  const WindowedAppearanceDescriptor& rhs = rhs_uncast.As<WindowedAppearanceDescriptor>();

  const int max_idx = std::max(end_window(), rhs.end_window());

  while (start_window() > rhs.start_window()) {
    windows_.emplace(windows_.begin(), nullptr);
    --start_window_;
  }

  while (max_idx > end_window()) {
    windows_.emplace_back(nullptr);
  }

  // Note: Some histograms might not be allocated yet!
  for (int k = start_window_; k < max_idx; ++k) {
    const int lhs_idx = k - start_window();
    const int rhs_idx = k - rhs.start_window();

    if (windows_[lhs_idx] == nullptr) {
      DCHECK(rhs.windows_[rhs_idx] != nullptr);
      windows_[lhs_idx].reset(new CalibratedHistogram(*rhs.windows_[rhs_idx]));
    } else {
      if (rhs_idx < 0 || rhs_idx >= rhs.windows_.size()) {
        continue;
      }
      DCHECK(rhs.windows_[rhs_idx] != nullptr);
      // Both histograms are calibrated w.r.t. to the same number offset.
      DCHECK(windows_[lhs_idx]->mean_values == rhs.windows_[rhs_idx]->mean_values);
      windows_[lhs_idx]->color_hist.MergeWithHistogram(
            rhs.windows_[rhs_idx]->color_hist);
    }
  }
}

RegionDescriptor* WindowedAppearanceDescriptor::CloneImpl() const {
  return new WindowedAppearanceDescriptor(*this);
}

void WindowedAppearanceDescriptor::AddToRegionFeatures(RegionFeatures* descriptor) const {
}

WindowedAppearanceDescriptor::WindowedAppearanceDescriptor(
    const WindowedAppearanceDescriptor& rhs)
    : RegionDescriptor(&typeid(*this)),
      window_size_(rhs.window_size_),
      start_window_(rhs.start_window_),
      compare_radius_(rhs.compare_radius_),
      luminance_bins_(rhs.luminance_bins_),
      color_bins_(rhs.color_bins_) {
  windows_.reserve(rhs.windows_.size());
  for (const auto& window_ptr : rhs.windows_) {
    if (window_ptr != nullptr) {
      windows_.emplace_back(new CalibratedHistogram(*window_ptr));
    } else {
      windows_.emplace_back(nullptr);
    }
  }
}

cv::Point3f WindowedAppearanceDescriptor::GetGainChange(
    const cv::Point3f& anchor_mean,
    const cv::Point3f& frame_mean) {
  cv::Point3f gain;
  gain.x = anchor_mean.x / (frame_mean.x + 1.0e-3f);
  gain.y = anchor_mean.y / (frame_mean.y + 1.0e-3f);
  gain.z = anchor_mean.z / (frame_mean.z + 1.0e-3f);
  return gain;
}


bool WindowedAppearanceDescriptor::GainChangeAboveThreshold(
    const cv::Point3f& gain_change,
    float threshold) {
  const float inv_threshold = 1.0 / threshold;
  return !(gain_change.x <= threshold &&
           gain_change.y <= threshold &&
           gain_change.z <= threshold &&
           gain_change.x >= inv_threshold &&
           gain_change.y >= inv_threshold &&
           gain_change.z >= inv_threshold);
}

float RegionSizePenalizer::RegionDistance(const RegionDescriptor& rhs_uncast) const {
  const RegionSizePenalizer& rhs = dynamic_cast<const RegionSizePenalizer&>(rhs_uncast);
  const int min_sz = std::min(ParentRegion()->size, rhs.ParentRegion()->size);
  // Determine scale based on relation to average region size.
  const float size_scale = 1.0f + penalizer_ * log(min_sz * inv_av_region_size_) / log(2);
  return std::min(1.0f, size_scale);
}

void RegionSizePenalizer::UpdateDescriptor(const RegionDescriptorUpdater& base_updater) {
  const RegionSizePenalizerUpdater& updater =
    base_updater.As<RegionSizePenalizerUpdater>();
  inv_av_region_size_ = updater.InvAverageRegionSize();
}

void RegionSizePenalizerUpdater::InitializeUpdate(const RegionInfoList& region_list) {
  if (region_list.empty()) {
    return;
  }

  vector<int> region_size;
  region_size.reserve(region_list.size());
  for (const auto& region_ptr : region_list) {
    region_size.push_back(region_ptr->size);
  }

  if (region_list.empty()) {
    inv_av_region_size_ = 1.0f;
    return;
  }

  auto median = region_size.begin() + region_size.size() / 2;
  std::nth_element(region_size.begin(), median, region_size.end());

  if (*median > 0) {
    inv_av_region_size_ = 1.0f / *median;
  } else {
    inv_av_region_size_ = 1.f;
  }
}

FlowExtractor::FlowExtractor(int flow_bins)
    : RegionDescriptorExtractor(&typeid(*this)), flow_bins_(flow_bins) {
}

FlowExtractor::FlowExtractor(int flow_bins,
                             const cv::Mat& flow)
    : RegionDescriptorExtractor(&typeid(*this)),
      flow_bins_(flow_bins),
      valid_flow_(true),
      flow_(flow) {
   DCHECK(!flow.empty());
}

FlowDescriptor::FlowDescriptor(int flow_bins)
    : RegionDescriptor(&typeid(*this)), flow_bins_(flow_bins) {
}

void FlowDescriptor::AddFeatures(const Rasterization& raster,
                                 const RegionDescriptorExtractor& extractor,
                                 int frame_num) {
  const FlowExtractor& flow_extractor = extractor.As<FlowExtractor>();
  if (!flow_extractor.HasValidFlow()) {   // No flow present at current frame.
    return;
  }

  if (start_frame_ < 0) {
    // First frame.
    start_frame_ = frame_num;
  }

  const int frame_idx = frame_num - start_frame();
  while (frame_idx >= flow_histograms_.size()) {
    flow_histograms_.emplace_back(nullptr);
  }

  if (flow_histograms_[frame_idx] == nullptr) {
    flow_histograms_[frame_idx].reset(new VectorHistogram(flow_bins_));
  }

  for (const auto& scan_inter : raster.scan_inter()) {
    const float* flow_ptr = flow_extractor.Flow().ptr<float>(scan_inter.y())
      + scan_inter.left_x() * 2;
    for (int x = scan_inter.left_x(); x <= scan_inter.right_x(); ++x, flow_ptr += 2) {
      flow_histograms_[frame_idx]->AddVector(flow_ptr[0], flow_ptr[1]);
    }
  }
}

float FlowDescriptor::RegionDistance(const RegionDescriptor& rhs_uncast) const {
  const FlowDescriptor& rhs = rhs_uncast.As<FlowDescriptor>();

  // Limit to valid intersection.
  const int start_idx = std::max(start_frame(), rhs.start_frame());
  const int end_idx = std::min(end_frame(), rhs.end_frame());

  double sum = 0;
  double sum_weight = 0;
  for (int i = start_idx; i < end_idx; ++i) {
    // Both histograms are valid within the interval.
    const int lhs_idx = i - start_frame();
    const int rhs_idx = i - rhs.start_frame();
    DCHECK_GE(lhs_idx, 0);
    DCHECK_GE(rhs_idx, 0);
    if (flow_histograms_[lhs_idx] == nullptr ||
        rhs.flow_histograms_[rhs_idx] == nullptr) {
      LOG(WARNING) << "Empty flow histogram found.";
      continue;
    }

    float weight = std::min(flow_histograms_[lhs_idx]->NumVectors(),
                            rhs.flow_histograms_[rhs_idx]->NumVectors());

    sum +=
      flow_histograms_[lhs_idx]->ChiSquareDist(*rhs.flow_histograms_[rhs_idx]) * weight;
    sum_weight += weight;
  }

  if (sum_weight > 0) {
    return sum / sum_weight;
  } else {
    return 0;
  }
}

void FlowDescriptor::PopulatingDescriptorFinished() {
  if (!is_populated_) {
    for(auto& hist_ptr : flow_histograms_) {
      if (hist_ptr != nullptr) {
        hist_ptr->NormalizeToOne();
      }
    }

    is_populated_ = true;
  }
}

void FlowDescriptor::MergeWithDescriptor(const RegionDescriptor& rhs_uncast) {
  const FlowDescriptor& rhs = rhs_uncast.As<FlowDescriptor>();
  const int max_frame = std::max(end_frame(), rhs.end_frame());

  while (start_frame() > rhs.start_frame()) {
    flow_histograms_.emplace(flow_histograms_.begin(), nullptr);
    --start_frame_;
  }

  while (rhs.end_frame() > end_frame()) {
    flow_histograms_.emplace_back(nullptr);
  }

  // Note: Some histograms might not be allocated yet.
  for (int k = start_frame(); k < end_frame(); ++k) {
    const int lhs_idx = k - start_frame();
    const int rhs_idx = k - rhs.start_frame();
      
    // If rhs histogram does not exist there is nothing to merge.
    if (rhs_idx < 0 || rhs_idx >= rhs.flow_histograms_.size() ||
        rhs.flow_histograms_[rhs_idx] == nullptr) {
      continue;
    }

    if (flow_histograms_[lhs_idx] == nullptr) {
      flow_histograms_[lhs_idx].reset(
          new VectorHistogram(*rhs.flow_histograms_[rhs_idx]));
    } else {
      flow_histograms_[lhs_idx]->MergeWithHistogram(*rhs.flow_histograms_[rhs_idx]);
    }
  }

  // Discard empty start and tail.
  while (flow_histograms_.size() > 0 && flow_histograms_[0] == nullptr) {
    flow_histograms_.erase(flow_histograms_.begin());
    ++start_frame_;
  }

  while (flow_histograms_.size() > 0 && flow_histograms_.back() == nullptr) {
    flow_histograms_.pop_back();
  }
}

RegionDescriptor* FlowDescriptor::CloneImpl() const {
  return new FlowDescriptor(*this);
}

FlowDescriptor::FlowDescriptor(const FlowDescriptor& rhs)
    : RegionDescriptor(&typeid(*this)),
      start_frame_(rhs.start_frame_),
      flow_bins_(rhs.flow_bins_),
      is_populated_(rhs.is_populated_) {
  flow_histograms_.reserve(flow_histograms_.size());
  for (const auto& hist_ptr : rhs.flow_histograms_) {
    if (hist_ptr != nullptr) {
      flow_histograms_.emplace_back(new VectorHistogram(*hist_ptr));
    } else {
      flow_histograms_.emplace_back(nullptr);
    }
  }
}

/*

MatchDescriptor::MatchDescriptor() : RegionDescriptor(MATCH_DESCRIPTOR) {

}

void MatchDescriptor::Initialize(int my_id) {
  my_id_ = my_id;
}

void MatchDescriptor::AddMatch(int match_id, float strength) {
  CHECK(strength >= 0 && strength <= 1);
  // Convert strength to distance.
  strength = 1.0 - strength;
  strength = std::max(strength, 0.1f);

  MatchTuple new_match = { match_id, strength };

  vector<MatchTuple>::iterator insert_pos = std::lower_bound(matches_.begin(),
                                                             matches_.end(),
                                                             new_match);
  if(insert_pos != matches_.end() &&
     insert_pos->match_id == match_id) {
    CHECK_EQ(insert_pos->strength, strength)
      << "Match already present in descriptor with different strength.";
  } else {
    matches_.insert(insert_pos, new_match);
  }
}

float MatchDescriptor::RegionDistance(const RegionDescriptor& rhs_uncast) const {
  const MatchDescriptor& rhs = dynamic_cast<const MatchDescriptor&>(rhs_uncast);

  MatchTuple lhs_lookup = { my_id_, 0 };
  vector<MatchTuple>::const_iterator lhs_match_iter =
      std::lower_bound(rhs.matches_.begin(), rhs.matches_.end(), lhs_lookup);

  MatchTuple rhs_lookup = { rhs.my_id_, 0 };
  vector<MatchTuple>::const_iterator rhs_match_iter =
      std::lower_bound(matches_.begin(), matches_.end(), rhs_lookup);

  float strength = 1;
  if (lhs_match_iter != rhs.matches_.end() &&
      lhs_match_iter->match_id == my_id_) {
    strength = lhs_match_iter->strength;
  }

  if (rhs_match_iter != matches_.end() &&
      rhs_match_iter->match_id == rhs.my_id_) {
    if (strength != 1) {
      strength = (strength + rhs_match_iter->strength) * 0.5;
      // LOG(WARNING) << "One sided match found!";
    } else {
      strength = rhs_match_iter->strength;
    }
  } else {
    if (strength != 1) {
    //  LOG(WARNING) << "One sided match found!";
    }
  }

  return strength;
}

void MatchDescriptor::MergeWithDescriptor(const RegionDescriptor& rhs_uncast) {
  const MatchDescriptor& rhs = dynamic_cast<const MatchDescriptor&>(rhs_uncast);

  if (my_id_ != rhs.my_id_) {
    // TODO: Think about this, no real merge! Winner takes it all.
    if (rhs.matches_.size() > matches_.size()) {
      if (matches_.size() > 0) {
        LOG(WARNING) << "Winner takes it all strategy applied.";
      }

      my_id_ = rhs.my_id_;
      matches_ = rhs.matches_;
    }
  }
}

RegionDescriptor* MatchDescriptor::CloneImpl() const {
  return new MatchDescriptor(*this);
}

void MatchDescriptor::OutputToAggregated(AggregatedDescriptor* descriptor) const {
  SegmentationDesc::MatchDescriptor* match =
    descriptor->mutable_match();

  for (vector<MatchTuple>::const_iterator tuple = matches_.begin();
       tuple != matches_.end();
       ++tuple) {
    SegmentationDesc::MatchDescriptor::MatchTuple* add_tuple = match->add_tuple();
    add_tuple->set_match_id(tuple->match_id);
    add_tuple->set_strength(tuple->strength);
  }
}

LBPDescriptor::LBPDescriptor() : RegionDescriptor(LBP_DESCRIPTOR) {

}

void LBPDescriptor::Initialize(int frame_width, int frame_height) {
  frame_width_ = frame_width;
  frame_height_ = frame_height;
  lbp_histograms_.resize(3);
  var_histograms_.resize(3);
  for (int i = 0; i < 3; ++i) {
    lbp_histograms_[i].reset(new VectorHistogram(10));
    var_histograms_[i].reset(new VectorHistogram(16));
  }
}

void LBPDescriptor::AddSamples(const RegionScanlineRep& scanline_rep,
                               const vector<shared_ptr<IplImage> >& lab_inputs) {
  int scan_idx = scanline_rep.top_y;

  for (vector<IntervalList>::const_iterator scanline = scanline_rep.scanline.begin();
       scanline != scanline_rep.scanline.end();
       ++scanline, ++scan_idx) {
    for (int f = 0; f < 3; ++f) {
      const int max_radius = 1 << f;

      // Check if in bounds.
      if (scan_idx < max_radius || scan_idx >= frame_height_ - max_radius) {
        continue;
      }

      const uchar* row_ptr = RowPtr<uchar>(lab_inputs[f], scan_idx);
      for (IntervalList::const_iterator interval = scanline->begin();
           interval != scanline->end();
           ++interval) {
        const uchar* interval_ptr = row_ptr + interval->first;
        for (int x = interval->first; x <= interval->second; ++x, ++interval_ptr) {
          if (x < max_radius || x >= frame_width_ - max_radius) {
            continue;
          }
        }
        AddLBP(interval_ptr, f, lab_inputs[f]->widthStep);
      }
    }
  }
}

void LBPDescriptor::AddLBP(const uchar* lab_ptr, int sample_id, int width_step) {
  const int threshold = 5;
  const int rad = 1 << sample_id;
  int directions[] = { -rad * width_step - rad, -rad * width_step,
                       -rad * width_step + rad, -rad,
                        rad,                     rad * width_step - rad,
                        rad * width_step,        rad * width_step + rad };

  int center_val = *lab_ptr;
  int lbp = 0;
  float sum = 0;
  float sq_sum = 0;

  for (int i = 0; i < 8; ++i) {
    const int sample = (int)lab_ptr[directions[i]];
    int diff = sample - center_val;
    if (diff > threshold) {
      lbp |= (1 << i);
    }

    sum += sample;
    sq_sum += sample * sample;
  }

  // Add to LBP histogram.
  lbp_histograms_[sample_id]->IncrementBin(MapLBP(lbp));

  sq_sum /= 8.f;
  sum /= 8.f;
  // stdev is in 0 .. 128 ( = 8 * 16), usually not higher than 64 though.
  const float stdev = std::sqrt(sq_sum - sum * sum) / 4.f;
  var_histograms_[sample_id]->IncrementBinInterpolated(std::min(stdev, 15.f));
}

void LBPDescriptor::PopulatingDescriptorFinished() {
  for (vector<shared_ptr<VectorHistogram> >::iterator h = lbp_histograms_.begin();
       h != lbp_histograms_.end();
       ++h) {
    (*h)->NormalizeToOne();
  }

  for (vector<shared_ptr<VectorHistogram> >::iterator h = var_histograms_.begin();
       h != var_histograms_.end();
       ++h) {
    (*h)->NormalizeToOne();
  }
}

void LBPDescriptor::MergeWithDescriptor(const RegionDescriptor& rhs_uncast) {
  const LBPDescriptor& rhs = dynamic_cast<const LBPDescriptor&>(rhs_uncast);

  for (size_t i = 0; i < lbp_histograms_.size(); ++i) {
    lbp_histograms_[i]->MergeWithHistogram(*rhs.lbp_histograms_[i]);
    var_histograms_[i]->MergeWithHistogram(*rhs.var_histograms_[i]);
  }
}

RegionDescriptor* LBPDescriptor::CloneImpl() const {
  LBPDescriptor* new_lbp = new LBPDescriptor();
  new_lbp->Initialize(frame_width_, frame_height_);
  for (int i = 0; i < 3; ++i) {
    new_lbp->lbp_histograms_[i].reset(new VectorHistogram(*lbp_histograms_[i]));
    new_lbp->var_histograms_[i].reset(new VectorHistogram(*var_histograms_[i]));
  }

  return new_lbp;
}

void LBPDescriptor::OutputToAggregated(AggregatedDescriptor* descriptor) const {
  for (int t = 0; t < 3; ++t) {
    SegmentationDesc::TextureDescriptor* texture =
        descriptor->add_texture();

    const float* lbp_values = lbp_histograms_[t]->BinValues();
    for (int i = 0; i < lbp_histograms_[t]->NumBins(); ++i, ++lbp_values) {
      texture->add_lbp_entry(*lbp_values);
    }
  }
}

float LBPDescriptor::RegionDistance(const RegionDescriptor& rhs_uncast) const {
  return 1;

  const LBPDescriptor& rhs = dynamic_cast<const LBPDescriptor&>(rhs_uncast);
  float var_dists[3];
  float lbp_dists[3];
  for (int i = 0; i < 3; ++i) {
    var_dists[i] = var_histograms_[0]->ChiSquareDist(*rhs.var_histograms_[0]);
    lbp_dists[i] = lbp_histograms_[0]->ChiSquareDist(*rhs.lbp_histograms_[0]);
  }

  const float var_dist =
      0.2 * var_dists[0] + 0.3 * var_dists[1] + 0.5 * var_dists[2];
  const float lbp_dist =
      0.2 * lbp_dists[0] + 0.3 * lbp_dists[1] + 0.5 * lbp_dists[2];

  return lbp_dist;
}

vector<int> LBPDescriptor::lbp_lut_;

void LBPDescriptor::ComputeLBP_LUT(int bits) {
  lbp_lut_.resize(1 << bits);
  for (int i = 0; i < (1 << bits); ++i) {
    // Determine number of bit changes and number of 1 bits.
    int ones = 0;
    int changes = 0;
    // Shift with last bit copied to first position and xor.
    const int change_mask = i xor ((i | (i & 1) << bits) >> 1);
    for (int j = 0; j < bits; ++j) {
      changes += (change_mask >> j) & 1;
      ones += (i >> j) & 1;
    }

    if (changes <= 2) {
      lbp_lut_[i] = ones;
    } else {
      lbp_lut_[i] = bits + 1;   // Special marker for non-uniform codes
                                // (more than two sign changes);
    }
  }
}
 */

}  // namespace segmentation.
