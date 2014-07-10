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

#ifndef VIDEO_SEGMENT_SEGMENTATION_REGION_SEGMENTATION_H__
#define VIDEO_SEGMENT_SEGMENTATION_REGION_SEGMENTATION_H__

#include "base/base.h"

#include <opencv2/core/core.hpp>

#include "segmentation/segmentation.h"

namespace segmentation {

// Options for region segmentation.
struct RegionSegmentationOptions {
  // Minimum number of regions. Hierarchical segmentation stops when number of regions
  // fall below specified threshold.
  int min_region_num = 10;

  // Maximum number of regions. It is guaranteed that the first level of
  // the hierarchical segmentation has at most max_region_num regions.
  int max_region_num = 10000;

  // Number of regions is reduced by approximately
  // 1.0f - level_cutoff_fraction at each level.
  float level_cutoff_fraction = 0.8;

  // Edge weights for small regions are multiplied by:
  // 1.0 + small_region_penalizer * log(region_sz / average_region_size),
  // larger penalizer will lead to fewer small regions.
  float small_region_penalizer = 0.25;

  // Number of histogram bins per channel used for lab color matching.
  int luminance_bins = 10;
  int color_bins = 20;
  int flow_bins = 16;

  // Size of chunk_sets and chunk_overlap (specified in number of chunks,
  // from over-segmentation).
  int chunk_set_size = 6;
  int chunk_set_overlap = 2;
  int constraint_chunks = 1;

  // If set RegionDescriptors will be added to the protobuffer.
  bool save_descriptors = false;

  // Region descriptors / extractors in order. Number of activated items has to agree
  // with number of items passed to ProcessFrame.
  // At least appearance or flow need to be requested.
  bool use_appearance = true;
  bool use_flow = true;
  bool use_size_penalizer = true;

  // If set computes vectorization for each unit.
  bool compute_vectorization = true;
};

class Segmentation;

// Default region segmentation implementation using appearance and flow.
// // Usage example:
//
// RegionSegmentation region_segmentation(RegionSegmentationOptions(), 640, 360);
// int num_frames =  // Initialize with number of frames.
// for (int k = 0; k < num_frames; ++k) {
//   cv::Mat image_frame(360, 640, CV_8UC3);  // Initialized from somewhere.
//   SegmentationDesc over_segmentation;     // Initialized from DenseSegmentation result.
//   std::vector<cv::Mat> features = {image_frame};
//   bool is_last_frame = (k + 1 == num_frames);
//   std::vector<std::unique_ptr<SegmentationDesc>> results;
//   if (region_segmentation.ProcessFrame(is_last_frame,
//                                       &over_segmentation,
//                                       &features,
//                                       &results)) {
//     // Process results.
//   }
// }
//
///////////////////////
//
// // Usage example for a single frame with over and region segmentation:
//
// DenseSegmentation dense_segmentation(DenseSegmentationOptions(), 640, 360);
// RegionSegmentation region_segmentation(RegionSegmentationOptions(), 640, 360);
// cv::Mat image_frame(360, 640, CV_8UC3);   // initialized from somewhere.
// 
// std::vector<cv::Mat> features = {image_frame};
// std::vector<std::unique_ptr<SegmentationDesc>> overseg_results;
// dense_segmentation.ProcessFrame(true, &features, nullptr, &overseg_results);
// CHECK_EQ(1, overseg_results.size());
//
// const SegmentationDesc& overseg_result = *overseg_results[0];
// std::vector<std::unique_ptr<SegmentationDesc>> region_results;
// region_segmentation.ProcessFrame(true, 
//                                  &overseg_result,
//                                  &features,
//                                  &region_results);
// CHECK_EQ(1, region_results.size());
// const SegmentationDesc& region_result = *region_results[0];
//
class RegionSegmentation {
 public:
  RegionSegmentation(const RegionSegmentationOptions& options,
                     int frame_width,
                     int frame_height);

  virtual ~RegionSegmentation() = default;


  // Process the next frame (pass next frame's segmentation and features)
  // and outputs results (if available) in results. Returns number of segmentations
  // in results. In default implementation only feature is CV_8UC3 image frame.
  // If only results are requested pass nullptr to features and segmentation.
  // Optionally pass dense flow (as 2 channel (x,y) float image to connect voxels
  // along flow.
  int ProcessFrame(bool flush,
                   const SegmentationDesc* segmentation,
                   const std::vector<cv::Mat>* features,
                   std::vector<std::unique_ptr<SegmentationDesc>>* results);

 protected:
  // Override the following functions to customize region segmentation's
  // extractors, descriptors and distances.

  // Called for each incoming feature. Expected to create a list of desciptor extractors
  // and updaters.
  // Overload for custom feature pre-processing.
  virtual void GetDescriptorExtractorAndUpdaters(
      const std::vector<cv::Mat>& features,
      DescriptorExtractorList* extractors,
      DescriptorUpdaterList* updaters);

  virtual int NumDescriptors() { return num_descriptors_; }

  // Returns region distance to be used for hierarchical segmentation.
  virtual std::unique_ptr<RegionDistance> GetRegionDistance();

 protected:
  // Creates SegmentationOptions based from RegionSegmentationOptions.
  void GetSegmentationOptions(SegmentationOptions* options) const;

  const RegionSegmentationOptions& options() const { return options_; }
  int frame_width() const { return frame_width_; }
  int frame_height() const { return frame_height_; }

 private:
  // Processes chunk boundary (segments and outputs results). If flush is false,
  // sets up new constrained Segmentation object.
  void ChunkBoundaryOutput(bool flush,
                           std::vector<std::unique_ptr<SegmentationDesc>>* results);

  // Called by above function for actual segmentation and output of results.
  // Specifies beginning of overlap and lookahead in frames.
  void SegmentAndOutputChunk(int overlap_start,
                             int lookahead_start,
                             std::vector<std::unique_ptr<SegmentationDesc>>* results);

  RegionSegmentationOptions options_;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int num_descriptors_ = 0;

  // Current processed chunk (as measured in processed chunk_sets).
  int chunk_sets_ = 0;

  // Input over-segmentation chunks.
  int read_chunks_ = 0;

  // Designates first frame in overlap chunkset. -1 for not set.
  int overlap_start_ = -1;

  // Designates first frame in lookahead (not being output). -1 for not set.
  int lookahead_start_ = -1;

  // Number of output frames.
  int num_output_frames_ = 0;

  // Used to chain appearance extractors for window support.
  std::shared_ptr<AppearanceExtractor> prev_appearance_extractor_;

  // Underlying segmentation object.
  std::unique_ptr<Segmentation> seg_;

  // Created in overlap. Next segmentation object.
  std::unique_ptr<Segmentation> new_seg_;

  std::vector<int> max_region_ids_;
};

}  // namespace segmentation.

#endif  // VIDEO_SEGMENT_SEGMENTATION_REGION_SEGMENTATION_H__
