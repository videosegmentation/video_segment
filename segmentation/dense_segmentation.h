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

#ifndef DENSE_SEGMENTATION_H__
#define DENSE_SEGMENTATION_H__

#include "base/base.h"

#include <opencv2/core/core.hpp>

#include "dense_seg_graph_interface.h"
#include "segmentation.h"

namespace segmentation {

// Options for dense segmentation.
struct DenseSegmentationOptions {
  // Presmoothing for input images.
  enum Presmoothing {
    PRESMOOTH_NONE = 0,
		PRESMOOTH_GAUSSIAN = 1,
		PRESMOOTH_BILATERAL = 2,
	};

  Presmoothing presmoothing = PRESMOOTH_BILATERAL;

  // Minimum region size specified w.r.t. frame dimensions.
  // Min size is computed as
  // frac_min_region_size * width *
  // frac_min_region_size * height *
  // chunk_size.
  float frac_min_region_size = 0.01;

  // Size of a chunk in frames. Over-segmentation will be performed in chunks of
  // chunk_size is enabled by processing_mode.
  // The previous chunk constrains the next chunk by using an overlap of
  // chunk_overlap_ratio and associates regions across the boundaries.
  // Needs to be at least 3 frames.
  int chunk_size = 20;

  // Overlap between chunks. Defaults to 4 frames. At least 2 frames are always used.
  float chunk_overlap_ratio = 0.2;

  // Segments first spatial edges, then temporal ones. If deactivated, all edges
  // will be used.
  bool two_stage_oversegment = false;

  // Number of frames used as constraints within the overlap.
  // Truncated to number of overlap frames, if neccessary. Needs to be at least 1.
  int num_constraint_frames = 1;

  // TODO(grundman): Does not work correctly yet.
  bool thin_structure_suppression = false;

  // Enforces voxel to be spatially (!) connected via N4 neighborhoods.
  bool enforce_n4_connectivity = true;

  // If set, enforces that spatio-temporal regions are always spatially connected.
  bool enforce_spatial_connectedness = true;

  // Color distance for appearance descriptor.
  enum ColorDistance {
    COLOR_DISTANCE_L1 = 0,
    COLOR_DISTANCE_L2 = 1,
  };

  ColorDistance color_distance = COLOR_DISTANCE_L2;

  bool compute_vectorization = false;
};

class Segmentation;

// Usage example:
// DenseSegmentation dense_segmentation(DenseSegmentationOptions(), 640, 360);
// int num_frames =  // initialize with number of frames.
// for (int k = 0; k < num_frames; ++k) {
//   cv::Mat image_frame(360, 640, CV_8UC3);  // initialized from somewhere.
//   std::vector<cv::Mat> features = {image_frame};
//   bool is_last_frame = (k + 1 == num_frames);
//   std::vector<std::unique_ptr<SegmentationDesc>> results;
//   // For flow pass pointer to CV_32FC2 cv::Mat instead of nullptr.
//   if (dense_segmentation.ProcessFrame(is_last_frame, &features, nullptr, &results)) {
//     // Process results.
//   }
// }
class DenseSegmentation : public DenseSegGraphCreatorInterface {
 public:
  DenseSegmentation(const DenseSegmentationOptions& options,
                    int frame_width,
                    int frame_height);

  virtual ~DenseSegmentation() = default;


  // Process the next frame (pass next frame's features via features)
  // and outputs results (if available) in results. Returns number of segmentations
  // in results. In default implementation only feature is CV_8UC3 image frame.
  // If only results are requested pass nullptr to features.
  // Optionally pass dense flow (as 2 channel (x,y) float image to connect voxels
  // along flow.
  // If flush is set to true, forces output of all buffered input.
  int ProcessFrame(bool flush,
                   const std::vector<cv::Mat>* features,
                   const cv::Mat* flow,  // optional.
                   std::vector<std::unique_ptr<SegmentationDesc>>* results);

  int ChunkSize() const { return options_.chunk_size; }

 protected:
  // Override the following functions to customize dense segmentation's
  // distances and processing details.

  // Called for each incoming feature before buffering. Perform smoothing or
  // whitening operations here. It is expected that this function creates a deep copy
  // or new cv::Mat for each entry in the vector, i.e. the data must be
  // owned by DenseSegmentation.
  // Overload for custom feature pre-processing.
  virtual void PreprocessFeatures(
      const std::vector<cv::Mat>& input_features,
      std::vector<cv::Mat>* output_features);

  // Expected to create a SpatialDistance from curr_features and call
  // AddFrameToSegmentation. Simply pass through constrained_segmentation.
  // Overload for distance customization.
  virtual void AddFrameToSegmentationFromFeatures(
      const std::vector<cv::Mat>& curr_features,
      const SegmentationDesc* constrained_segmentation);

  // Call with result from above function.
  template <class Distance>
  void AddFrameToSegmentation(const Distance& distance,
                              const SegmentationDesc* constained_segmentation);

  // Similar to AddFrameToSegmentationFromFeatures, converts features to TemporalDistance
  // and is expected to call ConnectSegmentationTemporally with the resulting distance.
  // Overload for distance customization.
  virtual void ConnectSegmentationTemporallyFromFeatures(
      const std::vector<cv::Mat>& curr_features,
      const std::vector<cv::Mat>& prev_features,
      const cv::Mat* flow);

  // Similar to above function to introduce temporal connections via
  // seg->ConnectTemporally. If flow is specified (!= nullptr) use
  // seg->ConnectTemporallyAlongFlow instead.
  template <class Distance>
  void ConnectSegmentationTemporally(const Distance& distance,
                                     const cv::Mat* flow);

  // Returns newly created DenseSegmentationGraph templated with appropiate
  // distance and descriptor traits.
  virtual DenseSegGraphInterface* CreateDenseSegGraph(int frame_width,
                                                      int frame_height,
                                                      int chunk_size) const;

  // Creates SegmentationOptions based from DenseSegmentationOptions.
  void GetSegmentationOptions(SegmentationOptions* options) const;

  const DenseSegmentationOptions& options() const { return options_; }
  int frame_width() const { return frame_width_; }
  int frame_height() const { return frame_height_; }

 private:
  // Processes chunk boundary (segments and outputs results). If flush is false,
  // sets up new constrained Segmentation object.
  void ChunkBoundaryOutput(bool flush,
                           std::vector<std::unique_ptr<SegmentationDesc>>* results);

  // Called by above function for actual segmentation and output of results.
  void SegmentAndOutputChunk(bool flush,
                             std::vector<std::unique_ptr<SegmentationDesc>>* results);

  DenseSegmentationOptions options_;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int input_frames_ = 0;

  // Current processed chunk.
  int chunk_id_ = 0;

  // Number of overlap frames.
  int overlap_frames_ = 2;

  // Number of constraints.
  int constraint_frames_ = 1;

  // Maximum region id so far.
  int max_region_id_ = 0;

  // Number of output frames.
  int num_output_frames_ = 0;

  // Buffer for passed images and flow.
  // Due to parallel graph construction we keep a deep copy.
  std::vector<std::vector<cv::Mat>> feature_buffer_;
  std::vector<cv::Mat> flow_buffer_;

  // Start of the current chunk in above buffers.
  int curr_chunk_start_ = 0;

  // Segmentation results in previous overlap.
  std::vector<std::unique_ptr<SegmentationDesc>> overlap_segmentations_;

  // Underlying segmentation object.
  std::unique_ptr<Segmentation> seg_;
};

template <class Distance>
void DenseSegmentation::AddFrameToSegmentation(
    const Distance& distance,
    const SegmentationDesc* constrained_segmentation) {
  if (constrained_segmentation) {
    seg_->AddGenericImageConstrained(*constrained_segmentation, distance);
  } else {
    seg_->AddGenericImage(distance);
  }
}

template <class Distance>
void DenseSegmentation::ConnectSegmentationTemporally(
    const Distance& distance,
    const cv::Mat* flow) {
  if (flow) {
    seg_->ConnectTemporallyAlongFlow(*flow, distance);
  } else {
    seg_->ConnectTemporally(distance);
  }
}

}  // namespace segmentation.


#endif  // DENSE_SEGMENTATION_H__
