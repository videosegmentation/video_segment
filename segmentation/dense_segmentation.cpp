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

#include "dense_segmentation.h"

#include <gflags/gflags.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "base/base_impl.h"
#include "imagefilter/image_filter.h"
#include "dense_segmentation_graph.h"
#include "pixel_distance.h"

DEFINE_string(dense_smoothing, "", "Set to either 'bilateral' to 'gaussian'");
DEFINE_string(dense_color_dist, "", "Set to either 'l1' or 'l2'");
DEFINE_double(dense_min_region_size, 0, "If set > 0.001 specifies fractional region size "
                                        "w.r.t. frame dimensions. Minimum size is "
                                        "computed as min_region_size * width * "
                                        "min_region_size * height * "
                                        "chunk_size");
DEFINE_int32(chunk_size, 0, "If set >= 3, determine size of chunks in pixels.");

namespace segmentation {

DenseSegmentation::DenseSegmentation(const DenseSegmentationOptions& options,
                                     int frame_width,
                                     int frame_height)
    : options_(options), frame_width_(frame_width), frame_height_(frame_height) {
  CHECK_GE(options_.chunk_size, 3) << "Chunk size needs to be at least 3 frames.";
  if (FLAGS_chunk_size >= 3) {
    options_.chunk_size = FLAGS_chunk_size;
  }

  overlap_frames_ = options_.chunk_overlap_ratio * options_.chunk_size + 0.5f;

  // At least 2 frames overlap.
  overlap_frames_ = std::min(overlap_frames_, 2);

  CHECK_LT(overlap_frames_, options_.chunk_size) << "Overlap needs to be smaller "
    << "than chunk_size. Decrease chunk_overlap_ratio.";

  // At least one constraint frame.
  CHECK_GE(options_.num_constraint_frames, 1);

  // Always need one more frame in the overlap than constraints.
  constraint_frames_ = std::min(options_.num_constraint_frames, overlap_frames_ - 1);

  VLOG(1) << "Performing oversegmentation in chunks of " << options_.chunk_size
          << " frames with " << overlap_frames_ << " frames overlap.";

  feature_buffer_.reserve(options_.chunk_size + 1);
  flow_buffer_.reserve(options_.chunk_size + 1);

  if (!FLAGS_dense_smoothing.empty()) {
    if (FLAGS_dense_smoothing == "bilateral") {
      options_.presmoothing = DenseSegmentationOptions::PRESMOOTH_BILATERAL;
    } else if (FLAGS_dense_smoothing == "gaussian") {
      options_.presmoothing = DenseSegmentationOptions::PRESMOOTH_GAUSSIAN;
    } else {
      LOG(ERROR) << "Undefined smoothing mode specified. Ignoring.";
    }
  }

  if (!FLAGS_dense_color_dist.empty()) {
    if (FLAGS_dense_color_dist == "l1") {
      options_.color_distance = DenseSegmentationOptions::COLOR_DISTANCE_L1;
    } else if (FLAGS_dense_color_dist == "l2") {
      options_.color_distance = DenseSegmentationOptions::COLOR_DISTANCE_L2;
    } else {
      LOG(ERROR) << "Undefined color distance specified. Ignoring.";
    }
  }

  if (FLAGS_dense_min_region_size >= 1e-3) {
    options_.frac_min_region_size = FLAGS_dense_min_region_size;
  }

  // Avoid re-allocation of buffers.
  feature_buffer_.reserve(options_.chunk_size + overlap_frames_);
  flow_buffer_.reserve(options_.chunk_size + overlap_frames_);
}

int DenseSegmentation::ProcessFrame(
    bool flush,
    const std::vector<cv::Mat>* features,
    const cv::Mat* flow,  // optional.
    std::vector<std::unique_ptr<SegmentationDesc>>* results) {
  if (seg_ == nullptr) {
    SegmentationOptions seg_options;
    GetSegmentationOptions(&seg_options);
    seg_.reset(new Segmentation(seg_options, frame_width_, frame_height_, chunk_id_));
    seg_->InitializeOverSegmentation(*this, options_.chunk_size);
  }

  LOG(INFO) << "Processing frame " << input_frames_;

  if (features) {
    // Parallel construction of the graph, buffer features and flow.
    vector<cv::Mat> processed_features(features->size());
    PreprocessFeatures(*features, &processed_features);
    feature_buffer_.push_back(processed_features);

    if (flow) {
      if (input_frames_ == 0) {
        DCHECK(flow->empty()) << "First frame's flow should be empty cv::Mat";
        flow_buffer_.push_back(cv::Mat());
      } else {
        DCHECK_EQ(frame_height_, flow->rows);
        DCHECK_EQ(frame_width_, flow->cols);
        cv::Mat flow_copy(flow->rows, flow->cols, CV_32FC2);
        // Deep copy, as we need to keep the memory around during parallel graph
        // construction.
        flow->copyTo(flow_copy);
        flow_buffer_.push_back(flow_copy);
        CHECK_EQ(flow_buffer_.size(), feature_buffer_.size())
          << "Flow always has to be passed or be absent.";
      }
    }

    AddFrameToSegmentationFromFeatures(feature_buffer_.back(), nullptr);

    if (feature_buffer_.size() > 1) {
      // Connect temporally.
      ConnectSegmentationTemporallyFromFeatures(
          feature_buffer_.end()[-1],
          feature_buffer_.end()[-2],
          flow_buffer_.empty() ? nullptr : &flow_buffer_.back());
    }
    ++input_frames_;
  }

  if (flush || feature_buffer_.size() - curr_chunk_start_ >= options_.chunk_size) {
    ChunkBoundaryOutput(flush, results);
    return results->size();
  }
  return 0;
}

void DenseSegmentation::PreprocessFeatures(
    const std::vector<cv::Mat>& input_features,
    std::vector<cv::Mat>* output_features) {
  CHECK_NOTNULL(output_features);
  CHECK_EQ(1, input_features.size())
    << "Only appearance supported by default DenseSegmentation";

  output_features->resize(1);

  const cv::Mat& input = input_features[0];
  CHECK_EQ(3, input.channels());
  CHECK_EQ(input.rows, frame_height_);
  CHECK_EQ(input.cols, frame_width_);
  output_features->at(0).create(input.rows, input.cols, CV_32FC3);
  cv::Mat& output = (*output_features)[0];

  cv::Mat tmp_frame(input.rows, input.cols, CV_32FC3);
  input.convertTo(tmp_frame, CV_32FC3, 1.0 / 255.0);

  // Smooth image based on our options.
  switch (options_.presmoothing) {
    case DenseSegmentationOptions::PRESMOOTH_NONE:
      output = tmp_frame;
      break;

    case DenseSegmentationOptions::PRESMOOTH_GAUSSIAN:
      cv::GaussianBlur(tmp_frame, output, cv::Size(3, 3), 1.5);
      break;

    case DenseSegmentationOptions::PRESMOOTH_BILATERAL:
      // TODO(grundman): Quite heavy, reduce ?  Original values: 1.5, 0.15
      imagefilter::BilateralFilter(tmp_frame, 3.0, 0.25, &output);
      break;
  }
}

void DenseSegmentation::AddFrameToSegmentationFromFeatures(
    const std::vector<cv::Mat>& curr_features,
    const SegmentationDesc* constrained_segmentation) {
  CHECK_EQ(curr_features.size(), 1) << "Only appearance supported by default "
                                    << "DenseSegmentation.";

  const cv::Mat& input_frame = curr_features[0];
  switch (options_.color_distance) {
    case DenseSegmentationOptions::COLOR_DISTANCE_L2: {
      SpatialCvMatDistance3L2 color_distance(input_frame);
      AddFrameToSegmentation(color_distance, constrained_segmentation);
      break;
    }

    case DenseSegmentationOptions::COLOR_DISTANCE_L1: {
      SpatialCvMatDistance3L1 color_distance(input_frame);
      AddFrameToSegmentation(color_distance, constrained_segmentation);
      break;
    }
  }
}

void DenseSegmentation::ConnectSegmentationTemporallyFromFeatures(
    const std::vector<cv::Mat>& curr_features,
    const std::vector<cv::Mat>& prev_features,
    const cv::Mat* flow) {
  CHECK(curr_features.size() == 1 &&
        prev_features.size() == 1) << "Only appearance supported by default "
                                   << "DenseSegmentation.";

  const cv::Mat& curr_frame = curr_features[0];
  const cv::Mat& prev_frame = prev_features[0];
  switch (options_.color_distance) {
    case DenseSegmentationOptions::COLOR_DISTANCE_L2: {
      TemporalCvMatDistance3L2 temporal_distance(curr_frame, prev_frame);
      ConnectSegmentationTemporally(temporal_distance, flow);
      break;
    }

    case DenseSegmentationOptions::COLOR_DISTANCE_L1: {
      TemporalCvMatDistance3L1 temporal_distance(curr_frame, prev_frame);
      ConnectSegmentationTemporally(temporal_distance, flow);
      break;
    }
  }
}

struct DistanceColorL2 : DistanceTraits<SpatialCvMatDistance3L2,
                                        TemporalCvMatDistance3L2> { };

struct DistanceColorL1 : DistanceTraits<SpatialCvMatDistance3L1,
                                        TemporalCvMatDistance3L1> { };

DenseSegGraphInterface* DenseSegmentation::CreateDenseSegGraph(
    int frame_width, int frame_height, int chunk_size) const {
  switch (options_.color_distance) {
    case DenseSegmentationOptions::COLOR_DISTANCE_L2:
      // TODO(grundman): Option for cutoffs.
      // Exactly one distance.
      return new DenseSegmentationGraph<DistanceColorL2, ColorMeanDescriptorTraits>(
          frame_width, frame_height, chunk_size, ColorMeanDescriptorTraits(0.001f));

    case DenseSegmentationOptions::COLOR_DISTANCE_L1:
      return new DenseSegmentationGraph<DistanceColorL1, ColorMeanDescriptorTraits>(
          frame_width, frame_height, chunk_size, ColorMeanDescriptorTraits(0.002f));
  }
}

void DenseSegmentation::GetSegmentationOptions(SegmentationOptions* options) const {
  CHECK_NOTNULL(options);
  options->min_region_size = options_.frac_min_region_size * frame_width_ *
                             options_.frac_min_region_size * frame_height_ *
                             options_.chunk_size;

  options->two_stage_segmentation = options_.two_stage_oversegment;
  options->enforce_n4_connectivity = options_.enforce_n4_connectivity;
  options->thin_structure_suppression = options_.thin_structure_suppression;
  options->compute_vectorization = options_.compute_vectorization;
}

void DenseSegmentation::ChunkBoundaryOutput(
    bool flush, std::vector<std::unique_ptr<SegmentationDesc>>* results) {
  LOG(INFO) << "Chunk boundary reached " << chunk_id_;
  SegmentAndOutputChunk(flush, results);

  if (flush) {
    seg_.reset();
    return;
  }

  // Initialize new Segmentation.
  VLOG(1) << "Creating new segmentation object with "
          << feature_buffer_.size() << " frames.";

  SegmentationOptions seg_options;
  GetSegmentationOptions(&seg_options);
  seg_.reset(new Segmentation(seg_options, frame_width_, frame_height_, chunk_id_));
  seg_->InitializeOverSegmentation(*this, curr_chunk_start_ + options_.chunk_size);

  CHECK_EQ(overlap_segmentations_.size(), constraint_frames_ + 1);

  // Add first two constraint images, very first as virtual nodes. Also connect via
  // virtual edges.
  CHECK_GE(overlap_segmentations_.size(), 2);
  seg_->AddVirtualImageConstrained(*overlap_segmentations_[0]);

  AddFrameToSegmentationFromFeatures(feature_buffer_[1],
                                     overlap_segmentations_[1].get());

  // Connect using virtual edges.
  if (!flow_buffer_.empty()) {
    seg_->ConnectTemporallyVirtualAlongFlow(flow_buffer_[1]);
  } else {
     seg_->ConnectTemporallyVirtual();
  }

  // Add remaining overlap images with constraints.
  for (int i = 2; i < overlap_frames_; ++i) {
    // Add with constraints.
    AddFrameToSegmentationFromFeatures(
        feature_buffer_[i],
        i < constraint_frames_ ? overlap_segmentations_[i].get() : nullptr);

    ConnectSegmentationTemporallyFromFeatures(
        feature_buffer_[i],
        feature_buffer_[i - 1],
        flow_buffer_.empty() ? nullptr : &flow_buffer_[i]);
  }

  overlap_segmentations_.clear();
}

void DenseSegmentation::SegmentAndOutputChunk(
    bool flush, std::vector<std::unique_ptr<SegmentationDesc>>* results) {
  // Compute resolution dependent min region size.
  seg_->RunOverSegmentation();

  const int overlap_start = feature_buffer_.size() - (flush ? 0 : overlap_frames_);

  // Always output first frame in overlap.
  const int last_output_frame = std::min<int>(feature_buffer_.size() - 1, overlap_start);

  CHECK_GT(overlap_start, curr_chunk_start_);

  // Only get results for frames in constrained overlap, ignore look ahead region.
  const int max_result_frame = std::min<int>(feature_buffer_.size() - 1,
                                             last_output_frame + constraint_frames_);

  LOG(INFO) << "Outputting from " << num_output_frames_ << " to "
            << num_output_frames_ + last_output_frame;

  // Include very first frame of overlap in output (necessary for correct connections,
  // these can be zero region area regions).
  seg_->ConstrainSegmentationToFrameInterval(0, last_output_frame + 1);

  // If first frame has virtual nodes, their area is zero anyways.
  seg_->AdjustRegionAreaToFrameInterval(0, last_output_frame + 1);

  // Assign unique ids and update.
  vector<int> new_max_region_id(1);
  const bool use_constraints = chunk_id_ > 0;
  seg_->AssignUniqueRegionIds(use_constraints,
                              vector<int>(1, max_region_id_),
                              &new_max_region_id);
  max_region_id_ = new_max_region_id[0];

  const int chunk_size = last_output_frame - curr_chunk_start_ + 1;

  // Retrieve results for frames from chunk_start to overlap (inclusive constraints).
  // Output only till overlap is reached.
  results->clear();
  results->reserve(max_result_frame);
  overlap_segmentations_.clear();
  const int hierarchy_frame_idx = num_output_frames_;

  for (int frame_idx = curr_chunk_start_; frame_idx <= max_result_frame; ++frame_idx) {
    // Retrieve segmentations and output.
    std::unique_ptr<SegmentationDesc> desc(new SegmentationDesc());
    const bool output_hierarchy = frame_idx == curr_chunk_start_;
    seg_->RetrieveSegmentation3D(frame_idx,
                                 output_hierarchy,
                                 false,       // No descriptors saved.
                                 desc.get());

    desc->set_chunk_size(chunk_size);
    desc->set_overlap_start(chunk_size);
    desc->set_hierarchy_frame_idx(hierarchy_frame_idx);

    // Output, also always output very first constrained frame.
    if (frame_idx <= last_output_frame) {
      if (frame_idx < last_output_frame) {
        results->push_back(std::move(desc));
      } else {
        std::unique_ptr<SegmentationDesc> copy(new SegmentationDesc(*desc));
        results->push_back(std::move(copy));
      }
      ++num_output_frames_;
    }

    // Buffer segmentation results in overlap.
    if (frame_idx >= last_output_frame) {
      overlap_segmentations_.push_back(std::move(desc));
    }
  }

  // Clear corresponding feature frames.
  feature_buffer_.erase(feature_buffer_.begin(),
                        feature_buffer_.begin() + last_output_frame);
  if (!flow_buffer_.empty()) {
    flow_buffer_.erase(flow_buffer_.begin(),
                       flow_buffer_.begin() + last_output_frame);
  }

  curr_chunk_start_ = flush ? 0 : 1;

  if (!flush) {
    CHECK_EQ(overlap_frames_, feature_buffer_.size());

    // We dont need the features in the very first buffered frame. Using as virtual
    // nodes only.
    for (auto& feature : feature_buffer_[0]) {
      feature = cv::Mat();
    }

    if (!flow_buffer_.empty()) {
      CHECK_EQ(overlap_frames_, flow_buffer_.size());
      flow_buffer_[0] = cv::Mat();
    }
  }

  ++chunk_id_;
}

}  // namespace segmentation.
