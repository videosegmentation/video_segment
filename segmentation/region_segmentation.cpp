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

#include "region_segmentation.h"

#include <gflags/gflags.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "base/base_impl.h"
#include "region_descriptor.h"

DEFINE_int32(min_region_num, 0, "If set > specifies number of minimum regions.");
DEFINE_int32(max_region_num, 0, "If set >0 specifies number of maximum regions");
DEFINE_double(level_cutoff_fraction, 0.0, "If set > 0 specifies cutoff fraction. "
                                         "Within (0, 0.95)");
DEFINE_double(small_region_penalizer, -1, "If set >= 0 defines amount small regions "
                                         "are penalized. ");
DEFINE_int32(chunk_set_size, 0, "If set >= 2, specifies the chunk set size for "
                                "hierarchical segmentation.");

namespace segmentation {

RegionSegmentation::RegionSegmentation(const RegionSegmentationOptions& options,
                                       int frame_width,
                                       int frame_height)
    : options_(options), frame_width_(frame_width), frame_height_(frame_height) {
  CHECK_GT(options_.chunk_set_size, 1)
      << "At least two chunks per chunk_set required.";

  if (FLAGS_chunk_set_size >= 2) {
    options_.chunk_set_size = FLAGS_chunk_set_size;
  }

  CHECK_GT(options_.chunk_set_overlap, 0)
      << "At least one chunk in overlap expected.";

  CHECK_LT(options_.chunk_set_overlap, options_.chunk_set_size)
      << "Overlap has to be strictly smaller than a chunk set.";

  CHECK_LE(options_.constraint_chunks, options_.chunk_set_overlap)
      << "Constraints must be smaller or equal to overlap";

  num_descriptors_ = 0;
  if (options_.use_appearance) {
    ++num_descriptors_;
  }
  if (options_.use_flow) {
    ++num_descriptors_;
  }
  if (options_.use_size_penalizer) {
    ++num_descriptors_;
  }

  CHECK(options_.use_appearance || options_.use_flow)
    << "At least apperance or flow need to be set.";


  if (FLAGS_min_region_num > 0) {
    options_.min_region_num = FLAGS_min_region_num;
  }
  if (FLAGS_max_region_num > 0) {
    options_.max_region_num = FLAGS_max_region_num;
  }
  if (FLAGS_level_cutoff_fraction > 0) {
    options_.level_cutoff_fraction = std::max(0.95, FLAGS_level_cutoff_fraction);
  }
  if (FLAGS_small_region_penalizer >= 0) {
    options_.small_region_penalizer = FLAGS_small_region_penalizer;
  }
}

int RegionSegmentation::ProcessFrame(
    bool flush,
    const SegmentationDesc* desc,
    const std::vector<cv::Mat>* features,
    std::vector<std::unique_ptr<SegmentationDesc>>* results) {
  if (desc == nullptr || features == nullptr) {
    CHECK(desc == nullptr &&
          features == nullptr) << "Requring both segmentation and features to be "
                               << "either set or null.";
  }

  if (seg_ == nullptr) {
    SegmentationOptions seg_options;
    GetSegmentationOptions(&seg_options);
    seg_.reset(new Segmentation(seg_options, frame_width_, frame_height_, chunk_sets_));
  }

  const int chunk_set_overlap_start =
    options_.chunk_set_size - options_.chunk_set_overlap;
  const int chunk_set_lookahead_start =
    chunk_set_overlap_start + options_.constraint_chunks;

  if (features) {
    // Get DescriptorExtractorList.
    DescriptorExtractorList extractor_list;
    DescriptorUpdaterList updater_list;
    GetDescriptorExtractorAndUpdaters(*features, &extractor_list, &updater_list);

    CHECK_EQ(NumDescriptors(), extractor_list.size());
    CHECK_EQ(extractor_list.size(), updater_list.size());

    // Set on each chunk boundary.
    bool is_chunk_boundary = false;
    if (desc->hierarchy_size() > 0) {
      ++read_chunks_;
      is_chunk_boundary = true;
    }

    // Is this the first frame in a new chunk set? -> Segment last chunk_set
    if(read_chunks_ > 0 &&
       read_chunks_ % options_.chunk_set_size == 0 &&
       is_chunk_boundary) {
      ChunkBoundaryOutput(false, results);
    }

    // Is this an overlap chunk?
    if (read_chunks_ % options_.chunk_set_size >= chunk_set_overlap_start) {
      if (new_seg_ == nullptr) {
        SegmentationOptions seg_options;
        GetSegmentationOptions(&seg_options);
        new_seg_.reset(new Segmentation(seg_options, frame_width_, frame_height_,
                                        chunk_sets_ + 1));
      }

      // Remember first frame in overlap, if not set yet.
      if (overlap_start_ < 0) {
        overlap_start_ = seg_->NumFramesAdded();
      }

      if (is_chunk_boundary) {
        // Use as constraints?
        RegionMapping mapping;
        RegionMapping* mapping_ptr = nullptr;
        // Only use RegionMapping within the overlap.
        if (read_chunks_ % options_.chunk_set_size < chunk_set_lookahead_start) {
          mapping_ptr = &mapping;
        }

        seg_->InitializeBaseHierarchyLevel(desc->hierarchy(0),
                                           extractor_list,
                                           updater_list,
                                           nullptr,
                                           mapping_ptr);

        new_seg_->InitializeBaseHierarchyLevel(desc->hierarchy(0),
                                               extractor_list,
                                               updater_list,
                                               mapping_ptr,
                                               nullptr);
      }

      seg_->AddOverSegmentation(*desc, extractor_list);
      new_seg_->AddOverSegmentation(*desc, extractor_list);
    } else {
      // Not in an overlap.
      if (is_chunk_boundary) {
        seg_->InitializeBaseHierarchyLevel(desc->hierarchy(0),
                                           extractor_list,
                                           updater_list,
                                           nullptr,      // no constraints.
                                           nullptr);
      }

      seg_->AddOverSegmentation(*desc, extractor_list);
    }

    // First frame in lookahead region (outside of contraints)? If so, remember.
    if (read_chunks_ % options_.chunk_set_size >= chunk_set_lookahead_start &&
        lookahead_start_ < 0) {
      lookahead_start_ = seg_->NumFramesAdded();
    }
  }  // End if features.

  if (flush) {
    ChunkBoundaryOutput(true, results);
  }

  return results->size();
}

std::unique_ptr<RegionDistance> RegionSegmentation::GetRegionDistance() {
  std::unique_ptr<RegionDistance> distance;
  if (options_.use_size_penalizer) {
    // Size penalizer counts as separate desciptor.
    switch (num_descriptors_) {
      case 3:
        distance.reset(new SquaredORDistanceSizePenalized<2>());
        break;
      case 2:
        distance.reset(new SquaredORDistanceSizePenalized<1>());
        break;
      default:
        LOG(FATAL) << "Unexpected number of features.";
    }
  } else {
    switch (num_descriptors_) {
      case 2:
        distance.reset(new SquaredORDistance<2>());
        break;
      case 1:
        distance.reset(new SquaredORDistance<1>());
        break;
      default:
        LOG(FATAL) << "Unexpected number of features.";
    }
  }
  return std::move(distance);
}


void RegionSegmentation::GetDescriptorExtractorAndUpdaters(
      const std::vector<cv::Mat>& features,
      DescriptorExtractorList* extractors,
      DescriptorUpdaterList* updaters) {
  CHECK_NOTNULL(extractors);
  CHECK_NOTNULL(updaters);

  int feature_idx = 0;
  if (options_.use_appearance) {
    shared_ptr<AppearanceExtractor> appearance_extractor(
        new AppearanceExtractor(options_.luminance_bins,
                                options_.color_bins,
                                0,                         // window_size.
                                features[feature_idx],
                                prev_appearance_extractor_.get()));
    extractors->push_back(appearance_extractor);
    updaters->push_back(shared_ptr<NonMutableUpdater>(new NonMutableUpdater()));
    prev_appearance_extractor_.swap(appearance_extractor);
    ++feature_idx;
  }

  if (options_.use_flow) {
    if (!features[feature_idx].empty()) {
      shared_ptr<FlowExtractor> flow_extractor(
          new FlowExtractor(options_.flow_bins, features[feature_idx]));
      extractors->push_back(flow_extractor);
    } else {
      extractors->push_back(shared_ptr<FlowExtractor>(
          new FlowExtractor(options_.flow_bins)));
    }
    updaters->push_back(shared_ptr<NonMutableUpdater>(new NonMutableUpdater()));
    ++feature_idx;
  }

  if (options_.use_size_penalizer) {
    extractors->push_back(shared_ptr<RegionSizePenalizerExtractor>(
        new RegionSizePenalizerExtractor(options_.small_region_penalizer)));

    updaters->push_back(shared_ptr<RegionSizePenalizerUpdater>(
        new RegionSizePenalizerUpdater()));
  }
}

void RegionSegmentation::GetSegmentationOptions(SegmentationOptions* options) const {
  CHECK_NOTNULL(options);
  options->min_region_num = options_.min_region_num;
  options->max_region_num = options_.max_region_num;
  options->level_cutoff_fraction = options_.level_cutoff_fraction;
  options->compute_vectorization = options_.compute_vectorization;
}

void RegionSegmentation::ChunkBoundaryOutput(
    bool flush,
    std::vector<std::unique_ptr<SegmentationDesc>>* results) {
  if (!flush) {
    int look_ahead = lookahead_start_ > 0 ? lookahead_start_
                                          : seg_->NumFramesAdded();
    SegmentAndOutputChunk(overlap_start_, look_ahead, results);
  } else {
    SegmentAndOutputChunk(seg_->NumFramesAdded(),
                          seg_->NumFramesAdded(), results);
  }

  // Reset frame indices.
  overlap_start_ = -1;
  lookahead_start_ = -1;

  if (!flush) {
    // Use new segmentation.
    seg_.swap(new_seg_);
    new_seg_.reset();
  } else {
    seg_.reset();
  }
}

void RegionSegmentation::SegmentAndOutputChunk(
    int overlap_start,
    int lookahead_start,
    std::vector<std::unique_ptr<SegmentationDesc>>* results) {
  // Segment all frames hierarchically and output result.
  std::unique_ptr<RegionDistance> distance(GetRegionDistance());
  seg_->RunHierarchicalSegmentation(*distance, true);
  int computed_levels = seg_->ComputedHierarchyLevels();

  if (computed_levels > max_region_ids_.size()) {
    max_region_ids_.resize(computed_levels, 0);
  }

  // Chunk set processing cleanup.
  seg_->ConstrainSegmentationToFrameInterval(0, lookahead_start);
  seg_->AdjustRegionAreaToFrameInterval(0, overlap_start);

  vector<int> new_max_region_ids(max_region_ids_.size());
  seg_->AssignUniqueRegionIds(chunk_sets_ > 0,
                              max_region_ids_,
                              &new_max_region_ids);

  max_region_ids_.swap(new_max_region_ids);

  // Pull segmentation result in overlap to new segmentation object.
  if (new_seg_ != nullptr) {
    new_seg_->PullCounterpartSegmentationResult(*seg_);
  }

  // Discard bottom hierarchy level here.
  seg_->DiscardBottomLevel();

  VLOG(1) << "Outputting segmentation of size "
          << overlap_start << " frames.";
  const int hierarchy_frame_idx = num_output_frames_;

  for (int frame_idx = 0; frame_idx < overlap_start; ++frame_idx) {
    std::unique_ptr<SegmentationDesc> desc(new SegmentationDesc);
    const bool output_hierarchy = frame_idx == 0;
    seg_->RetrieveSegmentation3D(frame_idx,
                                 output_hierarchy,
                                 options_.save_descriptors,
                                 desc.get());
    desc->set_hierarchy_frame_idx(hierarchy_frame_idx);
    desc->set_chunk_size(lookahead_start);
    desc->set_overlap_start(overlap_start);
    results->push_back(std::move(desc));
    LOG(INFO) << "Region output frame " << num_output_frames_;
    ++num_output_frames_;
  }

  ++chunk_sets_;
}

}  // namespace segmentation.
