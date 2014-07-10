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

#include "segmentation/dense_segmentation_graph.h"

DEFINE_bool(parallel_graph_construction, true, "If set creates graph edges in parallel.");

namespace segmentation {

float AverageTubeSliceSize(const Tube3D& ts) {
  if (ts.empty()) {
    return 0;
  }

  float area_sum = 0;
  for (const auto& slice : ts) {
    area_sum += slice.shape.size;
  }
  return area_sum / ts.size();
}

void MergeTube3D(const Tube3D& lhs, const Tube3D& rhs, Tube3D* result) {
  int lhs_idx = 0;
  int rhs_idx = 0;
  DCHECK_NE(&lhs, result);
  DCHECK_NE(&rhs, result);

  if (lhs.empty()) {
    *result = rhs;
    return;
  }

  if (rhs.empty()) {
    *result = lhs;
    return;
  }

  // Merge in lock-step.
  while (lhs_idx < lhs.size() && rhs_idx < rhs.size()) {
    if (lhs[lhs_idx].frame < rhs[rhs_idx].frame) {
      result->push_back(lhs[lhs_idx++]);
    } else if (lhs[lhs_idx].frame > rhs[rhs_idx].frame) {
      result->push_back(rhs[rhs_idx++]);
    } else {
      DCHECK_EQ(lhs[lhs_idx].frame, rhs[rhs_idx].frame);
      TubeSlice merged = lhs[lhs_idx];
      merged.MergeFrom(rhs[rhs_idx]);
      result->push_back(merged);
      ++lhs_idx;
      ++rhs_idx;
    }
  }

  // Add ends.
  while (lhs_idx < lhs.size()) {
    result->push_back(lhs[lhs_idx++]);
  }
  while (rhs_idx < rhs.size()) {
    result->push_back(rhs[rhs_idx++]);
  }
}

bool AreTubesTemporalNeighbors(const Tube3D& lhs, const Tube3D& rhs) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }

  ShapeDescriptor a, b;
  if (lhs[0].frame - 1 == rhs.back().frame) {
    a = lhs[0].shape;
    b = rhs.back().shape;
  } else if (lhs.back().frame + 1 == rhs[0].frame) {
    a = lhs.back().shape;
    b = rhs[0].shape;
  } else {
    return false;
  }

  const float size_ratio = std::min(a.size, b.size) * (1.0f / std::max(a.size, b.size));
  const cv::Point2f diff = a.center - b.center;
  LOG(INFO) << "Neighbors: " << size_ratio << " : " << diff;
  return size_ratio > 0.9 && hypot(diff.y, diff.x) < 20;
}

float AverageTubeDistance(const Tube3D& lhs, const Tube3D& rhs) {
  if (lhs.empty() || rhs.empty()) {
    return std::numeric_limits<float>::max();
  }

  int start_frame = std::max(lhs[0].frame, rhs[0].frame);
  int end_frame = std::min(lhs.back().frame, rhs.back().frame);

  int lhs_idx = 0;
  int rhs_idx = 0;
  float diff_sum = 0;
  int weight = 0;
  for (int f = start_frame; f <= end_frame; ++f) {
    // Advance to current frame.
    while (lhs[lhs_idx].frame < f) { ++lhs_idx; }
    while (rhs[rhs_idx].frame < f) { ++rhs_idx; }

    // Might not be present at current frame.
    if (lhs[lhs_idx].frame != f || rhs[rhs_idx].frame != f) {
      continue;
    }

    // Both point to f here.
    DCHECK_EQ(f, lhs[lhs_idx].frame);
    DCHECK_EQ(f, rhs[rhs_idx].frame);

    const cv::Point2f center_diff = lhs[lhs_idx].shape.center - rhs[rhs_idx].shape.center;
    diff_sum += hypot(center_diff.y, center_diff.x);
    ++weight;
  }

  if (weight > 0) {
    return diff_sum / weight;
  } else {
    return std::numeric_limits<float>::max();
  }
}

float Tube3DIntersection(const Tube3D& lhs, const Tube3D& rhs) {
  if (lhs.empty() || rhs.empty()) {
    return std::numeric_limits<float>::max();
  }

  int start_frame = std::max(lhs[0].frame, rhs[0].frame);
  int end_frame = std::min(lhs.back().frame, rhs.back().frame);

  int lhs_idx = 0;
  int rhs_idx = 0;
  int intersect_count = 0;
  int weight = 0;

  for (int f = start_frame; f <= end_frame; ++f) {
    // Advance to current frame.
    while (lhs[lhs_idx].frame < f) { ++lhs_idx; }
    while (rhs[rhs_idx].frame < f) { ++rhs_idx; }

    // Might not be present at current frame.
    if (lhs[lhs_idx].frame != f || rhs[rhs_idx].frame != f) {
      continue;
    }

    // Both point to f here.
    DCHECK_EQ(f, lhs[lhs_idx].frame);
    DCHECK_EQ(f, rhs[rhs_idx].frame);

    std::vector<cv::Point2f> lhs_box;
    std::vector<cv::Point2f> rhs_box;
    ShapeDescriptorBox(lhs[lhs_idx].shape, 10, &lhs_box);
    ShapeDescriptorBox(rhs[rhs_idx].shape, 10, &rhs_box);

    if (ShapeDescriptorBoxesIntersect(lhs_box, rhs_box)) {
      ++intersect_count;
    }
    ++weight;
  }

  if (weight > 0) {
    return intersect_count * (1.0f / weight);
  } else {
    return std::numeric_limits<float>::max();
  }
}

int GetClosestTube3D(const Tube3D& tube,
                     const std::vector<Tube3D>& tubes,
                     int ignore_index) {
  float min_dist = std::numeric_limits<float>::max();
  int min_idx = -1;
  for (int k = 0; k < tubes.size(); ++k) {
    if (k == ignore_index) {
      continue;
    }

    const float tube_dist = AverageTubeDistance(tube, tubes[k]);
    if (tube_dist < min_dist) {
      min_dist = tube_dist;
      min_idx = k;
    }
  }
  return min_idx;
}

}  // namespace segmentation.

