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

#ifndef VIDEO_SEGMENT_SEGMENTATION_BOUNDARY_H__
#define VIDEO_SEGMENT_SEGMENTATION_BOUNDARY_H__

#include "base/base.h"

#include <opencv2/core/core.hpp>

#include "segment_util/segmentation.pb.h"

namespace segmentation {

// Represents the boundary for a single 2D region.
// Note domain for boundaries is [0, frame_width] x [0, frame_height], i.e.
// bottom and right border are outside frame domain.
struct Boundary {
  // Vertices specify start and end position of Boundary::Segments
  struct Vertex {
    cv::Point2i pt;

    // Number of incident boundaries. Value between 1 to 4. For closed loop regions such
    // as hole an arbitrary vertex of order one is used.
    // Only set for start index (see below).
    int order = 0;
  };

  struct Segment {
    Vertex start;
    Vertex end;

    // Id's of left and right incident region. Only set if at least one non-vertex exists.
    int left_region = -1;
    int right_region = -1;

    // Points from start to end describing the segment (including start and end).
    std::vector<cv::Point2i> points;

    std::string ToString() const;
  };

  std::vector<Segment> segments;
  int region = -1;
  bool is_hole = false;

  // A simple boundary is defined as having only one segment and no vertices.
  bool IsSimple() const {
    return segments.size() == 1 && segments[0].start.order == 1;
  }

  int Length() const {
    int length = 0;
    for (const Segment& seg : segments) {
      length += seg.points.size() - 1;
    }
    return length;
  }
};

// Computes the joint boundary for a 2D segmentation result.
// Based on:
// "A contour tracing algorithm that preserves common boundaries between regions"
// Yuh-Tay Liow, CVGIP: Image Understanding Volume 53, Issue 3, May 1991, Pages 313–321
class BoundaryComputation {
 public:
  // Initializes new boundary computation for specified domain. Holes below
  // specified length are dropped.
  BoundaryComputation(int frame_width, int frame_height, int min_hole_length);

  // Computes boundaries for each component of a 2D region in seg.
  // Requires N4 connected input segmentation, CHECKED.
  void ComputeBoundary(const SegmentationDesc& seg,
                       std::vector<std::unique_ptr<Boundary>>* boundaries);

  // Fills in vectorization for segmentation from computed boundaries.
  void ComputeVectorization(
      const std::vector<std::unique_ptr<Boundary>>& boundaries,
      int min_segment_length,  // Needs to be >= 3 (we don't store segments without
      float max_error,         //                   interior).
      SegmentationDesc* seg);

 protected:
  // Freeman coding scheme. Direction of next pixel w.r.t. current one X
  // 3 2 1
  // 4 X 0
  // 5 6 7
  enum Direction {
    D_R = 0,
    D_TR = 1,
    D_T = 2,
    D_TL = 3,
    D_L = 4,
    D_BL = 5,
    D_B = 6,
    D_BR = 7,
  };

  // Conversion functions from and to Direction.
  cv::Point2i DirectionToVector(Direction dir) const { return freeman_map_[(int)dir]; }
  int DirectionToOffset(Direction dir) const { return offset_map_[(int)dir]; }
  Direction VectorToDirection(cv::Point2i vec) const;

  // Traces boundary for region_id beginning at start_pt in direction dir.
  void TraceBoundary(int region_id,
                     const cv::Point2i& start_pt,
                     Direction dir,
                     Boundary* boundary) const;

  // Returns next direction from current one.
  Direction NextDirection(const int32_t* curr_position,
                          Direction prev_dir,
                          int region_id) const;

  // Returns order (1 - 4) for specified offset location (into region id image).
  int VertexOrder(const int32_t* curr_position) const;

  // Set segment's left and right region id from current position and direction.
  void SetSegmentRegions(const int32_t* curr_position,
                         Direction prev_dir,
                         Boundary::Segment* segment) const;

  bool IsFrameRectanglePoint(const cv::Point2i& pt) const;
  bool IsFrameRectangleSegment(const Boundary::Segment& segment) const;


 private:
  int frame_width_;
  int frame_height_;
  int min_hole_length_;
  std::vector<cv::Point2i> freeman_map_;

  // Corresponding offset for each code w.r.t. id_image_.
  std::vector<int> offset_map_;

  // Image with one border boundary.
  cv::Mat id_image_;

  // View without boundary.
  cv::Mat id_view_;
};

// Unique key for pair-wise boundaries.
struct BoundarySegmentKey {
  BoundarySegmentKey(const Boundary::Segment& segment);

  bool operator==(const BoundarySegmentKey& rhs) const {
    return pt_1 == rhs.pt_1 &&
           pt_2 == rhs.pt_2 &&
           region_1 == rhs.region_1 &&
           region_2 == rhs.region_2;
  }

  // Debug function.
  std::string ToString() const;

  cv::Point2i pt_1;
  cv::Point2i pt_2;
  int region_1 = -1;
  int region_2 = -1;
};

struct BoundarySegmentHasher {
  BoundarySegmentHasher(int frame_width) : frame_width_(frame_width) { }

  // Hash based on starting vertex and region ids.
  size_t operator()(const BoundarySegmentKey& key) const {
    return (key.pt_1.y * frame_width_ + key.pt_1.x) * 10 +
      (key.region_1 % 7 + key.region_2 % 3);
  }

  int frame_width_ = 0;
};

typedef std::unordered_map<BoundarySegmentKey, const Boundary::Segment*,
        BoundarySegmentHasher> BoundarySegmentPtrHash;

typedef std::unordered_map<BoundarySegmentKey, int,
        BoundarySegmentHasher> BoundarySegmentIdxHash;


}  // namespace segmentation.


#endif  // VIDEO_SEGMENT_SEGMENTATION_BOUNDARY_H__
