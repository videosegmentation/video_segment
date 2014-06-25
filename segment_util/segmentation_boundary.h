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

#ifndef VIDEO_SEGMENT_SEGMENT_UTIL_SEGMENTATION_BOUNDARY_H__
#define VIDEO_SEGMENT_SEGMENT_UTIL_SEGMENTATION_BOUNDARY_H__

#include "base/base.h"
#include "segment_util/segmentation_util.h"

// Boundary methods for single regions. Does not create unique boundaries
// between regions. For unique boundaries each region has a vectorization.

namespace segmentation {

// Graph traversal operations.
// Note region id is not index of region in protobuffer necessarily.
struct BoundaryPoint {
  BoundaryPoint() : x(0), y(0) {}
  BoundaryPoint(int _x, int _y) : x(_x), y(_y) {}
  bool operator<(const BoundaryPoint& rhs) const {
    return ((*this).y < rhs.y) || ((*this).y == rhs.y && (*this).x < rhs.x);
  }
  int x;
  int y;
};

// Lexicographic compare.
class BoundaryPointComparator
    : public std::binary_function<bool, BoundaryPoint, BoundaryPoint> {
public:
  bool operator()(const BoundaryPoint& lhs, const BoundaryPoint& rhs) {
    return (lhs.x < rhs.x) || (lhs.x == rhs.x && lhs.y < rhs.y);
  }
};

// A region boundary is always sorted w.r.t. lexicographic order.
typedef std::vector<BoundaryPoint> RegionBoundary;

// Returns boundary for a rasterization by internally rendering the region and
// evaluating for each pixel if it is a boundary pixel. Uses N4 neighborhood, i.e.
// a boundary point is an inner (outer) boundary point, if the point is a region
// (non-region) pixel neighboring to a non-region (region) pixel.
// Uses a temporary buffer of size 3 * (frame_width + 2).
void GetBoundary(const Rasterization& raster,
                 int frame_width,
                 bool inner_boundary,
                 std::vector<uint8_t>* buffer,
                 RegionBoundary* boundary);

// Same as above for union of rasterizations.
void GetBoundary(const std::vector<const Rasterization*>& rasters,
                 int frame_width,
                 bool inner_boundary,
                 std::vector<uint8_t>* buffer,
                 RegionBoundary* boundary);

}  // namespace segmentation.

#endif  // VIDEO_SEGMENT_SEGMENT_UTIL_SEGMENTATION_BOUNDARY_H__
