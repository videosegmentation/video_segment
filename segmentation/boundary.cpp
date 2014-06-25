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

#include "segmentation/boundary.h"
#include "base/base_impl.h"

#include <opencv2/imgproc/imgproc.hpp>

#include "segment_util/segmentation_util.h"

namespace segmentation {

typedef SegmentationDesc::Rasterization Rasterization;

std::string Boundary::Segment::ToString() const {
  std::string res = base::StringPrintf("Segment with %d points:\n", points.size());
  res += base::StringPrintf("Start: %d, %d\n", start.pt.x, start.pt.y);
  res += base::StringPrintf("End: %d, %d\n", end.pt.x, end.pt.y);
  res += base::StringPrintf("Left: %d, Right: %d\n", left_region, right_region);

  for (const auto& pt : points) {
    res += base::StringPrintf("> %d, %d\n", pt.x, pt.y);
  }
  return res;
}

BoundaryComputation::BoundaryComputation(
    int frame_width, int frame_height, int min_hole_length)
    : frame_width_(frame_width),
      frame_height_(frame_height),
      min_hole_length_(min_hole_length) {
  // Init freeman map.
  // Freeman coding scheme w.r.t. X:
  // 3 2 1
  // 4 X 0
  // 5 6 7
  freeman_map_ = vector<cv::Point2i>{
    cv::Point2i(1, 0),
    cv::Point2i(1, -1),
    cv::Point2i(0, -1),
    cv::Point2i(-1, -1),
    cv::Point2i(-1, 0),
    cv::Point2i(-1, 1),
    cv::Point2i(0, 1),
    cv::Point2i(1, 1)
  };

  id_image_ = cv::Mat(frame_height_ + 2, frame_width_ + 2, CV_32S);
  id_image_.setTo(-1);

  id_view_ = cv::Mat(id_image_, cv::Rect(1, 1, frame_width_, frame_height_));

  for (const auto& pt : freeman_map_) {
    offset_map_.push_back(pt.x + pt.y * id_view_.step1());
  }
}

BoundaryComputation::Direction BoundaryComputation::VectorToDirection(cv::Point2i vec) const {
  switch (vec.y) {
    case -1:
      switch (vec.x) {
        case -1:
          return D_TL;
        case 0:
          return D_T;
        case 1:
          return D_TR;
        default:
          LOG(FATAL) << "Unexpected: " << vec.x << ", " << vec.y;
      }
    case 0:
      switch (vec.x) {
        case -1:
          return D_L;
        case 1:
          return D_R;
        default:
          LOG(FATAL) << "Unexpected: " << vec.x << ", " << vec.y;
      }
    case 1:
      switch (vec.x) {
        case -1:
          return D_BL;
        case 0:
          return D_B;
        case 1:
          return D_BR;
        default:
          LOG(FATAL) << "Unexpected: " << vec.x << ", " << vec.y;
      }
    default:
      LOG(FATAL) << "Unexpected: " << vec.x << ", " << vec.y;
  }
}

void BoundaryComputation::ComputeBoundary(
    const SegmentationDesc& seg,
    std::vector<std::unique_ptr<Boundary>>* boundaries) {
  CHECK_EQ(SegmentationDesc::N4_CONNECT, seg.connectedness())
    << "Requires N4 connected segmentation.";

  // Render to id_image.
  SegmentationDescToIdImage(0, seg, nullptr, &id_view_);

  for (int i = 0; i < id_view_.rows; ++i) {
    int32_t* ptr = id_view_.ptr<int32_t>(i);
    for (int j = 0; j < id_view_.cols; ++j) {
      DCHECK_GE(ptr[j], 0) << "ERR: " << j << ", " << i;
    }
  }

  // Split each region into connected components.
  for (const auto& region : seg.region()) {
    std::vector<Rasterization> components;
    ConnectedComponents(region.raster(), N8_CONNECT, &components);

    // Compute start point for each component. Use top-left one.
    std::vector<cv::Point2i> start_points;

    for (const auto& comp : components) {
      CHECK_GT(comp.scan_inter_size(), 0);
      start_points.push_back(cv::Point2i(comp.scan_inter(0).left_x(),
                                         comp.scan_inter(0).y()));
    }

    for (const auto& start_pt : start_points) {
      // Trace boundary for each component.
      std::unique_ptr<Boundary> boundary(new Boundary());
      TraceBoundary(region.id(), start_pt, D_B, boundary.get());

      // Remove small holes.
      if (boundary->IsSimple() && boundary->Length() < min_hole_length_) {
        continue;
      }

      boundaries->push_back(std::move(boundary));
    }
  }

  // Done tracing boundaries. However, we might have the corresponding
  // inner boundary for each hole.
  // For this we hash all boundaries and detect those without any corresponding
  // boundary.
  BoundarySegmentPtrHash boundary_hash(boundaries->size() * 20,
                                       BoundarySegmentHasher(frame_width_));

  for (const auto& boundary_ptr : *boundaries) {
    for (const auto& segment : boundary_ptr->segments) {
      if (segment.points.size() < 3) {
        // Can't hash segments without interior.
        continue;
      }
      if (IsFrameRectangleSegment(segment)) {
        // Frame segments have no corresponding boundary.
        continue;
      }

      BoundarySegmentKey key(segment);
      auto hash_pos = boundary_hash.find(key);

      if (hash_pos == boundary_hash.end()) {
        // First insert.
        boundary_hash[key] = &segment;
      } else {
        // Corresponding segment, remove.
        boundary_hash[key] = nullptr;
      }
    }
  }

  // Traverse hash to find non-corresponding holes.
  for (const auto& elem : boundary_hash) {
    if (elem.second != nullptr) {
      // Hole region found.
      std::unique_ptr<Boundary> hole(new Boundary());

      const Boundary::Segment* seg = elem.second;

      // Trace in opposite direction (for the other region).
      TraceBoundary(seg->right_region,
                    seg->points.back(),
                    VectorToDirection(seg->points[seg->points.size() - 2] -
                                      seg->points[seg->points.size() - 1]),
                    hole.get());
      hole->is_hole = true;

      for (const auto& segment : hole->segments) {
        if (segment.points.size() < 3) {
          // Can't hash segments without interior.
          continue;
        }

        // Holes can't be incident to the frame.
        DCHECK(!IsFrameRectangleSegment(segment))
          << segment.ToString();

        BoundarySegmentKey key(segment);
        auto hash_pos = boundary_hash.find(key);
        DCHECK(hash_pos != boundary_hash.end());
        DCHECK(hash_pos->second != nullptr);

        // Clear hole.
        boundary_hash[key] = nullptr;
      }

      boundaries->push_back(std::move(hole));
    }
  }

  // Check that all entries are nullptr's in the hash now.
  for (const auto& elem : boundary_hash) {
    DCHECK(elem.second == nullptr);
  }
}

void BoundaryComputation::TraceBoundary(int region_id,
                                        const cv::Point2i& start_pt,
                                        Direction dir,
                                        Boundary* boundary) const {

  boundary->region = region_id;

  // Create start segment.
  const int32_t* curr_position = &id_view_.at<int32_t>(start_pt.y, start_pt.x);
  Boundary::Segment segment;
  segment.start.pt = start_pt;
  segment.start.order = VertexOrder(curr_position);
  segment.points.push_back(start_pt);

  cv::Point2i curr_pt = start_pt + DirectionToVector(dir);
  curr_position += DirectionToOffset(dir);
  segment.points.push_back(curr_pt);

  // If this is a order 4 vertex the contour might pass through us twice.
  // In that case remember special termination position.
  const int32_t* termination_position = nullptr;
  if (segment.start.order == 4) {
    termination_position = curr_position;
  }

  Direction prev_dir = dir;

  // Trace until we hit the start_point again.
  // OR: In case of a vertex 4 start idx, until the next point is not the
  //     termination position.
  while (curr_pt != start_pt ||
         (termination_position &&
          curr_position + DirectionToOffset(
            NextDirection(curr_position, prev_dir, region_id)) != termination_position)) {
    // Start new segment if we reached a vertex.
    const int vertex_order = VertexOrder(curr_position);
    if (vertex_order > 1) {
      // Start new segment.
      segment.end.pt = curr_pt;
      DCHECK_EQ(segment.points.back(), curr_pt);

      boundary->segments.push_back(segment);

      segment = Boundary::Segment();
      segment.start.pt = curr_pt;
      segment.start.order = vertex_order;
      segment.points.push_back(curr_pt);
    } else {
      // Set left and right region id for this contour.
      SetSegmentRegions(curr_position, prev_dir, &segment);
      // Left should always point to inner region.
      CHECK_EQ(region_id, segment.left_region);
      // Right should never point to inner region.
      CHECK_NE(region_id, segment.right_region);
    }

    // Advance point.
    Direction next_dir = NextDirection(curr_position, prev_dir, region_id);
    curr_pt += DirectionToVector(next_dir);
    curr_position += DirectionToOffset(next_dir);
    segment.points.push_back(curr_pt);
    prev_dir = next_dir;
  }

  // Close segment.
  segment.end.pt = curr_pt;
  boundary->segments.push_back(segment);

  // Check for closed boundaries.
  if (boundary->segments.size() > 1) {
    DCHECK_EQ(boundary->segments[0].points[0], boundary->segments.back().points.back());
  }

  // Check for regions being set.
  for (const auto& segment : boundary->segments) {
    if (segment.points.size() > 3 &&
        !IsFrameRectangleSegment(segment)) {
      DCHECK_GE(segment.left_region, 0) << segment.ToString();
      DCHECK_GE(segment.right_region, 0) << segment.ToString();
    }
  }

  // As tracing starts at top left position of Rasterization, first vertex is
  // not necessarily a real vertex. Except for holes (one segment total), we
  // can merge first and last segment here.
  if (boundary->segments.size() > 1 &&
      boundary->segments[0].start.order < 2) {
    boundary->segments[0].start = boundary->segments.back().start;
    boundary->segments[0].points.insert(
      boundary->segments[0].points.begin(),
      boundary->segments.back().points.begin(),
      boundary->segments.back().points.end() - 1);  // Don't replicate middle point.

    // Remove last segment.
    boundary->segments.pop_back();

    // This boundary might not have regions assigned, but due to merge it is at least
    // 3 pixels long.
    CHECK_GE(boundary->segments[0].points.size(), 3);
    const cv::Point2i start = boundary->segments[0].points[0];
    const int32_t* curr_position = &id_view_.at<int32_t>(start.y, start.x);

    // Advance in direction.
    Direction dir = VectorToDirection(boundary->segments[0].points[1] - start);
    curr_position += DirectionToOffset(dir);
    DCHECK_LE(VertexOrder(curr_position), 2);
    SetSegmentRegions(curr_position, dir, &(boundary->segments[0]));
  }
}

BoundaryComputation::Direction BoundaryComputation::NextDirection(
    const int32_t* curr_position,
    Direction prev_dir,
    int region_id) const {
  // See Fig. 9 in Yuah-Tay's paper.
  switch (prev_dir) {
    case D_R:
      // a - c
      DCHECK_EQ(region_id, curr_position[DirectionToOffset(D_TL)]);
      DCHECK_NE(region_id, curr_position[DirectionToOffset(D_L)]);
      if (curr_position[DirectionToOffset(D_T)] != region_id) {
        return D_T;  // a
      } else if (curr_position[0] != region_id) {
        return D_R;  // b
      } else {
        return D_B;  // c
      }

    case D_T:
      DCHECK_EQ(region_id, curr_position[DirectionToOffset(D_L)]);
      DCHECK_NE(region_id, curr_position[0]);
      // d - f
      if (curr_position[DirectionToOffset(D_TL)] == region_id) {
        if (curr_position[DirectionToOffset(D_T)] == region_id) {
          return D_R;  // f
        } else {
          return D_T;  // a
        }
      } else {
        return D_L;  // e;
      }

     case D_L:
       // g - i
       DCHECK_EQ(region_id, curr_position[0]);
       DCHECK_NE(region_id, curr_position[DirectionToOffset(D_T)]);
       if (curr_position[DirectionToOffset(D_L)] == region_id) {
         if (curr_position[DirectionToOffset(D_TL)] != region_id) {
           return D_L;  // g;
         } else {
           return D_T;  // h;
         }
       } else {
         return D_B;  // i
       }

     case D_B:
      // j - l
      DCHECK_EQ(region_id, curr_position[DirectionToOffset(D_T)]);
      DCHECK_NE(region_id, curr_position[DirectionToOffset(D_TL)]);
      if (curr_position[0] == region_id) {
       if (curr_position[DirectionToOffset(D_L)] != region_id) {
         return D_B;   // j
       } else {
         return D_L;   // k
       }
      } else {
        return D_R;    // l 
      }

     default:
      LOG(FATAL) << "Unexpected direction for N4 trace!";
  }
}

int BoundaryComputation::VertexOrder(const int32_t* curr_position) const {
  // See Fig. 6-8 && page 316 of Yuah-Tay's paper.
  int curr = curr_position[0];
  int left = curr_position[DirectionToOffset(D_L)];
  int top = curr_position[DirectionToOffset(D_T)];
  int top_left = curr_position[DirectionToOffset(D_TL)];

  if (curr < 0) {
    // Incident to boundary (Fig. 8), bottom or right.
    if (left >= 0) {
      // Right case.
     return left != top_left ? 2 : 1;
    } else {
      // Bottom case.
     return top_left != top ? 2 : 1;
    }
  } else if (left < 0) {
    // Left boundary.
    return top != curr ? 2 : 1;
  } else if (top < 0) {
    // Top boundary.
    return left != curr ? 2 : 1;
  } else {
    // Within the frame. See Fig. 7 and 8.
    // Count id changes.
    const int id_changes = (int)(curr != left) +
                           (int)(left != top_left) +
                           (int)(top_left != top) +
                           (int)(top != curr);
    // If more than to id_changes we got a 3 or 4 vertex, otherwise we are just between
    // two regions, i.e. this is not a real vertex.
    return id_changes > 2 ? id_changes : 1;
  }
}

void BoundaryComputation::SetSegmentRegions(const int32_t* curr_position,
                                            Direction prev_dir,
                                            Boundary::Segment* segment) const {
  // See Fig. 9 in Yuah-Tay's paper.
  switch (prev_dir) {
    case D_R:
      segment->left_region = curr_position[DirectionToOffset(D_TL)];
      segment->right_region = curr_position[DirectionToOffset(D_L)];
      break;

    case D_T:
      segment->left_region = curr_position[DirectionToOffset(D_L)];
      segment->right_region = curr_position[0];
      break;

     case D_L:
       segment->left_region =  curr_position[0];
       segment->right_region = curr_position[DirectionToOffset(D_T)];
       break;

     case D_B:
       segment->left_region = curr_position[DirectionToOffset(D_T)];
       segment->right_region = curr_position[DirectionToOffset(D_TL)];
       break;

     default:
       LOG(FATAL) << "Unexpected direction for N4 trace.";
  }
}


bool BoundaryComputation::IsFrameRectanglePoint(const cv::Point2i& pt) const {
  return pt.x == 0 || pt.y == 0 || pt.x == frame_width_ || pt.y == frame_height_;
}

bool BoundaryComputation::IsFrameRectangleSegment(
    const Boundary::Segment& segment) const {
  for (const auto& pt : segment.points) {
    if (!IsFrameRectanglePoint(pt)) {
      return false;
    }
  }

  return true;
}

namespace {

string PointVectorString(const std::vector<cv::Point2i>& pt) {
  string result = base::StringPrintf("vec with %d points\n", pt.size());
  for (const auto& p : pt) {
    result += base::StringPrintf("> %d, %d \n", p.x, p.y);
  }
  return result;
}

}  //  namespace.

void BoundaryComputation::ComputeVectorization(
    const std::vector<std::unique_ptr<Boundary>>& boundaries,
    int min_segment_length,
    float max_error,
    SegmentationDesc* seg) {
  // Stores all polygon segments. Indexed via hashed boundary segments.
  std::vector<std::vector<cv::Point2i>> polygon_segments;
  polygon_segments.reserve(20 * boundaries.size());

  // Hashes each segment to a position in above polygon segments.
  BoundarySegmentIdxHash boundary_hash(boundaries.size() * 20,
                                       BoundarySegmentHasher(frame_width_));

  min_segment_length = std::max(3, min_segment_length);

  // Hash a cv::Point2i to its location in the VectorMesh.
  struct PointHasher {
    PointHasher(int frame_width) : frame_width_(frame_width) { }
    size_t operator()(const cv::Point2i& pt) const {
      return pt.y * frame_width_ + pt.x;   // Map to offset.
    }
    int frame_width_ = 0;
  };

  std::unordered_map<cv::Point2i, int, PointHasher> vector_mesh_map(
      boundaries.size() * 50, PointHasher(frame_width_));
  auto* vector_mesh = seg->mutable_vector_mesh();

  // Convert each Boundary into a polyline.
  for (const auto& boundary_ptr : boundaries) {
    // Polygon for this region contour.
    std::vector<cv::Point2i> polygon;
    polygon.reserve(boundary_ptr->Length());

    for (const auto segment : boundary_ptr->segments) {
      // Note for each segment we only store begin to end - 1 (to close the polygon
      // and avoid doubling vertices).

      // Determine if the segment is closed.
      const bool is_closed = segment.start.pt == segment.end.pt;

      // Determine if this segment can be collapsed into a start and end (only
      // store start in this case).
      const bool should_collapse =
        !is_closed && segment.points.size() < min_segment_length;

      if (should_collapse) {
        polygon.push_back(segment.points[0]);
        continue;
      }

      // Do we already have a polyline (corresponding one was computed).
      BoundarySegmentKey key(segment);
      auto pos = boundary_hash.find(key);
      if (pos == boundary_hash.end()) {
        // Create poly-line for this segment.
        std::vector<cv::Point2i> result;
        cv::approxPolyDP(segment.points, result, max_error, is_closed);
        if (is_closed) {
          result.push_back(result[0]);
        }

        DCHECK_GE(result.size(), 2);

        polygon.insert(polygon.end(), result.begin(), result.end() - 1);
        polygon_segments.push_back(result);
        boundary_hash[key] = polygon_segments.size() - 1;
      } else {
        // Insert points in opposite order (corresponding edge).
        const std::vector<cv::Point2i>& poly_seg = polygon_segments[pos->second];
        polygon.insert(polygon.end(), poly_seg.rbegin(), poly_seg.rend() - 1);
      }
    }

    // Close polygon.
    polygon.push_back(polygon[0]);

    DCHECK_GE(polygon.size(), 3);

    // Skip polygons with no interior. Could have been created due to some 
    // small discretization errors.
    if (polygon.size() == 3 && polygon[0] == polygon[2]) {
      continue;
    }

    // Add polygon to vectorization.
    DCHECK_GE(boundary_ptr->region, 0);
    Region2D* region = GetMutableRegion2DFromId(boundary_ptr->region, seg);
    DCHECK(region != nullptr);

    auto* poly = region->mutable_vectorization()->add_polygon();
    poly->set_hole(boundary_ptr->is_hole);
    for (const auto& pt : polygon) {
      // Already part of the VectorMesh?
      DCHECK_GE(pt.x, 0);
      DCHECK_LE(pt.x, frame_width_);
      DCHECK_GE(pt.y, 0);
      DCHECK_LE(pt.y, frame_height_);

      auto pos = vector_mesh_map.find(pt);
      if (pos != vector_mesh_map.end()) {
        poly->add_coord_idx(pos->second);
      } else {
        int idx = vector_mesh->coord_size();
        DCHECK_EQ(0, idx % 2);
        vector_mesh->add_coord(pt.x);
        vector_mesh->add_coord(pt.y);
        poly->add_coord_idx(idx);
        // Store in vector mesh for next lookup.
        vector_mesh_map[pt] = idx;
      }
    }
  }
}

BoundarySegmentKey::BoundarySegmentKey(const Boundary::Segment& segment) {
  DCHECK_GE(segment.points.size(), 3) << "Only edges with interior are hashable.";

  cv::Point2i start = segment.start.pt;
  cv::Point2i end = segment.end.pt;

  // Lexicographic ordering.
  if (start.x < end.x || (start.x == end.x && start.y < end.y)) {
    pt_1 = start;
    pt_2 = end;
    region_1 = segment.left_region;
    region_2 = segment.right_region;
  } else if (start == end) {
    pt_1 = pt_2 = start;
    region_1 = std::min(segment.left_region, segment.right_region);
    region_2 = std::max(segment.left_region, segment.right_region);
  } else {
    pt_1 = end;
    pt_2 = start;
    region_1 = segment.right_region;
    region_2 = segment.left_region;
  }
}


std::string BoundarySegmentKey::ToString() const {
  return base::StringPrintf("pt_1: %d, %d, pt_2: %d, %d, reg_1: %d, reg_2: %d",
                            pt_1.x, pt_1.y, pt_2.x, pt_2.y, region_1, region_2);
}

}  // namespace segmentation.
