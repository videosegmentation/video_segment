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


#include "segment_util/segmentation_util.h"
#include "base/base_impl.h"

#include <cstdio>

#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/core/core.hpp>

#ifdef _WIN32
#undef min
#undef max
#endif

namespace {

template <class T>
T* PtrOffset(T* t, int offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(t) + offset);
}

template <class T>
const T* PtrOffset(const T* t, int offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<uint8_t*>(t) + offset);
}

}

namespace segmentation {

namespace {

// Compare Region's by id.
struct Region2DComparator {
  bool operator()(const Region2D& lhs, const Region2D& rhs) const {
    return lhs.id() < rhs.id();
  }
};

// Compare RegionFeature's by id.
struct RegionFeaturesComparator {
  bool operator()(const RegionFeatures& lhs, const RegionFeatures& rhs) const {
    return lhs.id() < rhs.id();
  }
};

// Compare compound regions by id.
struct CompoundRegionComparator {
  bool operator()(const CompoundRegion& lhs, const CompoundRegion& rhs) const {
    return lhs.id() < rhs.id();
  }
};

}  // namepace

const Region2D& GetRegion2DFromId(int id,
                                  const SegmentationDesc& desc) {
  Region2D to_find;
  to_find.set_id(id);
  auto region = std::lower_bound(desc.region().begin(),
                                 desc.region().end(),
                                 to_find,
                                 Region2DComparator());
  DCHECK(region != desc.region().end());
  DCHECK_EQ(region->id(), id);

  return *region;
}

bool ContainsRegion2D(int id,
                      const SegmentationDesc& desc) {
  Region2D to_find;
  to_find.set_id(id);
  auto region = std::lower_bound(desc.region().begin(),
                                 desc.region().end(),
                                 to_find,
                                 Region2DComparator());
  return region != desc.region().end() && region->id() == id;
}

Region2D* GetMutableRegion2DFromId(int id,
                                   SegmentationDesc* desc) {
  Region2D to_find;
  to_find.set_id(id);
  auto region = std::lower_bound(desc->mutable_region()->begin(),
                                 desc->mutable_region()->end(),
                                 to_find,
                                 Region2DComparator());
  DCHECK(region != desc->mutable_region()->end());
  DCHECK_EQ(region->id(), id);

  return &(*region);
}

const CompoundRegion& GetCompoundRegionFromId(int id,
                                              const HierarchyLevel& hier_level) {
  CompoundRegion to_find;
  to_find.set_id(id);

  auto region = std::lower_bound(hier_level.region().begin(),
                                 hier_level.region().end(),
                                 to_find,
                                 CompoundRegionComparator());
  DCHECK(region != hier_level.region().end());
  DCHECK_EQ(region->id(), id);

  return *region;
}

CompoundRegion* GetMutableCompoundRegionFromId(int id,
                                               HierarchyLevel* hier_level) {
  CompoundRegion to_find;
  to_find.set_id(id);

  auto region = std::lower_bound(hier_level->mutable_region()->begin(),
                                 hier_level->mutable_region()->end(),
                                 to_find,
                                 CompoundRegionComparator());
  DCHECK(region != hier_level->mutable_region()->end());
  DCHECK_EQ(region->id(), id);

  return &(*region);
}

const RegionFeatures& GetRegionFeaturesFromId(int id, const SegmentationDesc& desc) {
  RegionFeatures to_find;
  to_find.set_id(id);
  auto features = std::lower_bound(desc.features().begin(),
                                   desc.features().end(),
                                   to_find,
                                   RegionFeaturesComparator());
  CHECK(features != desc.features().end());
  CHECK_EQ(features->id(), id) << "Not found: " << id;

  return *features;
}

int GetParentId(int region_id,
                int level,
                int query_level,
                const Hierarchy& seg_hier) {
  if (level == query_level) {
    return region_id;
  }

  DCHECK_GT(query_level, level);

  int parent_id;
  DCHECK_GE(seg_hier.size(), query_level);
  parent_id = GetCompoundRegionFromId(region_id, seg_hier.Get(level)).parent_id();

  if (query_level == level + 1) {
    return parent_id;
  } else {
    return GetParentId(parent_id, level + 1, query_level, seg_hier);
  }
}

void SortRegions2DById(SegmentationDesc* desc) {
  std::sort(desc->mutable_region()->begin(),
            desc->mutable_region()->end(),
            Region2DComparator());
}

void SortCompoundRegionsById(HierarchyLevel* hierarchy) {
  std::sort(hierarchy->mutable_region()->begin(),
            hierarchy->mutable_region()->end(),
            CompoundRegionComparator());
}

void GetParentMap(int level,
                  const SegmentationDesc& seg,
                  const Hierarchy& seg_hier,
                  ParentMap* parent_map) {
  if (level >= seg_hier.size()) {
    level = seg_hier.size() - 1;
    LOG(WARNING) << "Clamping requested level to " << level;
  }

  // Fill each region.
  const auto& regions = seg.region();
  for(const auto& region : seg.region()) {
    int parent_id = GetParentId(region.id(), 0, level, seg_hier);
    (*parent_map)[parent_id].push_back(&region);
  }
}

void GetChildrenIds(int region_id,
                    int level,
                    int query_level,
                    const Hierarchy& seg_hier,
                    vector<int>* children_ids) {
  CHECK_GT(level, query_level);
  const CompoundRegion& region = GetCompoundRegionFromId(region_id,
                                                         seg_hier.Get(level));

  if (query_level + 1 == level) {
    // Add all children and return.
    for (int child_id : region.child_id()) {
      children_ids->push_back(child_id);
    }
    return;
  } else {
    // Call recursively on children. Note: children are disjoint, to no unifiquation
    // is necessary.
    for (int child_id : region.child_id()) {
      GetChildrenIds(child_id,
                     level - 1,
                     query_level,
                     seg_hier,
                     children_ids);
    }
  }
}

bool GetShapeDescriptorFromShapeMoments(const vector<const ShapeMoments*>& moments,
                                        ShapeDescriptor* shape_desc) {
  // Compute mixed moments.
  float mixed_x = 0, mixed_y = 0, mixed_xx = 0, mixed_xy = 0, mixed_yy = 0;
  float area_sum = 0;

  // Aggregate over all passed moments.
  for (const auto& moment_ptr : moments) {
    const ShapeMoments& moment = *moment_ptr;
    const float area = moment.size();
    area_sum += area;

    mixed_x += moment.mean_x() * area;
    mixed_y += moment.mean_y() * area;
    mixed_xx += moment.moment_xx() * area;
    mixed_xy += moment.moment_xy() * area;
    mixed_yy += moment.moment_yy() * area;
  }

  CHECK_GT(area_sum, 0);

  // Normalize by inv_area_sum.
  const float inv_area_sum = 1.0f / area_sum;
  mixed_x *= inv_area_sum;
  mixed_y *= inv_area_sum;
  mixed_xx *= inv_area_sum;
  mixed_xy *= inv_area_sum;
  mixed_yy *= inv_area_sum;

  shape_desc->center = cv::Point2f(mixed_x, mixed_y);

  if (area_sum < 10) {
    return false;
  }

  // Compute variance matrix.
  const float var_xx = mixed_xx - mixed_x * mixed_x;
  const float var_xy = mixed_xy - mixed_x * mixed_y;
  const float var_yy = mixed_yy - mixed_y * mixed_y;

  // Compute eigenvectors of variance matrix.
  const float trace = var_xx + var_yy;
  const float det = var_xx * var_yy - var_xy * var_xy;

  const float discriminant = 0.25 * trace * trace - det;
  CHECK_GE(discriminant, 0);
  const float sqrt_disc = sqrt(discriminant);
  const float e_1 = trace * 0.5 - sqrt_disc;
  const float e_2 = trace * 0.5 + sqrt_disc;

  // Eigenvalues e_1 and e_2 are variance along major/minor axis.
  if (std::min(fabs(e_1), fabs(e_2)) < 1) {  // Extend along one axis is less
                                             // than one pixel.
    return false;
  }

  cv::Point2f ev_1(1.0f, 0.0f);
  cv::Point2f ev_2(0.0f, 1.0f);
  if (fabs(det) > 1e-6f) {
    ev_1 = cv::Point2f(e_1 - var_yy, var_xy);
    ev_2 = cv::Point2f(e_2 - var_yy, var_xy);
    ev_1 *= 1.0f / (cv::norm(ev_1));
    ev_2 *= 1.0f / (cv::norm(ev_2));
  }

  float e_1_sigma = sqrt(fabs(e_1));
  float e_2_sigma = sqrt(fabs(e_2));

  if (e_1_sigma < e_2_sigma) {
    std::swap(e_1_sigma, e_2_sigma);
    std::swap(ev_1, ev_2);
  }

  shape_desc->center = cv::Point2f(mixed_x, mixed_y);
  shape_desc->mag_major = e_1_sigma;
  shape_desc->mag_minor = e_2_sigma;
  shape_desc->dir_major = ev_1;
  shape_desc->dir_minor = ev_2;

  return true;
}

bool GetShapeDescriptorFromShapeMoment(const ShapeMoments& moment,
                                       ShapeDescriptor* shape_desc) {
  return GetShapeDescriptorFromShapeMoments({&moment}, shape_desc);
}

bool GetShapeDescriptorFromRegion(const Region2D& r,
                                  ShapeDescriptor* shape_desc) {
  return GetShapeDescriptorFromShapeMoments({&r.shape_moments()}, shape_desc);
}

bool GetShapeDescriptorFromRegions(const vector<const Region2D*>& regions,
                                   ShapeDescriptor* shape_desc) {
  vector<const ShapeMoments*> moments;
  moments.reserve(regions.size());
  for (const auto& region_ptr : regions) {
    moments.push_back(&region_ptr->shape_moments());
  }

  return GetShapeDescriptorFromShapeMoments(moments, shape_desc);
}


void RenderShapeDescriptor(const vector<int>& overseg_region_ids,
                           const SegmentationDesc& desc,
                           cv::Mat* output,
                           const string* label) {
  CHECK_NOTNULL(output);
  vector<const Region2D*> regions;
  // Get regions from id's.
  for (int overseg_region_id : overseg_region_ids) {
    regions.push_back(&GetRegion2DFromId(overseg_region_id, desc));
  }

  ShapeDescriptor shape_desc;
  if (!GetShapeDescriptorFromRegions(regions, &shape_desc)) {
    return;
  }

  cv::Scalar solid_white(255, 255, 255, 255);

  // Draw 6x6 cross around the center.
  if (label == nullptr) {
    cv::Point left = shape_desc.center - shape_desc.dir_major * 3;
    cv::Point right = shape_desc.center + shape_desc.dir_major * 3;
    cv::Point top = shape_desc.center - shape_desc.dir_minor * 3;
    cv::Point bottom = shape_desc.center + shape_desc.dir_minor * 3;

    cv::line(*output, left, right, solid_white);
    cv::line(*output, top, bottom, solid_white);
  } else {
    cv::putText(*output, *label, shape_desc.center, cv::FONT_HERSHEY_PLAIN, 0.8,
                solid_white);
  }

  // Determine angle.
  const float angle = atan2(shape_desc.dir_major.y, shape_desc.dir_major.x);
  cv::ellipse(*output,
            shape_desc.center,
            cv::Size(shape_desc.mag_major, shape_desc.mag_minor),
            angle / M_PI * 180.0f,
            0,
            360,
            solid_white);
}


void MergeRasterization(const Rasterization& lhs,
                        const Rasterization& rhs,
                        Rasterization* merged) {
  CHECK_NOTNULL(merged);

  auto lhs_scan = lhs.scan_inter().begin();
  auto rhs_scan = rhs.scan_inter().begin();

  auto lhs_end = lhs.scan_inter().end();
  auto rhs_end = rhs.scan_inter().end();

  vector<int> interval_offsets;

  // Traverse scanlines in lockstep.
  while (lhs_scan != lhs_end || rhs_scan != rhs_end) {
    const int lhs_y = (lhs_scan == lhs_end ? 1 << 30 : lhs_scan->y());
    const int rhs_y = (rhs_scan == rhs_end ? 1 << 30 : rhs_scan->y());

    if (lhs_y < rhs_y) {
      merged->add_scan_inter()->CopyFrom(*lhs_scan++);
    } else if (rhs_y < lhs_y) {
      merged->add_scan_inter()->CopyFrom(*rhs_scan++);
    } else {
      // y-coords are equal.
      DCHECK_EQ(lhs_y, rhs_y);

      // Collect all intervals in this scanline order by lhs x coord.
      interval_offsets.clear();
      // Set to true if corresponding iterator points still to same y scanline.
      bool left_cond, right_cond;
      while ( (left_cond = (lhs_scan != lhs_end && lhs_scan->y() == lhs_y)) ||
              (right_cond = (rhs_scan != rhs_end && rhs_scan->y() == rhs_y)) ) {
        // Smaller x coordinate comes first.
        const int lhs_x = left_cond ? lhs_scan->left_x()
                                    : std::numeric_limits<int>::max();
        const int rhs_x = right_cond ? rhs_scan->left_x()
                                     : std::numeric_limits<int>::max();
        if (lhs_x < rhs_x) {
          interval_offsets.push_back(lhs_scan->left_x());
          interval_offsets.push_back(lhs_scan->right_x());
          ++lhs_scan;
        } else {
          interval_offsets.push_back(rhs_scan->left_x());
          interval_offsets.push_back(rhs_scan->right_x());
          ++rhs_scan;
        }
      }

      DCHECK(interval_offsets.size() % 2 == 0);

      // Create actual ScanIntervals, while merging adjacent offsets.
      // k points to beginning of current interval,
      // l to beginning of to be inserted interval.
      int k = 0, l = 0, sz_k = interval_offsets.size();
      while (k < sz_k) {
        // Last interval -> insert and break.
        if (k + 2 == sz_k) {
          ScanInterval* scan_inter = merged->add_scan_inter();
          scan_inter->set_y(lhs_y);
          scan_inter->set_left_x(interval_offsets[l]);
          scan_inter->set_right_x(interval_offsets[k + 1]);
          break;
        } else if (interval_offsets[k + 2] - 1 == interval_offsets[k + 1]) {
          // Connected -> skip.
          k += 2;
        } else {
          // Not, connected: Reset and add.
          ScanInterval* scan_inter = merged->add_scan_inter();
          scan_inter->set_y(lhs_y);
          scan_inter->set_left_x(interval_offsets[l]);
          scan_inter->set_right_x(interval_offsets[k + 1]);
          k += 2;
          l = k;
        }
      }
    }
  }
}

void MergeRasterizations(const vector<const Rasterization*>& rasters,
                         Rasterization* result) {
  if (rasters.empty()) {
    return;
  }

  std::unique_ptr<Rasterization> merged_raster(new Rasterization());
  std::unique_ptr<Rasterization> prev_raster(new Rasterization());
  prev_raster->MergeFrom(*rasters[0]);

  for (int i = 1; i < rasters.size(); ++i) {
    merged_raster->Clear();
    MergeRasterization(*rasters[i], *prev_raster, merged_raster.get());
    prev_raster.swap(merged_raster);
  }

  result->Clear();
  result->MergeFrom(*prev_raster);
}

void GetCompoundRegionRasterizations(
    const ParentMap& parent_map,
    unordered_map<int, Rasterization>* parent_rasterizations) {
  for (const auto& map_entry : parent_map) {
    const int region_id = map_entry.first;
    vector<const Rasterization*> rasters;
    rasters.reserve(map_entry.second.size());
    for (const auto& child_ptr : map_entry.second) {
      rasters.push_back(&(child_ptr->raster()));
    }
    (*parent_rasterizations)[region_id] = Rasterization();
    MergeRasterizations(rasters, &(*parent_rasterizations)[region_id]);
  }
}

void MergeRasterization3D(const Rasterization3D& lhs,
                          const Rasterization3D& rhs,
                          Rasterization3D* merged) {
  auto lhs_iter = lhs.begin();
  auto rhs_iter = rhs.begin();

  while (lhs_iter != lhs.end() || rhs_iter != rhs.end()) {
    // Traverse rasterizations ordered by frame.
    const int lhs_frame = (lhs_iter == lhs.end() ? std::numeric_limits<int>::max()
                                                 : lhs_iter->first);
    const int rhs_frame = (rhs_iter == rhs.end() ? std::numeric_limits<int>::max()
                                                 : rhs_iter->first);

    if (lhs_frame < rhs_frame) {
      shared_ptr<Rasterization> deep_copy(new Rasterization());
      deep_copy->CopyFrom(*lhs_iter->second);
      merged->push_back(std::make_pair(lhs_frame, deep_copy));
      ++lhs_iter;
    } else if (rhs_frame < lhs_frame) {
      shared_ptr<Rasterization> deep_copy(new Rasterization());
      deep_copy->CopyFrom(*rhs_iter->second);
      merged->push_back(std::make_pair(rhs_frame, deep_copy));
      ++rhs_iter;
    } else {
      // Frames are equal.
      DCHECK_EQ(lhs_frame, rhs_frame);
      shared_ptr<Rasterization> merged_raster(new Rasterization());
      MergeRasterization(*lhs_iter->second,
                         *rhs_iter->second,
                         merged_raster.get());
      merged->push_back(std::make_pair(lhs_frame, merged_raster));
      ++lhs_iter;
      ++rhs_iter;
    }
  }
}

int RasterizationArea(const Rasterization& raster) {
  int area = 0;
  for (const auto& scan_inter : raster.scan_inter()) {
    area += scan_inter.right_x() - scan_inter.left_x() + 1;
  }
  return area;
}

void ShapeMomentsFromRasterization(const Rasterization& raster,
                                   ShapeMoments* moments) {
  CHECK_NOTNULL(moments);

  float mean_x = 0;
  float mean_y = 0;
  float moment_xx = 0;
  float moment_yy = 0;
  float moment_xy = 0;
  float area_sum = 0;

  for(const auto& scan_inter : raster.scan_inter()) {
    // Add to shape descriptor.
    const float m = scan_inter.left_x();
    const float n = scan_inter.right_x();
    const float curr_y = scan_inter.y();
    const float len = (n - m + 1);
    area_sum += len;
    const float center_x = (n + m) * 0.5;

    const float sum_x = center_x * len;
    const float sum_y = curr_y * len;

    mean_x += sum_x;
    mean_y += sum_y;
    moment_xy += curr_y * sum_x;
    moment_yy += curr_y * sum_y;

    // sum_m^n x_i^2  -->
    // 1/6 * Factor[2 n^3 + 3 n^2 + n  - 2 (m - 1)^3 - 3*(m - 1)^2 - (m - 1)]
    // = 1/6 * (1 + n - m) (-m + 2 m^2 + n + 2 m n + 2 n^2)
    moment_xx += len * (-m + 2 * m * m + n + 2 * m * n + 2 * n * n) / 6.0f;
  }

  const float inv_area = 1.0f / area_sum;
  moments->set_size(area_sum);
  moments->set_mean_x(mean_x * inv_area);
  moments->set_mean_y(mean_y * inv_area);
  moments->set_moment_xx(moment_xx * inv_area);
  moments->set_moment_xy(moment_xy * inv_area);
  moments->set_moment_yy(moment_yy * inv_area);
}

void SetShapeMomentsFromRegion(Region2D* r) {
  ShapeMomentsFromRasterization(r->raster(), r->mutable_shape_moments());
}

void ConstrainHierarchyToFrameInterval(int lhs,
                                       int rhs,
                                       const HierarchyLevel& input_hierachy,
                                       HierarchyLevel* constraint_hierarchy) {
  CHECK_NOTNULL(constraint_hierarchy);
  // Stores regions that fall within the interval.
  unordered_set<int> outside_regions;
  // First pass: Determine regions to be removed.
  for (const auto& region : input_hierachy.region()) {
    if (region.start_frame() > rhs || region.end_frame() < lhs) {
      outside_regions.insert(region.id());
    }
  }

  // No regions have to be removed: Simply copy and return.
  if (outside_regions.size() == 0) {
    *constraint_hierarchy = input_hierachy;
    return;
  }

  // Copy only regions within interval.
  for (const auto& region : input_hierachy.region()) {
    if (outside_regions.count(region.id()) > 0) {
      continue;
    }

    CompoundRegion* proto_region = constraint_hierarchy->add_region();
    // Copy.
    *proto_region = region;

    // Reset neighbors.
    proto_region->clear_neighbor_id();

    for (int neighbor : region.neighbor_id()) {
      if (outside_regions.count(neighbor) == 0) {
        proto_region->add_neighbor_id(neighbor);
      }
    }
  }
}


void SegmentationDescToIdImage(int level,
                               const SegmentationDesc& seg,
                               const Hierarchy* seg_hier,
                               cv::Mat* id_image) {
  if (level > 0) {
    CHECK(seg_hier) << "Hierarchy level requested but seg_hier not set.";

    if (level > seg_hier->size()) {
      level = seg_hier->size();
      LOG(ERROR) << "Clamping requested level to " << level;
    }
  }

  // Fill each region with it's id.
  for (const auto& region : seg.region()) {
    // Get id.
    int region_id = region.id();
    if (level != 0) {
      region_id = GetParentId(region_id, 0, level, *seg_hier);
    }

    // Render scanline with region_id.
    for (const auto& s : region.raster().scan_inter()) {
      int32_t* output_ptr = id_image->ptr<int32_t>(s.y()) + s.left_x();
      for (int j = 0, len = s.right_x() - s.left_x() + 1; j < len; ++j) {
        output_ptr[j] = region_id;
      }
    }
  }
}

int GetOversegmentedRegionIdFromPoint(int x, int y, const SegmentationDesc& seg) {
  for (const auto& region : seg.region()) {
    auto s = LocateScanLine(y, region.raster());
    while (s != region.raster().scan_inter().end() &&
           s->y() == y) {
      if (x >= s->left_x() && x <= s->right_x()) {
        // Get my id and return.
        return region.id();
      }
      ++s;
    }
  }

  return -1;
}

namespace {

// Truncates passed hierarchy to specified number of levels. Resets parent ids of highest 
// remaining level.
void TruncateHierarchy(int levels, Hierarchy* hierarchy) {
  CHECK_GT(levels, 0) << "At least one level needs to be present.";

  if (hierarchy->size() <= levels) {
    return;
  }

  while (hierarchy->size() > levels) {
    hierarchy->RemoveLast();
  }

  for (auto& region : *hierarchy->Mutable(hierarchy->size() - 1)->mutable_region()) {
    region.set_parent_id(-1);   // Erase previous parents if necessary.
  }
}

void MergeCompoundRegion(const CompoundRegion& region_1,
                         const CompoundRegion& region_2,
                         CompoundRegion* merged_region) {
  CHECK_NOTNULL(merged_region);
  CHECK_EQ(region_1.id(), region_2.id());
  CHECK_EQ(region_1.parent_id(), region_2.parent_id());

  merged_region->set_id(region_1.id());
  merged_region->set_size(region_1.size() + region_2.size());
  merged_region->set_parent_id(region_1.parent_id());

  // Merge children and neighbors.
  std::set_union(region_1.neighbor_id().begin(),
                 region_1.neighbor_id().end(),
                 region_2.neighbor_id().begin(),
                 region_2.neighbor_id().end(),
                 google::protobuf::RepeatedFieldBackInserter(
                     merged_region->mutable_neighbor_id()));

  std::set_union(region_1.child_id().begin(),
                 region_1.child_id().end(),
                 region_2.child_id().begin(),
                 region_2.child_id().end(),
                 google::protobuf::RepeatedFieldBackInserter(
                     merged_region->mutable_child_id()));

  merged_region->set_start_frame(std::min(region_1.start_frame(),
                                          region_2.start_frame()));
  merged_region->set_end_frame(std::max(region_1.end_frame(),
                                        region_2.end_frame()));
}

// Merges regions across passed hierarchy levels.
void MergeHierarchyLevel(const HierarchyLevel& level_1,
                         const HierarchyLevel& level_2,
                         HierarchyLevel* merged_level) {
  CHECK_NOTNULL(merged_level);

  // Merge compound regions while preserving ordering by id.
  auto region_1_ptr = level_1.region().begin();
  auto region_2_ptr = level_2.region().begin();
  // Traverse regions ordered by id.
  while (region_1_ptr != level_1.region().end() &&
         region_2_ptr != level_2.region().end()) {
    if (region_1_ptr->id() < region_2_ptr->id()) {
      merged_level->mutable_region()->Add()->CopyFrom(*region_1_ptr++);
    } else if (region_2_ptr->id() < region_1_ptr->id()) {
      merged_level->mutable_region()->Add()->CopyFrom(*region_2_ptr++);
    } else {
      // Region id's are equal. Merge compound regions.
      CompoundRegion* merged_region = merged_level->mutable_region()->Add();
      MergeCompoundRegion(*region_1_ptr, *region_2_ptr, merged_region);
      ++region_1_ptr;
      ++region_2_ptr;
    }
  }

  // Finish remaining regions.
  while (region_1_ptr != level_1.region().end()) {
    merged_level->mutable_region()->Add()->CopyFrom(*region_1_ptr++);
  }

  while (region_2_ptr != level_2.region().end()) {
    merged_level->mutable_region()->Add()->CopyFrom(*region_2_ptr++);
  }
}

}  // namespace.

void BuildGlobalHierarchy(const Hierarchy& chunk_hierarchy,
                          int chunk_frame_start,
                          Hierarchy* global_hierarchy) {
  CHECK_NOTNULL(global_hierarchy);

  // On first call, copy and return.
  if (global_hierarchy->size() == 0) {
    global_hierarchy->MergeFrom(chunk_hierarchy);
    return;
  }

  // Levels need to be compatible.
  if (global_hierarchy->size() > chunk_hierarchy.size()) {
    TruncateHierarchy(chunk_hierarchy.size(), global_hierarchy);
  }

  Hierarchy merged;
  for (int level = 0; level < global_hierarchy->size(); ++level) {
    const HierarchyLevel& level_1 = global_hierarchy->Get(level);
    // Local copy.
    HierarchyLevel level_2 = chunk_hierarchy.Get(level);

    bool clear_parent = false;
    // Last level, might have to adjust members in chunk hierarchy.
    if (level + 1 == global_hierarchy->size() &&
        global_hierarchy->size() < chunk_hierarchy.size()) {
      clear_parent = true;
    }

    // Offset level_2 by frame_start.
    for (auto& region : *level_2.mutable_region()) {
      region.set_start_frame(region.start_frame() + chunk_frame_start);
      region.set_end_frame(region.end_frame() + chunk_frame_start);
      if (clear_parent) {
        region.set_parent_id(-1);
      }
    }

    // Merge Levels.
    HierarchyLevel merged_level;
    MergeHierarchyLevel(level_1, level_2, &merged_level);
    merged.Add()->MergeFrom(merged_level);
  }

  global_hierarchy->Clear();
  global_hierarchy->MergeFrom(merged);
}

bool VerifyGlobalHierarchy(const Hierarchy& hierarchy) {
  // Check that neighbors as well as parents and children are in a mutal relationship.
  LOG(INFO) << "Verifying global hierarchy.";
  int hier_levels = hierarchy.size();
  for (int level = 0; level < hier_levels; ++level) {
    const HierarchyLevel& curr_level = hierarchy.Get(level);
    for (const auto& region : curr_level.region()) {
      // Check neighbors.
      for (int neighbor_id : region.neighbor_id()) {
        const CompoundRegion& neighbor = GetCompoundRegionFromId(neighbor_id,
                                                                 curr_level);
        auto insert_pos = std::lower_bound(neighbor.neighbor_id().begin(),
                                           neighbor.neighbor_id().end(),
                                           region.id());
        if (insert_pos == neighbor.neighbor_id().end() ||
            *insert_pos != region.id()) {
          LOG(ERROR) << "Mutual neighbor error for region " << region.id()
                     << " and neighbor " << neighbor_id;
          return false;
        }
      }

      // Check parents.
      if (level + 1 < hier_levels) {
        const HierarchyLevel& next_level = hierarchy.Get(level + 1);
        if (region.parent_id() < 0) {
          LOG(ERROR) << "Region " << region.id() << " has no parent, but "
                        " is expected to have one.";
          return false;
        }

        const CompoundRegion& parent = GetCompoundRegionFromId(region.parent_id(),
                                                               next_level);
        auto insert_pos = std::lower_bound(parent.child_id().begin(),
                                           parent.child_id().end(),
                                           region.id());
        if (insert_pos == parent.child_id().end() ||
            *insert_pos != region.id()) {
          LOG(ERROR) << "Mutual parent/child error for region " << region.id()
                     << " and parent " << parent.id();
          return false;
        }
      }

      // Check children.
      if (level > 0) {
        const HierarchyLevel& prev_level = hierarchy.Get(level - 1);
        int aggregated_size = 0;
        int aggregated_start = std::numeric_limits<int>::max();
        int aggregated_end = std::numeric_limits<int>::min();

        for (int child_id : region.child_id()) {
          const CompoundRegion& child = GetCompoundRegionFromId(child_id,
                                                                prev_level);
           if (child.parent_id() != region.id()) {
             LOG(ERROR) << "Mutual child parent error for parent region "
                        << region.id() << " and child " << child_id;
            return false;
          }

          aggregated_size += child.size();
          aggregated_start = std::min(aggregated_start, child.start_frame());
          aggregated_end = std::max(aggregated_end, child.end_frame());
        }
        if (aggregated_size != region.size()) {
          LOG(ERROR) << "Child region size does not sum up to region size "
                     << " for region " << region.id();
          return false;
        }

        if (aggregated_start != region.start_frame() ||
            aggregated_end != region.end_frame()) {
          LOG(ERROR) << "Aggregated start and end over child regions is "
                     << "incompatible.";
          return false;
        }
      }
    }  // end regions.
  } // end levels.
  return true;
}

namespace {
// Returns true if ScanIntervals are neighbors w.r.t. N8 connectedness.
bool ScanIntervalsNeighbored(const ScanInterval& lhs,
                             const ScanInterval& rhs,
                             Connectedness connect) {
  switch (connect) {
    case N8_CONNECT:
      return abs(lhs.y() - rhs.y()) <= 1 &&
          std::max(lhs.left_x(), rhs.left_x()) -
          std::min(lhs.right_x(), rhs.right_x()) <= 1;
     case N4_CONNECT:
      return abs(lhs.y() - rhs.y()) <= 1 &&
          std::max(lhs.left_x(), rhs.left_x()) <=
          std::min(lhs.right_x(), rhs.right_x());
  }
}
}  // namespace.

int ConnectedComponents(const Rasterization& raster,
                        Connectedness connect,
                        vector<Rasterization>* components) {
  const int scan_inter_size = raster.scan_inter_size();

  // Compute disjoint sets.
  // Underlying storage for disjoint set.
  // Original keys.
  vector<int> ranks(scan_inter_size);
  // Representatives.
  vector<int> parents(scan_inter_size);

  // Actual original elems (preserve for later iteration).
  vector<int> elements(scan_inter_size);
  boost::disjoint_sets<int*, int*> classes(&ranks[0], &parents[0]);

  int last_change_idx = -1;  // Index of scanline where new row was encountered the
                             // last time.
  int last_y = -2;           // y coordinate for previous processed row, not scanline!
  int test_idx = 0;          // Index where to start neighboring tests from.

  // One pass merge algorithm.
  for (int i = 0; i < scan_inter_size; ++i) {
    // Place every scanline into its own set.
    classes.make_set(i);
    elements[i] = i;  // Init for later iteration.

    const ScanInterval& curr_scan = raster.scan_inter(i);
    if (curr_scan.y() != last_y) {
      // If neighboring update test_idx.
      if (last_y + 1 == curr_scan.y()) {
        test_idx = last_change_idx;         // Start at first index or prev. row.
      } else {
        test_idx = i;                       // 2 rows apart, start here.
      }

      last_y = curr_scan.y();
      last_change_idx = i;
    }

    for (int k = test_idx; k < i; ++k) {
      if (ScanIntervalsNeighbored(curr_scan, raster.scan_inter(k), connect)) {
        classes.union_set(i, k);
      }
    }
  }

  int num_components = classes.count_sets(elements.begin(), elements.end());
  if (num_components == 1) {
    if (components != nullptr) {
      components->push_back(raster);
    }
    return 1;
  }

  if (components != nullptr) {
    // Compile component rasterizations.
    // Maps representative to Rasterization.
    components->reserve(num_components);
    unordered_map<int, Rasterization*> rep_to_component_map;
    for (int i = 0; i < scan_inter_size; ++i) {
      const ScanInterval& curr_scan = raster.scan_inter(i);
      int rep = classes.find_set(i);
      auto rep_iter = rep_to_component_map.find(rep);
      if (rep_iter == rep_to_component_map.end()) {
       components->push_back(Rasterization());
       components->back().add_scan_inter()->CopyFrom(curr_scan);
       rep_to_component_map[rep] = &components->back();
     } else {
       rep_iter->second->add_scan_inter()->CopyFrom(curr_scan);
     }
    }
    CHECK_EQ(num_components, components->size());
  }

  return num_components;
}

namespace {

// Represents an edge to be rasterized. Beginning y is given by insert position
// in corresponding edge_list.
struct EdgeEntry {
  // Maximum y-coordinate for edge
  float curr_x = 0;
  float y_max = 0;
  float dx = 0;
  bool is_left_edge = true;
  void Advance() { curr_x += dx; }

  bool operator<(const EdgeEntry& rhs) const {
    static constexpr float eps = 1e-3f;
    // Ordered by curr_x.
    if (curr_x < rhs.curr_x - eps) {
      return true;
    } else if (curr_x > rhs.curr_x + eps) {
      return false;
    }

    // X - positions are equal.
    // Left edge is always smaller.
    if (is_left_edge && !rhs.is_left_edge) {
      return true;
    } else if (rhs.is_left_edge && !is_left_edge) {
      return false;
    }

    // Two left edges that are incident. Decides which one will stay left (the one
    // with smaller dx).
    return dx < rhs.dx;
  }
};

}  // namespace.

void RasterVectorization(const Vectorization& vec,
                         const VectorMesh& mesh,
                         int frame_height,
                         Rasterization* raster) {
  DCHECK_NOTNULL(raster);
  raster->Clear();

  // Create an edge entry for each polyon line.
  // Stores for each y-location edges that start at that scanline.
  vector<vector<EdgeEntry>> edge_list(frame_height + 1);

  int start_y = frame_height;
  int end_y = 0;

  // Erased hole regions might exist.
  if (vec.polygon_size() == 0) {
    return;
  }

  for (const auto& poly : vec.polygon()) {
    CHECK_GT(poly.coord_idx_size(), 0);
    for (int c = 1; c < poly.coord_idx_size(); ++c) {
      const int idx_1 = poly.coord_idx(c - 1);
      const int idx_2 = poly.coord_idx(c);
      cv::Point2f p1(mesh.coord(idx_1), mesh.coord(idx_1 + 1));
      cv::Point2f p2(mesh.coord(idx_2), mesh.coord(idx_2 + 1));

      // Ignore horizontal edges.
      if (std::abs(p1.y - p2.y) < 1e-3f) {
        continue;
      }

      EdgeEntry entry;
      // Order points, such that smallest y comes first.
      if (p2.y < p1.y) {
        std::swap(p1, p2);
        entry.is_left_edge = false;
      }

      // Global start and end.
      start_y = std::min<int>(std::floor(p1.y), start_y);
      end_y = std::max<int>(std::ceil(p2.y), end_y);

      entry.curr_x = p1.x;
      entry.y_max = p2.y;
      entry.dx = (p2.x - p1.x) / (p2.y - p1.y);   // Denom != 0 due to above test.
      DCHECK_GE(p1.y, 0);
      DCHECK_LE(p1.y, frame_height);
      edge_list[p1.y].push_back(entry);
    }
  }

  if (start_y > end_y) {
    return;
  }

  // Rasterize.
  std::vector<EdgeEntry> aet;   // active edge list.
  for (int y = start_y; y <= end_y; ++y) {
    // Add new edges at current y.
    if (!edge_list[y].empty()) {
      aet.insert(aet.end(), edge_list[y].begin(), edge_list[y].end());
    }

    // Remove edges that reached maximum y.
    for (auto pos = aet.begin(); pos != aet.end(); ) {
      if (pos->y_max < y + 1) {
        aet.erase(pos);
      } else {
        ++pos;
      }
    }

    std::sort(aet.begin(), aet.end());
    DCHECK_EQ(0, aet.size() % 2) << "Parity failure.";

    for (int k = 0; k < aet.size(); k += 2) {
      int x_start = std::ceil(aet[k].curr_x - 1e-6f);
      float frac_x = aet[k + 1].curr_x;
      int x_end = std::floor(frac_x);

      // Right border is always non-inclusive.
      if (std::abs(frac_x - x_end) < 1e-6f) {
        --x_end;
      }

      ScanInterval* scan = raster->add_scan_inter();
      scan->set_y(y);
      scan->set_left_x(x_start);
      scan->set_right_x(x_end);
    }

    for (auto& entry : aet) {
      entry.Advance();
    }
  }
}

void ReplaceRasterizationFromVectorization(SegmentationDesc* desc) {
  DCHECK(desc->has_vector_mesh());
  for (auto& region : *desc->mutable_region()) {
    RasterVectorization(region.vectorization(),
                        desc->vector_mesh(),
                        desc->frame_height(),
                        region.mutable_raster());
  }
}

void ScaleVectorization(int width, int height, SegmentationDesc* desc) {
  CHECK(desc->has_vector_mesh());

  int parity = 0;

  const float scale_x = width * (1.0f / desc->frame_width());
  const float scale_y = height * (1.0f / desc->frame_height());

  desc->set_frame_width(width);
  desc->set_frame_height(height);

  for (float& coord : *desc->mutable_vector_mesh()->mutable_coord()) {
    if (parity % 2 == 0) {
      coord = std::min<float>(width, coord * scale_x);    // x coord.
    } else {
      coord = std::min<float>(height, coord * scale_y);   // y coord.
    }
    ++parity;
  }
}

void RemoveRasterization(SegmentationDesc* desc) {
  CHECK(desc->has_vector_mesh());
  for (auto& region : *desc->mutable_region()) {
    region.clear_raster();
  }
  desc->set_rasterization_removed(true);
}

}  // namespace segmentation.
