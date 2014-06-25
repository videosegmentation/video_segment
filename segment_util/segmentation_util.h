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

#ifndef VIDEO_SEGMENT_SEGMENT_UTIL_SEGMENTATION_UTIL_H__
#define VIDEO_SEGMENT_SEGMENT_UTIL_SEGMENTATION_UTIL_H__

#include "base/base.h"

#include <google/protobuf/repeated_field.h>
#include <opencv2/core/core.hpp>

#include "segment_util/segmentation.pb.h"

namespace cv {
  class Mat;
}

namespace segmentation {

typedef SegmentationDesc::Region2D Region2D;
typedef SegmentationDesc::CompoundRegion CompoundRegion;
typedef SegmentationDesc::HierarchyLevel HierarchyLevel;
typedef google::protobuf::RepeatedPtrField<HierarchyLevel> Hierarchy;
typedef SegmentationDesc::Rasterization Rasterization;
typedef SegmentationDesc::Rasterization::ScanInterval ScanInterval;
typedef SegmentationDesc::ShapeMoments ShapeMoments;
typedef SegmentationDesc::VectorMesh VectorMesh;
typedef SegmentationDesc::Vectorization Vectorization;

// Common usage for all functions:
// The level of the hierarchy is passed in hierarchy_level.
// = 0  :  denotes the over-segmentation.
// >= 1 : denotes a hierarchical level
//
// Functions will threshold the passed level to max. level present in the hierarchy.
//
// The pixel level segmentation in SegmentationDesc and the actual hierarchy can
// be separated, as the hierarchy is saved only ONCE for the whole video volume.


//// Accessor functions for Region2D and CompoundRegion.
//////////////////////////////////////////////////////////

// Finds region with id in proto buffer desc. Fails in debug if not found.
const Region2D& GetRegion2DFromId(int id,
                                  const SegmentationDesc& desc);

// Returns true, if region with specified id exists.
bool ContainsRegion2D(int id,
                      const SegmentationDesc& desc);

// Returns mutable region with id in proto buffer desc. Fails in debug if not found.
Region2D* GetMutableRegion2DFromId(int id,
                                   SegmentationDesc* desc);

// Same as above for compound regions. Returns region with id at passes hierarchy level.
const CompoundRegion& GetCompoundRegionFromId(int id,
                                              const HierarchyLevel& hierarchy);

// Mutable version of above function.
CompoundRegion* GetMutableCompoundRegionFromId(int id,
                                               HierarchyLevel* hierarchy);

// Locate RegionFeature based on corresponding region's id. Fails with CHECK if not found.
const RegionFeatures& GetRegionFeaturesFromId(int id, const SegmentationDesc& desc);


//// Hierarchical traversal functions.
//////////////////////////////////////

// For a specific region with id region_id at hierarchy level "level", returns parent id 
// at level query_level. Note: level <= query_level (checked in debug mode).
int GetParentId(int region_id,
                int level,
                int query_level,
                const Hierarchy& seg_hier);

// Sorts region's by id (useful if id's are altered, e.g. constraints are propagated).
void SortRegions2DById(SegmentationDesc* desc);
void SortCompoundRegionsById(HierarchyLevel* hierarchy);

// Obtains map, that maps a parent_id to list of over-segmented Regions for a specific
// segmentation frame seg.
// Specifically each region in seg will be inserted into the vector
// specified by its parent's id at hierarchy level level.
// If requested level does not exist in seg_hier, level is clamped to highest level
// that exists in hierarchy.
typedef std::unordered_map<int, std::vector<const Region2D*>> ParentMap;
void GetParentMap(int level,
                  const SegmentationDesc& seg,
                  const Hierarchy& hierarchy,
                  ParentMap* parent_map);

// Returns list of ALL spatio-temporal children in the segmentation tree at query_level
// for the specified region at level.
// Note: If you use per-frame operations you probably want to use GetParentMap
// (which is faster to construct, this is a recursive search function with complexity
//  O(log(level) * children_ids->size())).
void GetChildrenIds(int region_id,
                    int level,
                    int query_level,
                    const Hierarchy& seg_hier,
                    std::vector<int>* children_ids);


//// Shape descriptors.
///////////////////////

// Shape moment representation as ellipse.
// Computed from co-variance matrix, mag_major and mag_minor represent one standard
// deviation in the corresponding direction.
struct ShapeDescriptor {
  cv::Point2f center;
  // Magnitude of major and minor axis.
  float mag_major = 0;
  float mag_minor = 0;
  // Direction of major and minor axis (normalized).
  cv::Point2f dir_major = cv::Point2f(1.0f, 0.0f);
  cv::Point2f dir_minor = cv::Point2f(0.0f, 1.0f);
};

// Return value indicates if reasonable shape descriptor could be computed.
// For example if region is too small (< 10 pixels), major and minor axis
// can be unstable. Similar if region is highly eccentric (e.g. is a 2D line).
// However, computation of the centroid is always stable.
bool GetShapeDescriptorFromShapeMoment(const ShapeMoments& moment,
                                       ShapeDescriptor* shape_desc);

// Same as above, but aggregates across all moments (e.g. for compound regions).
bool GetShapeDescriptorFromShapeMoments(const std::vector<const ShapeMoments*>& moments,
                                        ShapeDescriptor* shape_desc);

// Convenience functions calling above functions on each regions rasterization.
bool GetShapeDescriptorFromRegions(const std::vector<const Region2D*>& regions,
                                   ShapeDescriptor* shape_desc);

bool GetShapeDescriptorFromRegion(const Region2D& r,
                                  ShapeDescriptor* shape_desc);

// Renders the shape descriptor of a joint set of regions to output.
// If label is set, label will be printed instead of center cross.
void RenderShapeDescriptor(const std::vector<int>& overseg_region_ids,
                           const SegmentationDesc& desc,
                           cv::Mat* output,
                           const std::string* label = 0);

//// Rasterization.
///////////////////

// Lexicographic ordering for ScanIntervals (first y, then x).
class ScanIntervalComparator {
 public:
  bool operator()(const ScanInterval& lhs, const ScanInterval& rhs) const {
    return lhs.y() < rhs.y() || (lhs.y() == rhs.y() && lhs.left_x () < rhs.left_x());
  }
};


// Perform binary search on scanlines within a region's rasterization.
inline google::protobuf::RepeatedPtrField<ScanInterval>::const_iterator
LocateScanLine(int y, const Rasterization& raster) {
  ScanInterval scan_inter;
  scan_inter.set_y(y);
  scan_inter.set_left_x(-1);
  return lower_bound(raster.scan_inter().begin(),
                     raster.scan_inter().end(),
                     scan_inter,
                     ScanIntervalComparator());
}

// Merges two rasterizations into one.
void MergeRasterization(const Rasterization& raster_1,
                        const Rasterization& raster_2,
                        Rasterization* merged_raster);

// Merges list of rasterizations into one.
void MergeRasterizations(const std::vector<const Rasterization*>& rasters,
                         Rasterization* merged_raster);

// Computes merged rasterization for each CompoundRegion in ParentMap.
// Returns mapping of CompoundRegion's id to it corresponding rasterization.
void GetCompoundRegionRasterizations(
    const ParentMap& parent_map,
    std::unordered_map<int, Rasterization>* parent_rasterizations);

// Helper  construct to represent aggregated 2D Rasterizations across several frames,
// forming a 3D raster.
// List of tuple (#frame, Rasterization ptr). Orderded by #frame.
typedef std::vector<std::pair<int, std::shared_ptr<Rasterization>>> Rasterization3D;

// Merges 3D Rasterization's across all frames.
void MergeRasterization3D(const Rasterization3D& raster_1,
                          const Rasterization3D& raster_2,
                          Rasterization3D* merged_raster);

// Rasterization3D ordered by frame.
struct Rasterization3DLocator {
  bool operator()(const std::pair<int, std::unique_ptr<Rasterization> >& raster_1,
                  const std::pair<int, std::unique_ptr<Rasterization> >& raster_2) {
    return raster_1.first <= raster_2.first;
  }
};

inline Rasterization3D::const_iterator LocateRasterization(
    int frame, const Rasterization3D& raster) {
  return lower_bound(raster.begin(),
                     raster.end(),
                     std::make_pair(frame, std::shared_ptr<Rasterization>()));
}

// Returns area in pixels.
int RasterizationArea(const Rasterization& raster);

// Returns ShapeMoments from a Rasterization.
void ShapeMomentsFromRasterization(const Rasterization& raster,
                                   ShapeMoments* moments);

// Convenience function applying above method to a regions rasterization, setting
// shape_moments member to result.
void SetShapeMomentsFromRegion(Region2D* r);

// Removes spatio-temporal regions not completely contained in [lhs, rhs]
// from a hierarchy level. Removes neighbor ids outside the interval.
// Note: Does not alter children or parent members.
void ConstrainHierarchyToFrameInterval(int lhs,
                                       int rhs,
                                       const HierarchyLevel& input_hierachy,
                                       HierarchyLevel* constraint_hierarchy);

// Converts Segmentation description to image by assigning each pixel its
// corresponding region id. Ouput must be of same dimensions as the segmentation and
// of type CV_32S.
void SegmentationDescToIdImage(int hierarchy_level,
                               const SegmentationDesc& seg,
                               const Hierarchy* seg_hier,   // optional, not needed for
                                                            // hierarchy_level == 0
                               cv::Mat* output);

// Returns region_id at corresponding (x, y) location in an image,
// return value -1 indicates error.
int GetOversegmentedRegionIdFromPoint(int x,
                                      int y,
                                      const SegmentationDesc& seg);

// Builds global hierarchy from several chunk hierarchies. Example:
// // First frame of each chunk containing the hierarchy.
// vector<SegmentationDesc> chunks;
// Hierarchy video_global;
// for (const auto& chunk : chunks) {
//   BuildGlobalHierarchy(chunk.hierarchy(), chunk.hierarchy_frame_idx(), &video_global);
// }
// If passed hierarchy's vary in their number of levels, the global hierarchy will be
// truncated to minimum number of levels across all passed chunk_hierarchy's.
void BuildGlobalHierarchy(const Hierarchy& chunk_hierarchy,
                          int chunk_frame_number,
                          Hierarchy* global_hierarchy);

// Returns true if hierarchy is consistent (checks neighbor relations and child/parent
// relations.
bool VerifyGlobalHierarchy(const Hierarchy& hierarchy);

// Partitions Rasterization into a number of connected components, using disjoint-set
// operations. Returns number of connected components. Components contains rasterization
// of each component if specified.

enum Connectedness {
  N8_CONNECT,
  N4_CONNECT,
};

int ConnectedComponents(const Rasterization& raster,
                        Connectedness connect,
                        std::vector<Rasterization>* components);   // optional.

// Converts a vectorization back to a rasterization.
void RasterVectorization(const Vectorization& vec,
                         const VectorMesh& mesh,
                         int frame_height,
                         Rasterization* raster);

// Convenience function invoking RasterVectorization for each region.
void ReplaceRasterizationFromVectorization(SegmentationDesc* desc);

// Scales vectorization to specified domain.
void ScaleVectorization(int width, int height, SegmentationDesc* desc);

// If vectorization is computed, this can be used to remove the rasterized result.
void RemoveRasterization(SegmentationDesc* desc);

}  // namespace segmentation.

#endif  // VIDEO_SEGMENT_SEGMENT_UTIL_SEGMENTATION_UTIL_H__
