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


#ifndef SEGMENTATION_H__
#define SEGMENTATION_H__

#include "base/base.h"
#include "segment_util/segmentation.pb.h"
#include "segment_util/segmentation_util.h"

#include "dense_seg_graph_interface.h"
#include "segmentation_common.h"
#include "segmentation_graph.h"

namespace segmentation {

typedef SegmentationDesc::HierarchyLevel HierarchyLevel;
class DenseSegGraphCreatorInterface;

struct SegmentationOptions {
  // Over-segmentation options.
  ////////////////////////////////

  // Minimum region size in voxels.
  int min_region_size = 200;

  // If two_stage_segmentation is set, graph is first segmented spatially before segmented
  // in time as well.
  bool two_stage_segmentation = false;

  // Enforces voxel to be spatially (!) connected via N4 neighborhoods.
  bool enforce_n4_connectivity = true;

  // TODO(grundman): Does not work correctly. Do not set to true.
  bool thin_structure_suppression = false;

  // TODO(grundman): Implement connectivity enum for N4 vs N8

  // If set, enforces that spatio-temporal regions are always spatially connected.
  bool enforce_spatial_connectedness = true;


  // Hierarchical segmentation options.
  ///////////////////////////////////////

  // The number of regions is reduced at every level to level_cutoff_fraction times the
  // number of regions in the previous level, until number of regions falls
  // below min_region_num (see below).
  float level_cutoff_fraction = 0.8;

  // Minimum number of regions. Hierarchical segmentation stops when number of regions
  // fall below specified threshold.
  int min_region_num = 10;

  // Maximum number of regions. If RunHierarchicalSegmentation is called with
  // enforce_max_region_num set to true, it is guaranteed that the second level of
  // the hierarchical segmentation has at most max_region_num regions. In that case,
  // the most bottom level should be discarded via DiscardBottomLevel after running
  // segmentation.
  int max_region_num = 2000;

  // Specifies number of buckets that interval [0, 1] is partioned into
  // for hierarchical segmentation. More buckets equals smaller discritization error.
  int num_domain_buckets = 2048;

  // General options.
  ///////////////////////////////
  bool compute_vectorization = false;
};


// Basic usage:
// (for details see DenseSegmentation and HierarchicalSegmentation classes).
// Segmentation seg(640, 480, 0);
//
// // For over-segmentation.
// seg.InitializeOverSegmentation(...);
// seg.AddGenericImage(...);
// seg.AddGenericImage(...);
// seg.ConnectTemporally(...);
// seg.AddGenericImage(...);
// seg.ConnectTemporally(...);
// ... add more frames ...
// seg.RunOverSegmentation();
//
// // For hierarchial segmentation.
// seg.InitializeBaseHierarchyLevel(...);
// seg.AddOverSegmentation(...);
// seg.AddOverSegmentation(...);
// seg.AddOverSegmentation(...);
// ... add more frames ...
// seg.RunHierarchicalSegmentation();
//
// // Get results.
// seg.AssignUniqueRegionIds(...)
// for (int k = 0; k < num_frames_added; ++k) {
//   seg.RetrieveSegmentation3D(...);
// }
class Segmentation {
public:
  // Creates a new Segmentation object to perform either over- or hierarchical
  // segmentation. Pass increasing chunk_ids for streaming mode (starting at zero).
  Segmentation(const SegmentationOptions& options,
               int frame_width,
               int frame_height,
               int chunk_id);

  ~Segmentation() = default;

  // Call before any AddImage* function to allocate memory of sufficient size.
  // Does nothing in case AddImage* was called before.
  void InitializeOverSegmentation(const DenseSegGraphCreatorInterface&,
                                  int chunk_size);

  // Builds incrementally the dense pixel graph.
  // Note: It is required that all data passed to any Add* and Connect* remains valid
  // until RunOversegmentation is called. AFTER calling RunOversegmentation the data
  // can be freed.
  template <class SpatialPixelDistance>
  void AddGenericImage(const SpatialPixelDistance& distance);

  template <class SpatialPixelDistance>
  void AddGenericImageConstrained(const SegmentationDesc& desc,
                                  const SpatialPixelDistance& distance);

  void AddVirtualImageConstrained(const SegmentationDesc& desc);

  template <class TemporalPixelDistance>
  void ConnectTemporally(const TemporalPixelDistance& distance);

  template <class TemporalPixelDistance>
  void ConnectTemporallyAlongFlow(const cv::Mat& flow,
                                  const TemporalPixelDistance& distance);

  void ConnectTemporallyVirtual();
  void ConnectTemporallyVirtualAlongFlow(const cv::Mat& flow);

  // Initializes base level hierarchy from an oversegmentation hierarchy level
  // (containing all regions and their topological information).
  // Specifies descriptor extractors and updaters used during hierarchical
  // segmentation. Note: Extractors are used during this stage to create the descriptors.
  // If input_mapping is specified, the RegionMapping is used to set the counterpart
  // member in each newly added RegionInformation. Conversly, the order of added
  // RegionInformation's is reported in output_mapping if specified.
  // Can be called multiple times when chunking is used.
  void InitializeBaseHierarchyLevel(const HierarchyLevel& hierarchy_level,
                                    const DescriptorExtractorList& extractors,
                                    const DescriptorUpdaterList& updaters,
                                    RegionMapping* input_mapping,
                                    RegionMapping* output_mapping);

  // Adds oversegmentation from obtained from DenseSegmentationUnit to be used during
  // hierarchical segmentation.
  // Note: Extractors are used during this stage to fill the descriptors.
  // Note: InitializeBaseHierarchyLevel has to be called first for each chunk.
  void AddOverSegmentation(const SegmentationDesc& desc,
                           const DescriptorExtractorList& extractors);

  // Enables chunk set based hierarchical segmentation by pulling the segmentation
  // result from each's region counterpart that will be used as constraint.
  // This needs to be called after the counterpart segmentation finished via
  // RunHierarchicalSegmenation.
  void PullCounterpartSegmentationResult(const Segmentation& prev_seg);

  // Runs the actual over-segmentation. Optionally, you can pass the optical flow
  // connecting pixels across time which are used if enforce_spatial_connectedness is set.
  void RunOverSegmentation(const std::vector<cv::Mat>* flows);

  // This requires an external segmentation to be supplied via AddOverSegmentation and
  // InitializeBaseHierarchyLevel.
  // If enforce_max_region_num is set to true, the first reduction is performed such that
  // the number does not exceed max_region_num, regardless of cutoff fraction.
  // If the number of regions does not exceed max_region_num, the first level is
  // duplicated. In either case, the merge or duplication is performed with
  // Rasterizations, i.e. it is save to call DiscardBottomLevel();
  void RunHierarchicalSegmentation(const RegionDistance& distance,
                                   bool enforce_max_region_num);

  // Iterates over all regions in desc testing if rasterization is in [lhs, rhs).
  // Regions, that are completly outside this bound will be labeled as
  // FLAGGED_FOR_REMOVAL and are excluded on RetrieveSegmentation3D.
  // SuperRegions are flagged for removal if all of its children are flagged.
  void ConstrainSegmentationToFrameInterval(int lhs, int rhs);

  // Iterates over regions in all hierarchy levels, adjusting area to only consist
  // of those slices within [lhs, rhs).
  void AdjustRegionAreaToFrameInterval(int lhs, int rhs);

  // Assigns each region in the hierarchy a unique id. Specifically, if a region is
  // constrained and flag use_constraint_ids is set, a regions constrained_id will
  // be used as region id. Otherwise, region id is set to a regions index shifted
  // by region_id_offset[level] yielding unique ids across clips.
  // Maximum id at each level is returned in max_region_ids (OPTIONAL).
  // Vector region_id_offset has to be at least of size ComputedHierarchyLevels().
  void AssignUniqueRegionIds(bool use_constraint_ids,
                             const std::vector<int>& region_id_offset,
                             std::vector<int>* max_region_ids);

  // Retrieves segmentation at a specific frame_number. If output_hierarchy is set to
  // true, hierarchy will be output to desc as well (usually you set this for frame 0).
  void RetrieveSegmentation3D(int frame_number,
                              bool output_hierarchy,
                              bool save_descriptors,
                              SegmentationDesc* desc);

  int ComputedHierarchyLevels() const { return region_infos_.size(); }
  int NumRegionsAtLevel(int level) const { return region_infos_[level]->size(); }

  // Discards bottom most level of the segmentation (only to be used with hierarchical
  // segmentation, after RunHierarchicalSegmentation is called with
  // enforce_max_region_num_, CHECKED).
  void DiscardBottomLevel();

  int NumFramesAdded() const { return frame_number_; }
  int NumBaseHierarchiesAdded() const { return num_base_hierarchies_; }
protected:
  // For each compound region at the current level, find one child in the base-level,
  // query the child's hierarchial_region_ids member at level and sets corresponding
  // output_ids member to the result.
  // It also returns the skeleton for the current hierarchy level, i.e. for each
  // constraint id the set of region that are constraint to it.
  void SetupRegionConstraints(int level,
                              std::vector<int>* output_ids,
                              std::unordered_map<int, std::vector<int>>* skeleton);

  // Adds new Region2D from RegionInformation ri to desc at current frame_number.
  void AddRegion2DToSegmentationDesc(const RegionInformation& ri,
                                     int frame_number,
                                     SegmentationDesc* desc) const;

  // Maps region id to minimum and maximum frame number, when region is present.
  typedef std::unordered_map<int, std::pair<int, int>> IdToBoundsMap;

  // Same as above for compound region.
  // Determines minimum and maximum frame for each compound region, from
  // input prev_bound_map and outputs map for current level in curr_bound_map.
  // If level == 0, you can pass an empty map for prev_bound_map.
  void AddCompoundRegionToSegmentationDesc(const RegionInformation& ri,
                                           int level,
                                           const IdToBoundsMap& prev_bound_map,
                                           IdToBoundsMap* curr_bound_map,
                                           HierarchyLevel* hier) const;
protected:
  SegmentationOptions options_;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int chunk_id_ = 0;

  // Current processed frame number.
  int frame_number_ = 0;

  // Number of times InitializeBaseLevelHierarchy was invoked.
  int num_base_hierarchies_ = 0;

  // Set to true, if PullCounterpartSegmentationResult was called.
  bool is_constrained_hierarchical_segmentation_ = false;

  // Set to true, if RunHierarchicalSegmentation was called with corresponding flag.
  bool enforce_max_region_num_ = false;

  // Set by AssignUniqueRegionIds.
  bool assigned_constrained_ids_ = false;
  bool assigned_unique_ids_ = false;

  // Remember regions added to previous chunk. Flag is used internally to flag regions
  // that are also present in the current chunk.
  std::unordered_map<int, bool> regions_added_to_prev_chunk_;

  // Holds the segmentation graph that is used during over-segmentation.
  std::unique_ptr<DenseSegGraphInterface> dense_seg_graph_;

  // RegionInformation structs with consecutive assigned id's.
  // Each vector entry represents a level of the segmentation.
  std::vector<std::unique_ptr<RegionInfoList>> region_infos_;

  // Used by subsequent calls of AddOverSegmentation. Maps an over-segmentation id
  // to its corresponding RegionInformation object.
  RegionInfoPtrMap region_info_map_;

  // Used during hierarchical segmentation to update descriptors.
  DescriptorUpdaterList descriptor_updaters_;
};

template <class SpatialPixelDistance>
void Segmentation::AddGenericImage(const SpatialPixelDistance& distance) {
  CHECK(dense_seg_graph_ != nullptr) << "Call InitializeOverSegmentation first.";
  dense_seg_graph_->AddNodesAndSpatialEdges(distance);
  ++frame_number_;
}

template <class SpatialPixelDistance>
void Segmentation::AddGenericImageConstrained(const SegmentationDesc& desc,
                                              const SpatialPixelDistance& distance) {
  CHECK(dense_seg_graph_ != nullptr) << "Call InitializeOverSegmentation first.";
  dense_seg_graph_->AddNodesAndSpatialEdgesConstrained(distance, desc);
  ++frame_number_;
}

template <class TemporalPixelDistance>
void Segmentation::ConnectTemporally(const TemporalPixelDistance& distance) {
  CHECK_GE(frame_number_, 2) << "Add at least two images before introducing "
                             << "temporal connections.";
  dense_seg_graph_->AddTemporalEdges(distance);
}

template <class TemporalPixelDistance>
void Segmentation::ConnectTemporallyAlongFlow(const cv::Mat& flow,
                                              const TemporalPixelDistance& distance) {
  CHECK_GE(frame_number_, 2) << "Add at least two images before introducing "
                             << "temporal connections.";
  dense_seg_graph_->AddTemporalFlowEdges(distance, flow);
}

}  // namespace segmentation.

#endif
