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

#ifndef VIDEO_SEGMENT_SEGMENTATION_SEGMANTATION_COMMON_H__
#define VIDEO_SEGMENT_SEGMENTATION_SEGMANTATION_COMMON_H__

#include "base/base.h"
#include "segment_util/segmentation_util.h"
#include "segmentation/region_descriptor.h"

namespace segmentation {

// Holds all information of a Region that is needed for hierarchical segmentation.
struct RegionInformation {
  RegionInformation() = default;

  int index = -1;                    // Location in RegionInfoList.
  int size = 0;                      // Size in voxels.

  // Id of my super-region in the hierarchy level above.
  int parent_idx = -1;

  enum RegionStatus {
    NORMAL, FLAGGED_FOR_REMOVAL
  };

  // Set if region is isolated and not part of the segmentation anymore,
  // region will not be output during RetrieveSegmentation.
  RegionStatus region_status = NORMAL;

  // Sorted array of neighboring regions' index.
  std::vector<int> neighbor_idx;

  // Optional information.
  // A region is either a node, i.e. has a scanline representation,
  // or is a super-region, i.e. has a list of children id's.
  std::unique_ptr<Rasterization3D> raster;                      // Frame slices.
  std::unique_ptr<std::vector<int>> child_idx;                  // Children list, sorted.

  // Information to constrain adjacent RegionSegmentations.
  // Pointer to counterpart in previous segmentation chunk set.
  RegionInformation* counterpart = nullptr;
  int constrained_id = -1;            // Id of parent in previous segmentation chunk
                                      // set.
                                      // Regions with the same id, should belong to the
                                      // same super-region.
  int region_id = -1;                 // Output id in protobuffer. If region is
                                      // constrained, region_id == constraint_id.

  // Will be filled by PullSegmentationResultsFromCounterparts for regions
  // in the over-segmentation if a region has a counterpart. Specifically, we
  // save the result of the counterpart's segmentation as list of parent region_id's
  // from the leaf to the root.
  std::unique_ptr<std::vector<int>> counterpart_region_ids;
  std::vector<std::unique_ptr<RegionDescriptor>> region_descriptors_;

 public:
  // Creates a basic copy (lacking resulting region ids and counterparts).
  std::unique_ptr<RegionInformation> BasicCopy(bool with_rasterization) const;

  // Returns a vector containing each descriptor pair's distance.
  // Distances is expected to be sized correctly, i.e. size needs to be equal to number
  // of AddRegionDescriptor.
  void DescriptorDistances(const RegionInformation& rhs,
                           std::vector<float>* distances) const;

  // Adds RegionDescriptor to list of descriptors for this region.
  // Note, framework guarantees that order of added descriptors is constant for all
  // regions.
  void AddRegionDescriptor(std::unique_ptr<RegionDescriptor> desc);

  RegionDescriptor* GetDescriptorAtIdx(int idx) {
    DCHECK_LT(idx, region_descriptors_.size());
    DCHECK(region_descriptors_[idx] != nullptr);
    return region_descriptors_[idx].get();
  }

  int NumDescriptors() const { return region_descriptors_.size(); }

  // Calls PopulatingDescriptorFinished for each RegionDescriptor.
  void PopulatingDescriptorsFinished();

  // Merges or copies (if not present in this RegionInformation)
  // all RegionDescriptors from rhs.
  void MergeDescriptorsFrom(const RegionInformation& rhs);

  void UpdateDescriptors(const DescriptorUpdaterList& updater_list);

  // Outputs all regions descriptors to an AggregatedDescriptor in segmentation.proto
  void OutputRegionFeatures(RegionFeatures* features) const;
};

// Maps the id of the region representative to its assigned RegionInformation.
typedef std::unordered_map<int, RegionInformation*> RegionInfoPtrMap;

// Represents all information of a segmentation at a specific level of the hierarchy.
typedef std::vector<std::unique_ptr<RegionInformation>> RegionInfoList;

// A region mapping map region id to created RegionInformation. It is used to 
// associate the irregular segmentation graphs between two chunk sets.
typedef std::unordered_map<int, RegionInformation*> RegionMapping;

// Merges all information of src into dst.
void MergeRegionInfoInto(const RegionInformation* src, RegionInformation* dst);

// Use after MergeRegionInfoInto, to propagate merge to neighboring regions, i.e.
// indicate to all regions in the tree that the region with current_idx at hierarchy
// level at_level was merged into region new_idx.
// After calling this method region[current_idx] = -1, i.e. region be will effectively
// isolated from the tree.
// TODO(grundman): Is this currently used?
void CompleteRegionTreeIndexChange(
    int current_idx,
    int at_level,
    int new_idx,
    std::vector<std::unique_ptr<RegionInfoList>>* region_tree);

// Returns true if item was inserted, if it was present already return false.
template<class T> bool InsertSortedUniquely(const T& t, std::vector<T>* array) {
  auto insert_pos = lower_bound(array->begin(), array->end(), t);
  if (insert_pos == array->end() || *insert_pos != t) {
    array->insert(insert_pos, t);
    return true;
  } else {
    return false;
  }
}

// Common utility classes and functions to hash undirected egdes.
struct UnorderedTuple {
  UnorderedTuple() : node_1(-1), node_2(-1) { }
  UnorderedTuple(int node_1_, int node_2_) {
    if (node_1_ < node_2_) {
      node_1 = node_1_;
      node_2 = node_2_;
    } else {
      node_1 = node_2_;
      node_2 = node_1_;
    }
  }

  bool operator==(const UnorderedTuple& rhs) const {
    return node_1 == rhs.node_1 && node_2 == rhs.node_2;
  }

  int node_1;
  int node_2;
};

class UnorderedTupleHasher {
 public:
  UnorderedTupleHasher(int av_neighbors = 1) : av_neighbors_(av_neighbors) { }

  size_t operator()(const UnorderedTuple& tuple) const {
    return hasher(tuple.node_1 * av_neighbors_ + tuple.node_2 % av_neighbors_);
  }

 private:
  std::hash<int> hasher;
  int av_neighbors_;
};

}  // namespace segmentation.

#endif // VIDEO_SEGMENT_SEGMENTATION_SEGMANTATION_COMMON_H__
