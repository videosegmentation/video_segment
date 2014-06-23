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

#include "segmentation_common.h"

#include "base/base_impl.h"
#include "segment_util/segmentation_util.h"

namespace segmentation {

std::unique_ptr<RegionInformation> RegionInformation::BasicCopy(
    bool with_rasterization) const {
  std::unique_ptr<RegionInformation> result(new RegionInformation());
  result->size = size;
  result->neighbor_idx = neighbor_idx;
  result->MergeDescriptorsFrom(*this);
  if (with_rasterization) {
    result->raster.reset(new Rasterization3D(*raster));
  }

  return std::move(result);
}

void RegionInformation::DescriptorDistances(const RegionInformation& rhs,
                                            vector<float>* distances) const {
  DCHECK_EQ(region_descriptors_.size(), rhs.region_descriptors_.size());
  DCHECK_EQ(distances->size(), region_descriptors_.size());
  for (int idx = 0; idx < region_descriptors_.size(); ++idx) {
    (*distances)[idx] =
        region_descriptors_[idx]->RegionDistance(*rhs.region_descriptors_[idx]);
  }
}

void RegionInformation::AddRegionDescriptor(std::unique_ptr<RegionDescriptor> desc) {
  desc->SetParent(this);
  region_descriptors_.push_back(std::move(desc));
}

void RegionInformation::PopulatingDescriptorsFinished() {
  for (auto& descriptor_ptr : region_descriptors_) {
    descriptor_ptr->PopulatingDescriptorFinished();
  }
}

void RegionInformation::MergeDescriptorsFrom(const RegionInformation& rhs) {
  // This could be a new super-region (empty descriptors). Allocate in that case.
  const int num_descriptors = rhs.region_descriptors_.size();

  if (region_descriptors_.empty()) {
    region_descriptors_.resize(num_descriptors);
  }

  for (int descriptor_idx = 0; descriptor_idx < num_descriptors;  ++descriptor_idx) {
    DCHECK(rhs.region_descriptors_[descriptor_idx] != nullptr);
    if (region_descriptors_[descriptor_idx] == nullptr) {
      region_descriptors_[descriptor_idx].reset(
          rhs.region_descriptors_[descriptor_idx]->Clone());
      // Update parent.
      region_descriptors_[descriptor_idx]->SetParent(this);
    } else {
      region_descriptors_[descriptor_idx]->MergeWithDescriptor(
          *rhs.region_descriptors_[descriptor_idx]);
    }
  }
}

void RegionInformation::UpdateDescriptors(const DescriptorUpdaterList& updater_list) {
  DCHECK_EQ(updater_list.size(), region_descriptors_.size());
  for (int desc_idx = 0, desc_num = region_descriptors_.size();
       desc_idx < desc_num;
       ++desc_idx) {
    region_descriptors_[desc_idx]->UpdateDescriptor(*updater_list[desc_idx]);
  }
}

void RegionInformation::OutputRegionFeatures(
    RegionFeatures* features) const {
  for (const auto& descriptor_ptr : region_descriptors_) {
    descriptor_ptr->AddToRegionFeatures(features);
  }
}

void MergeRegionInfoInto(const RegionInformation* src, RegionInformation* dst) {
  // Update area.
  dst->size += src->size;

  // Merge neighbor ids, avoid duplicates.
  vector<int> merged_neighbors;
  merged_neighbors.reserve(src->neighbor_idx.size() + dst->neighbor_idx.size());
  std::set_union(src->neighbor_idx.begin(), src->neighbor_idx.end(),
                 dst->neighbor_idx.begin(), dst->neighbor_idx.end(),
                 std::back_inserter(merged_neighbors));

  // Avoid adding dst->region_id to merged_neighbors.
  vector<int>::iterator dst_id_location =
    std::lower_bound(merged_neighbors.begin(),
                     merged_neighbors.end(),
                     dst->index);

  if (dst_id_location != merged_neighbors.end() &&
      *dst_id_location == dst->index) {
    merged_neighbors.erase(dst_id_location);
  }

  dst->neighbor_idx.swap(merged_neighbors);

  // Merge scanline representation.
  if (src->raster != nullptr) {
    // Both RegionInfo's need to have rasterization present.
    DCHECK(dst->raster != nullptr);
    std::unique_ptr<Rasterization3D> merged_raster(new Rasterization3D());
    MergeRasterization3D(*src->raster, *dst->raster, merged_raster.get());
    dst->raster.swap(merged_raster);
  }

  // Merge children.
  if (src->child_idx != nullptr) {
    DCHECK(dst->child_idx != nullptr);
    vector<int> merged_children;
    merged_children.reserve(src->child_idx->size() + dst->child_idx->size());
    std::set_union(src->child_idx->begin(), src->child_idx->end(),
                   dst->child_idx->begin(), dst->child_idx->end(),
                   back_inserter(merged_children));
    dst->child_idx->swap(merged_children);
  }

  // Merge histograms.
  dst->MergeDescriptorsFrom(*src);
}

void CompleteRegionTreeIndexChange(
    int current_idx,
    int at_level,
    int new_idx,
    std::vector<std::unique_ptr<RegionInfoList>>* region_tree) {
  // Fetch RegionInformation from region_tree.
  CHECK(region_tree);
  CHECK_LT(at_level, region_tree->size());
  CHECK_LT(current_idx, (*region_tree)[at_level]->size());

  RegionInformation* region_info = (*region_tree)[at_level]->at(current_idx).get();

  // Notify neighbor's of id change.
  for (int neighbor_idx : region_info->neighbor_idx) {
    RegionInformation* neighbor_info = (*region_tree)[at_level]->at(neighbor_idx).get();
    // Find current_idx.
    auto idx_pos = std::lower_bound(neighbor_info->neighbor_idx.begin(),
                                    neighbor_info->neighbor_idx.end(),
                                    current_idx);
    DCHECK_EQ(*idx_pos, current_idx);
    // Erase and insert new_id, except if new_id and neighbor's id are identical
    // (no region should be neighbor of itself). In this case only erase.
    neighbor_info->neighbor_idx.erase(idx_pos);

    if (neighbor_info->index != new_idx) {
      InsertSortedUniquely(new_idx, &neighbor_info->neighbor_idx);
    }
  }

  // Notify children of id change.
  if (region_info->child_idx != nullptr) {
    for (int child_idx : *region_info->child_idx) {
      RegionInformation* child_info = (*region_tree)[at_level - 1]->at(child_idx).get();
      DCHECK_EQ(child_info->parent_idx, current_idx);
      child_info->parent_idx = new_idx;
    }
  }

  // Notify parent of id change.
  if (region_info->parent_idx >= 0) {
    RegionInformation* parent_info =
        (*region_tree)[at_level + 1]->at(region_info->parent_idx).get();

    // Find current id in parent's children.
    DCHECK(parent_info->child_idx);

    auto child_pos = std::lower_bound(parent_info->child_idx->begin(),
                                      parent_info->child_idx->end(),
                                      current_idx);
    DCHECK_EQ(*child_pos, current_idx);

    // Erase and insert new id.
    parent_info->child_idx->erase(child_pos);
    InsertSortedUniquely(new_idx, parent_info->child_idx.get());
  }

  // Isolate region, after propagating merge to all adjacent regions.
  region_info->index = -1;
}

}  // namespace segmentation.
