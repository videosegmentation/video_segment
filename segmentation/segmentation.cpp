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


#include "segmentation/segmentation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base/base_impl.h"
#include "segment_util/segmentation_util.h"
#include "segment_util/segmentation_render.h"
#include "segmentation/boundary.h"
#include "segmentation/dense_segmentation_graph.h"
#include "segmentation/region_segmentation_graph.h"

namespace segmentation {

typedef SegmentationDesc::Region2D SegRegion;
typedef SegmentationDesc::CompoundRegion SegCompRegion;

Segmentation::Segmentation(const SegmentationOptions& options,
                           int frame_width,
                           int frame_height,
                           int chunk_id)
    : options_(options),
      frame_width_(frame_width),
      frame_height_(frame_height),
      chunk_id_(chunk_id) {
}

void Segmentation::InitializeOverSegmentation(
    const DenseSegGraphCreatorInterface& creator,
    int chunk_size) {
  dense_seg_graph_.reset(creator.CreateDenseSegGraph(frame_width_,
                                                     frame_height_,
                                                     chunk_size));
  dense_seg_graph_->InitializeGraph();
}

void Segmentation::AddVirtualImageConstrained(const SegmentationDesc& desc) {
  dense_seg_graph_->AddVirtualNodesConstrained(desc);
  ++frame_number_;
}

void Segmentation::ConnectTemporallyVirtual() {
  dense_seg_graph_->AddTemporalVirtualEdges();
}

void Segmentation::ConnectTemporallyVirtualAlongFlow(const cv::Mat& flow) {
  dense_seg_graph_->AddTemporalFlowVirtualEdges(flow);
}

void Segmentation::InitializeBaseHierarchyLevel(
    const HierarchyLevel& hierarchy_level,
    const DescriptorExtractorList& extractors,
    const DescriptorUpdaterList& updaters,
    RegionMapping* input_mapping,
    RegionMapping* output_mapping) {
  // First call to init a base level hierarchy.
  if (region_infos_.size() != 1) {
    region_infos_.resize(1);
    region_infos_[0].reset(new RegionInfoList());
  }

  // Before we start a new chunk, all regions added in the previous chunk that are
  // not present in the current one can be compacted.
  // First pass for compound regions:
  // Test for each region if it was present in previous chunk -> flag if so
  for (const auto& region : hierarchy_level.region()) {
    // Present in prev. chunk?
    auto map_pos = regions_added_to_prev_chunk_.find(region.id());
    // Flag as present.
    if (map_pos != regions_added_to_prev_chunk_.end()) {
      map_pos->second = true;
    }
  }

  // Compact all regions in previous chunk that are not present anymore.
  int num_compacted = 0;
  for (const auto& map_entry : regions_added_to_prev_chunk_) {
    // Not present anymore.
    if (!map_entry.second) {
      DCHECK(region_info_map_.find(map_entry.first) != region_info_map_.end());
      region_info_map_[map_entry.first]->PopulatingDescriptorsFinished();
      ++num_compacted;
    }
  }

  VLOG(1) << "Compacted: " << num_compacted << " regions";

  if (output_mapping) {
    output_mapping->clear();
  }

  // Clear flags.
  regions_added_to_prev_chunk_.clear();

  // For each passed compound region, create new RegionInformation or
  // add to already existing one (ids and neighbors only, actual image
  // content is filled from Region2D by AddOversegmentation).
  for (const CompoundRegion& region : hierarchy_level.region()) {
    const int region_id = region.id();

    // Flag as added.
    regions_added_to_prev_chunk_[region_id] = false;

    // Does RegionInfo already exists?
    auto map_iter = region_info_map_.find(region_id);
    RegionInformation* ri = nullptr;
    if (map_iter == region_info_map_.end()) {
      // This is a new region, create new RegionInformation.
      std::unique_ptr<RegionInformation> new_info(new RegionInformation());
      ri = new_info.get();

      // Indices will be assigned subsequently. Index should correspond to position
      // in RegionInfoList.
      new_info->index = region_infos_[0]->size();
      new_info->size = region.size();
      new_info->raster.reset(new Rasterization3D);

      // Setup descriptors.
      for (const auto& extractor_ptr : extractors) {
        new_info->AddRegionDescriptor(extractor_ptr->CreateDescriptor());
      }

      // Set counterpart from input_mapping.
      if (input_mapping) {
        const auto counterpart = input_mapping->find(region_id);
        // Not guaranteed to be present.
        if (counterpart != input_mapping->end()) {
          new_info->counterpart = counterpart->second;
        }
      }

      // Save in base level.
      region_infos_[0]->push_back(std::move(new_info));
      // Save to map of added regions.
      region_info_map_.insert(std::make_pair(region_id, ri));
    } else {
      // Exisiting region, add size.
      ri = map_iter->second;
      ri->size += region.size();
    }

    // Remember RegionInformation's for each region in order.
    if (output_mapping) {
      (*output_mapping)[region_id] = ri;
    }
  } // end loop over compound regions

  // All regions in this chunk and potential overlap are added now.
  // Add neighbors if not existing yet.
  for (const CompoundRegion& region : hierarchy_level.region()) {
    auto ri_iter = region_info_map_.find(region.id());
    DCHECK(ri_iter != region_info_map_.end());
    RegionInformation* ri = ri_iter->second;

    for (int n_id : region.neighbor_id()) {
      // Only add if it does not have our neighbors already.
      // We need to map neighbors to index of the corresponding RegionInformation.
      auto neighbor_iter = region_info_map_.find(n_id);
      DCHECK(neighbor_iter != region_info_map_.end());
      RegionInformation* neighbor_ri = neighbor_iter->second;
      const int n_idx = neighbor_ri->index;

      InsertSortedUniquely(n_idx, &ri->neighbor_idx);
    }
  }

  // Record number of times this function was called.
  ++num_base_hierarchies_;

  // Save updaters for hierarchical segmentation stage.
  descriptor_updaters_ = updaters;
}

void Segmentation::AddOverSegmentation(const SegmentationDesc& desc,
                                       const DescriptorExtractorList& extractors) {
  // Add regions of desc to current set of regions.
  for (const SegRegion& r : desc.region()) {
    const int region_id = r.id();

    // Needs to have corresponding RegionInformation at this point.
    auto map_iter = region_info_map_.find(region_id);

    DCHECK(map_iter != region_info_map_.end()) << "Region "
        << region_id << " should be present here! chunk: " << chunk_id_;
    RegionInformation* ri = map_iter->second;

    // This ScanlineRep should not present at current frame.
    // Allow for skipped frames in rasterization but expect it to be
    // monotonically order.
    if (ri->raster->size() != 0) {
      DCHECK(ri->raster->back().first < frame_number_)
        << "Rasterization slice needs to be monotonically increasing and "
           "not present at current frame";
    }

    ri->raster->push_back(std::make_pair(frame_number_, std::shared_ptr<Rasterization>(
        new Rasterization())));
    ri->raster->back().second->CopyFrom(r.raster());

    // Fill descriptors.
    int num_descriptors = ri->NumDescriptors();
    CHECK_EQ(num_descriptors, extractors.size());
    for (int d = 0; d < num_descriptors; ++d) {
      ri->GetDescriptorAtIdx(d)->AddFeatures(r.raster(), *extractors[d], frame_number_);
    }
  }
  ++frame_number_;
}

void Segmentation::PullCounterpartSegmentationResult(const Segmentation& prev_seg) {
  // Traverse all region information's added so far.
  const int levels = prev_seg.region_infos_.size();

  for (const auto& region_ptr : *region_infos_[0]) {
    // Ignore, if counterpart isn't set.
    if (region_ptr->counterpart == nullptr) {
      continue;
    }

    // Copy constraint for over-segmentation.
    region_ptr->constrained_id = region_ptr->counterpart->region_id;

    // Pull hierarchical constraints, that is record region_ids of segmentation result
    // up the hierarchy.
    std::unique_ptr<std::vector<int>> constrained_ids(new std::vector<int>(levels - 1));

    int curr_idx = region_ptr->counterpart->parent_idx;
    for (int l = 1; l < levels; ++l) {
      // Record.
      (*constrained_ids)[l - 1] = (*prev_seg.region_infos_[l])[curr_idx]->region_id;
      // Advance.
      curr_idx = (*prev_seg.region_infos_[l])[curr_idx]->parent_idx;
    }

    region_ptr->counterpart_region_ids.swap(constrained_ids);
  }

  is_constrained_hierarchical_segmentation_ = true;
}

void Segmentation::RunOverSegmentation() {
  std::unique_ptr<RegionInfoList> region_list(new RegionInfoList());

  CHECK(dense_seg_graph_ != nullptr) << "Call InitializeOverSegmentation first.";

  // Wait for parallel graph construction to finish.
  dense_seg_graph_->FinishBuildingGraph();

  if (options_.two_stage_segmentation) {
    // Skip temporal edges in first pass.
    dense_seg_graph_->SegmentGraphSpatially();
    dense_seg_graph_->SegmentFullGraph(options_.min_region_size, true);
  } else {
    dense_seg_graph_->SegmentFullGraph(options_.min_region_size, true);
  }

  RegionInfoPtrMap map;
  dense_seg_graph_->ObtainResults(
      region_list.get(),
      &map,
      options_.thin_structure_suppression,
      options_.enforce_n4_connectivity);

  dense_seg_graph_->DetermineNeighborIds(region_list.get(), &map);

  // De-allocate dense_graph.
  dense_seg_graph_.reset();
  region_infos_.clear();
  region_infos_.push_back(std::move(region_list));
}

void Segmentation::RunHierarchicalSegmentation(const RegionDistance& distance,
                                               bool enforce_max_region_num) {

  CHECK(!region_infos_.empty() && region_infos_[0] != nullptr)
    << "InitializeBaseHierarchyLevel wasn't called.";
  VLOG(1) << "Hierarchical segmentation with " << frame_number_ << " frames.";
  enforce_max_region_num_ = enforce_max_region_num;

  // Finish descriptors.
  for (auto& region_ptr : *region_infos_[0]) {
    region_ptr->PopulatingDescriptorsFinished();
  }

  // Counts number of total hierarchy levels that will be output to SegmentationDesc
  // protobuffer.
  int hierarchy_levels = 0;
  int curr_region_num = region_infos_[0]->size();

  // Edge map used to create subsequent levels more efficiently.
  RegionAgglomerationGraph::EdgeWeightMap edge_weight_map;

  while (curr_region_num > options_.min_region_num) {
    // Setup new graph and segment.
    std::unique_ptr<RegionAgglomerationGraph> region_graph(
        new RegionAgglomerationGraph(1.0f, options_.num_domain_buckets, &distance));

    // Update descriptors.
    for (auto& updater : descriptor_updaters_) {
      updater->InitializeUpdate(*region_infos_[hierarchy_levels]);
    }

    for (auto& region_ptr : *region_infos_[hierarchy_levels]) {
      region_ptr->UpdateDescriptors(descriptor_updaters_);
    }

    // Build graph for this level.
    if (is_constrained_hierarchical_segmentation_) {
      std::vector<int> parent_constraint_ids;
      std::unordered_map<int, vector<int>> skeleton;
      SetupRegionConstraints(hierarchy_levels, &parent_constraint_ids, &skeleton);
      region_graph->AddRegionEdgesConstrained(
          *region_infos_[hierarchy_levels],
          hierarchy_levels == 0 ? nullptr : &edge_weight_map,
          parent_constraint_ids,
          skeleton);
    } else {
      region_graph->AddRegionEdges(*region_infos_[hierarchy_levels],
                                   hierarchy_levels == 0 ? nullptr : &edge_weight_map);
    }

    // Perform segmentation.
    if (hierarchy_levels == 0 && enforce_max_region_num_) {
      // Ensure first segmentation results into approximately max_region_num region
      float cutoff = std::min(1.0f, options_.max_region_num *
                              (1.0f / region_infos_[0]->size()));
      region_graph->SegmentGraph(true, cutoff);  // merge rasterization to enable
                                                 // discarding bottom level.
    } else {
      if (!region_graph->SegmentGraph(false, options_.level_cutoff_fraction)) {
        // If no merge at all, warn and break.
        LOG(ERROR) << "No merge possible for current cut_off_fraction. "
                   << "Premature return.";
        break;
      }
    }

    region_infos_.push_back(std::unique_ptr<RegionInfoList>(new RegionInfoList()));
    region_graph->ObtainSegmentationResult(region_infos_[hierarchy_levels].get(),
                                           region_infos_.back().get(),
                                           &edge_weight_map);

    if (region_infos_.back()->size() <= 1) {
      LOG(WARNING) << "Merging process resulted in only one region!\n";
    }

    // Update termination.
    curr_region_num = region_infos_[hierarchy_levels]->size();
    ++hierarchy_levels;

    VLOG(1) << "Finished level " << hierarchy_levels << " with "
            << curr_region_num << " elems.\n";
  }
}


void Segmentation::ConstrainSegmentationToFrameInterval(int lhs, int rhs) {
  // Traverse over all regions in over-segmentation.
  int num_removed_regions = 0;
  for (auto& region_ptr : *region_infos_[0]) {
    // Is slice region out of bounds or empty?
    if (region_ptr->raster == nullptr || region_ptr->raster->empty() ||
        region_ptr->raster->front().first >= rhs ||
        region_ptr->raster->back().first < lhs) {
      ++num_removed_regions;
      region_ptr->region_status = RegionInformation::FLAGGED_FOR_REMOVAL;
    }
  }

  // Process remaining levels.
  for (int level = 1; level < region_infos_.size(); ++level) {
    for (auto& region_ptr : *region_infos_[level]) {
      RegionInformation::RegionStatus status = RegionInformation::FLAGGED_FOR_REMOVAL;
      // Try to find at least one child not flagged for removal.
      for (int child : *region_ptr->child_idx) {
        if (region_infos_[level - 1]->at(child)->region_status ==
            RegionInformation::NORMAL) {
          status = RegionInformation::NORMAL;
          break;
        }
      }
      region_ptr->region_status = status;
    }
  }
}

void Segmentation::AdjustRegionAreaToFrameInterval(int lhs, int rhs) {
  // Maps region id to how much we change in region area
  // (of previously processed hierarchy level).
  std::unordered_map<int, int> prev_size_adjust;

  // Process over-segmentation first.
  for (auto& region_ptr : *region_infos_[0]) {
    int size_increment = 0;
    if (region_ptr->raster == nullptr) {
      continue;
    }
    for (const auto& slice : *region_ptr->raster) {
      if (slice.first < lhs || slice.first >= rhs) {
        size_increment -= RasterizationArea(*slice.second);
      }
    }
    region_ptr->size += size_increment;
    prev_size_adjust[region_ptr->index] = size_increment;
  }

  // Process remaining levels.
  for (int level = 1; level < region_infos_.size(); ++level) {
    std::unordered_map<int, int> curr_size_adjust;
    for (auto& region_ptr : *region_infos_[level]) {
      int size_increment = 0;
      for (int child : *region_ptr->child_idx) {
        size_increment += prev_size_adjust[child];
      }

      region_ptr->size += size_increment;
      curr_size_adjust[region_ptr->index] = size_increment;
    }
    prev_size_adjust.swap(curr_size_adjust);
  }
}

void Segmentation::RetrieveSegmentation3D(int frame_number,
                                          bool output_hierarchy,
                                          bool save_descriptors,
                                          SegmentationDesc* desc) {
  // If no explicit assignment requested, use region's index as region_id.
  if (!assigned_unique_ids_) {
    LOG(WARNING) << "No ids assigned yet. Calling AssignUniqueRegionIds with defaults: "
                 << "No constraints, no offsets.";
    AssignUniqueRegionIds(false, vector<int>(1, 0), nullptr);
  }

  const RegionInfoList& curr_list = *region_infos_[0];

  const int levels = ComputedHierarchyLevels();

  desc->set_frame_width(frame_width_);
  desc->set_frame_height(frame_height_);
  desc->set_chunk_id(chunk_id_);
  desc->set_connectedness(
      options_.enforce_n4_connectivity ? SegmentationDesc::N4_CONNECT
                                       : SegmentationDesc::N8_CONNECT);

  // Add 2d oversegmenation regions at current frames.
  for (const auto& region_ptr : curr_list) {
    AddRegion2DToSegmentationDesc(*region_ptr, frame_number, desc);
  }

  // Sort regions by id.
  if (assigned_constrained_ids_) {
    SortRegions2DById(desc);
  }

  // Save descriptors when hierarchy is requested.
  if (output_hierarchy && save_descriptors) {
    for (const auto& region_ptr : curr_list) {
      if (region_ptr->region_status == RegionInformation::FLAGGED_FOR_REMOVAL) {
        continue;
      }

      RegionFeatures* features = desc->add_features();
      features->set_id(region_ptr->region_id);
      region_ptr->OutputRegionFeatures(features);
    }
  }

  if (output_hierarchy) {
    IdToBoundsMap prev_bound_map;
    IdToBoundsMap curr_bound_map;
    for (int l = 0; l < levels; ++l) {
      const RegionInfoList& hier_list = *region_infos_[l];
      HierarchyLevel* hier = desc->add_hierarchy();
      curr_bound_map.clear();

      for (const auto& region_ptr : hier_list) {
        AddCompoundRegionToSegmentationDesc(*region_ptr,
                                            l,
                                            prev_bound_map,
                                            &curr_bound_map,
                                            hier);
      }

      prev_bound_map.swap(curr_bound_map);

      if (assigned_constrained_ids_) {
        SortCompoundRegionsById(desc->mutable_hierarchy(l));
      }
    }
  }

  if (options_.compute_vectorization) {
    BoundaryComputation boundary_comp(frame_width_, frame_height_, 10);
    std::vector<std::unique_ptr<Boundary>> boundaries;
    boundary_comp.ComputeBoundary(*desc, &boundaries);
    boundary_comp.ComputeVectorization(boundaries, 4, 1.0, desc);
  }
}

namespace {

int ConstraintIdFromRegion(const RegionInformation& region,
                           bool use_constraint_ids,
                           int id_offset) {
  if (use_constraint_ids && region.constrained_id >= 0) {
    return region.constrained_id;
  } else {
    return region.index + id_offset;
  }
}

}  // namespace.

void Segmentation::AssignUniqueRegionIds(bool use_constrained_ids,
                                         const vector<int>& region_id_offsets,
                                         vector<int>* max_region_ids) {
  assigned_constrained_ids_ = use_constrained_ids;
  assigned_unique_ids_ = true;

  CHECK_GE(region_id_offsets.size(), region_infos_.size());

  if (max_region_ids) {
    CHECK_GE(max_region_ids->size(), region_infos_.size());
  }

  vector<int> local_id_offsets = region_id_offsets;
  if (local_id_offsets.size() < ComputedHierarchyLevels()) {
    local_id_offsets.resize(ComputedHierarchyLevels());
  }

  // Traverse levels.
  for (int l = 0; l < region_infos_.size(); ++l) {
    // Traverse regions.
    int max_id = -1;
    for (auto& region_ptr : *region_infos_[l]) {
      region_ptr->region_id = ConstraintIdFromRegion(*region_ptr,
                                                     use_constrained_ids,
                                                     local_id_offsets[l]);
      max_id = std::max(max_id, region_ptr->region_id);
    }

    if (max_region_ids) {
      // Guarantee max id is monotonically increasing.
      max_region_ids->at(l) = std::max(region_id_offsets[l], max_id + 1);
    }
  }
}

void Segmentation::DiscardBottomLevel() {
  CHECK(enforce_max_region_num_) << "Requires RunHierarchicalSegmentation to be called"
                                 << " with enforce_max_region_num == true";

  // Clear children in next level.
  for (auto& region_ptr : *region_infos_[1]) {
    region_ptr->child_idx.reset();
  }

  // Discard level 0.
  region_infos_.erase(region_infos_.begin());
}

void Segmentation::SetupRegionConstraints(int level,
                                          vector<int>* output_ids,
                                          unordered_map<int, vector<int>>* skeleton) {
  output_ids->clear();
  output_ids->reserve(region_infos_[level]->size());

  for (const auto& region_ptr : *region_infos_[level]) {
    // Find one constraint child in base-level, if it exists.
    int constraint_child_idx = region_ptr->index;
    if (level > 0) {
      // Traverse from current to base level.
      for (int l = level; l > 0; --l) {
        // Is there at least one constraint child (at any level).
        bool constraint_child_found = false;
        const RegionInformation& child =
            *region_infos_[l]->at(constraint_child_idx);
        for (int test_child : *child.child_idx) {
          if ((*region_infos_[l - 1])[test_child]->constrained_id >= 0) {
            constraint_child_idx = test_child;
            constraint_child_found = true;
            break;     // Found one child, break out of children loop
          }
        }

        if (!constraint_child_found) {
          // There is no hope to find a constraint base-level child, i.e. this
          // region is unconstrained.
          constraint_child_idx = -1;

          // If this happens, it has to occur at the top-level, otherwise there
          // is some kind of inconsistency.
          DCHECK_EQ(l, level) << "Sudden switch from constraint to unconstraint "
                              << "level. Inconsistency.";

          break;      // break out of level loop.
        }
      }  // end level traversal.
    } else {
      if (region_ptr->constrained_id < 0) {
        constraint_child_idx = -1;
      }
    }

    // child_idx denotes the region at the base level that is contrained.

    // Pull constraint, if applicable.
    if (constraint_child_idx >= 0) {
      const RegionInformation& base_level_child =
          *(*region_infos_[0])[constraint_child_idx];
      if (base_level_child.counterpart_region_ids != nullptr) {
        // Only update if enough constraints are available.
        if (level < base_level_child.counterpart_region_ids->size()) {
          const int constraint_id = (*base_level_child.counterpart_region_ids)[level];
          // Constraint is hold the constrained if for the current region at level
          // level.
          output_ids->push_back(constraint_id);
          (*skeleton)[constraint_id].push_back(region_ptr->index);
        } else {
          output_ids->push_back(-1);  // Unconstrained, level not present.
        }
      } else {
        output_ids->push_back(-1);   // Unconstrained, no counter parts
        LOG(FATAL) << "Lack of counterparts should not happen ...";
      }
    } else {
      output_ids->push_back(-1);  // Unconstrained, no constrained child.
    }
  }
}

void Segmentation::AddRegion2DToSegmentationDesc(const RegionInformation& ri,
                                                 int frame_number,
                                                 SegmentationDesc* desc) const {
  // Does current RegionInfo has scanline in current frame?
  if (ri.raster == nullptr) {
    DCHECK_EQ(0, ri.size) << "Expected virtual node.";
    return;
  }

  auto raster_iter = LocateRasterization(frame_number, *ri.raster);
  if (raster_iter == ri.raster->end() || raster_iter->first != frame_number) {
    // Not every region is present at the request frame.
    return;
  }

  // We always return the segmentation for a requested frame, even if it is
  // flagged for removal, i.e. part of the overlap. This is as the rasterization
  // for constraints is needed for the next chunk.

  // Should not be empty.
  CHECK_NE(raster_iter->second->scan_inter_size(), 0);

  // Add info to desc.
  SegRegion* r = desc->add_region();
  r->set_id(ri.region_id);
  r->mutable_raster()->CopyFrom(*raster_iter->second);
  SetShapeMomentsFromRegion(r);
}

void Segmentation::AddCompoundRegionToSegmentationDesc(
    const RegionInformation& ri,
    int level,
    const IdToBoundsMap& prev_bound_map,
    IdToBoundsMap* curr_bound_map,
    HierarchyLevel* hier) const {
  DCHECK(curr_bound_map);

  if (ri.region_status == RegionInformation::FLAGGED_FOR_REMOVAL) {
    return;
  }

  SegCompRegion* r = hier->add_region();
  r->set_id(ri.region_id);
  r->set_size(ri.size);

  const RegionInfoList& curr_list = *region_infos_[level];
  for (int n : ri.neighbor_idx) {
    if (curr_list[n]->region_status == RegionInformation::FLAGGED_FOR_REMOVAL) {
      continue;   // not present anymore.
    }

    r->add_neighbor_id(curr_list[n]->region_id);
  }

  // We might have to sort neighbors.
  if (assigned_constrained_ids_) {
    std::sort(r->mutable_neighbor_id()->begin(), r->mutable_neighbor_id()->end());
  }

  const int levels = ComputedHierarchyLevels();

  if (level < levels - 1) {
    r->set_parent_id(region_infos_[level + 1]->at(ri.parent_idx)->region_id);
  }

  // Set children and determine minimum and maximum frame bounds.
  int min_frame = std::numeric_limits<int>::max();
  int max_frame = 0;
  if (level > 0) {       // no children present in oversegmentation.
    DCHECK(ri.child_idx != nullptr);
    for (int c : *ri.child_idx) {
      if (region_infos_[level - 1]->at(c)->region_status ==
          RegionInformation::FLAGGED_FOR_REMOVAL) {
        continue;   // not present anymore.
      }

      r->add_child_id(region_infos_[level - 1]->at(c)->region_id);

      // Update bounds.
      const auto child_bound_pos = prev_bound_map.find(c);
      DCHECK(child_bound_pos != prev_bound_map.end())
          << "Child bound " << c << " not found.";

      min_frame = std::min(min_frame, child_bound_pos->second.first);
      max_frame = std::max(max_frame, child_bound_pos->second.second);
    }

    // Might need to sort due to constraints.
    if (assigned_constrained_ids_) {
      std::sort(r->mutable_child_id()->begin(), r->mutable_child_id()->end());
    }
  } else {
    // Populate and write frame bounds.
    CHECK(ri.raster);
    min_frame = ri.raster->front().first;
    max_frame = ri.raster->back().first;
  }

  // Set bounds and save for next level.
  r->set_start_frame(min_frame);
  r->set_end_frame(max_frame);
  (*curr_bound_map)[ri.index] = std::make_pair(min_frame, max_frame);
}

}  // namespace segmentation.
