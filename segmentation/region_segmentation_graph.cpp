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


#include "base/base_impl.h"
#include "segmentation/region_segmentation_graph.h"

namespace segmentation {

RegionAgglomerationGraph::RegionAgglomerationGraph(float max_weight,
                                                   int num_buckets,
                                                   const RegionDistance* distance)
    : max_weight_(max_weight * 1.01f),
      num_buckets_(num_buckets),
      distance_(distance) {
  edge_scale_ = num_buckets * (1.0f / max_weight_);
  edge_buckets_.resize(num_buckets + 1);   // Last bucket used for virtual edges.
}

void RegionAgglomerationGraph::AddRegionEdges(const RegionInfoList& list,
                                              const EdgeWeightMap* weight_map) {
  AddRegionEdgesImpl(list,
                     vector<int>(list.size(), -1),   // all regions are unconstrained.
                     weight_map);
}

void RegionAgglomerationGraph::AddRegionEdgesConstrained(
    const RegionInfoList& list,
    const EdgeWeightMap* weight_map,
    const vector<int>& constraint_ids,
    const Skeleton& skeleton) {
  AddRegionEdgesImpl(list, constraint_ids, weight_map);

  // Add skeleton. Connect all regions with the same constraint via virtual edges
  // so that merges can result in exactly the same one region.
  for (const auto& entry : skeleton) {
    int prev_region_idx = entry.second.front();
    for (auto region_iter = entry.second.begin() + 1;
         region_iter != entry.second.end();
         ++region_iter) {
      // Add to last bucket, virtual edge.
      AddEdge(prev_region_idx, *region_iter, max_weight_ * 2);
      prev_region_idx = *region_iter;
    }
  }
}

int RegionAgglomerationGraph::SegmentGraph(bool merge_rasterization,
                                           float cutoff_fraction) {
  LOG(INFO) << "Segmenting region graph...";
  merge_rasterization_ = merge_rasterization;

  CHECK(cutoff_fraction > 0 &&
        cutoff_fraction <= 1) << "Cutoff fraction expected to be between 0 and 1.";
  // Approximate goal of merges.
  int num_merges = regions_.size() * (1.0f - cutoff_fraction);

  // Anticipated number of forced merges from constraining skeleton.
  // TODO: Evaluate how good this approximation is!
  const int constraint_merges = edge_buckets_.back().size() * cutoff_fraction;

  // Reduce num_merges by additional forced merges due to constraints.
  const int requested_merges = num_merges;
  num_merges -= constraint_merges;

  // Guarantee that at least one region remains.
  num_merges = std::min<int>(num_merges, regions_.size() - 1);

  // Determines lowest, non-empty bucket.
  int lowest_bucket = 0;
  while (lowest_bucket < num_buckets_ && edge_buckets_[lowest_bucket].empty()) {
    ++lowest_bucket;
  }

  int actual_merges = 0;
  for (int merge = 0; merge < num_merges; ++merge) {
    if (lowest_bucket >= num_buckets_) {
      break;
    }

    // Grab lowest cost edge -> perform merge.
    bool merge_performed = false;

    while (!merge_performed) {
      DCHECK_LT(lowest_bucket, num_buckets_);
      DCHECK(!edge_buckets_[lowest_bucket].empty());
      std::list<Edge>::iterator first_edge = edge_buckets_[lowest_bucket].begin();

      // TODO: Redundant. Remove.
      Region* region_1 = GetRegion(first_edge->region_1);
      Region* region_2 = GetRegion(first_edge->region_2);
      DCHECK_EQ(region_1->id, first_edge->region_1);
      DCHECK_EQ(region_2->id, first_edge->region_2)
          << region_2->id << " vs " << first_edge->region_2;
      DCHECK_NE(region_1, region_2) << "Internal edge. Should not be present.";

      if (!AreRegionsMergable(*region_1, *region_2)) {
        // If regions have different constraints, remove from buckets and flag in
        // EdgePositionMap by pointing to end.
        CHECK_EQ(edge_position_map_[*first_edge].bucket, lowest_bucket);
        edge_position_map_[*first_edge].iter = edge_buckets_[lowest_bucket].end();
        first_edge = edge_buckets_[lowest_bucket].erase(first_edge);
      } else {
        int min_bucket =
            MergeRegions(region_1, region_2, merge_rasterization) * edge_scale_;
        ++actual_merges;
        // Get next minimum edge.
        if (min_bucket < lowest_bucket) {
          lowest_bucket = min_bucket;
          break;
        }

        first_edge = edge_buckets_[lowest_bucket].begin();  // Edge erased, point to
                                                            // next one in this bucket.
        merge_performed = true;
      }

      if (first_edge == edge_buckets_[lowest_bucket].end()) {
        // Advance to next occupied bucket.
        do {
          ++lowest_bucket;
        } while (lowest_bucket < num_buckets_ && edge_buckets_[lowest_bucket].empty());
        // Delay virtual edge merges to the loop below.
        if (lowest_bucket >= num_buckets_) {
          break;
        }
      }
    }
  }

  // Perform forced merges from skeleton (last bucket = virtual edges).
  // TODO: Skeleton and other forced merges above a specific threshold should not
  // be performed.
  for (auto edge : edge_buckets_.back()) {
    // Necessary to lookup regions here as they are not in the EdgePositionMap.
    Region* region_1 = GetRegion(edge.region_1);
    Region* region_2 = GetRegion(edge.region_2);
    if (region_1 != region_2) {
      CHECK(region_1->constraint_id == region_2->constraint_id &&
            region_1->constraint_id >= 0);
      MergeRegions(region_1, region_2, merge_rasterization);
      ++actual_merges;
      ++num_merges;
    }
  }

  // LOG(INFO) << "Requested merges: " << requested_merges
  //           << ", performed merges: " << num_merges;

  LOG(INFO) << "... done. ";
  return actual_merges;
}


// Assign ids and establish parent - child relationship.
void RegionAgglomerationGraph::ObtainSegmentationResult(RegionInfoList* prev_level,
                                                        RegionInfoList* curr_level,
                                                        EdgeWeightMap* weight_map) {
  int child_idx = 0;
  int next_assigned_idx = 0;

  // Records which representatives have been assigned ids yet (in that case map points
  // to the created RegionInformation).
  std::unordered_map<int, RegionInformation*> assigned_results;
                                         // assigned ids already.
  std::vector<int> representative_id;    // Representative for each region in curr_level
                                         // in order.

  // Assign ids and RegionInformation to results.
  for (auto child_region = prev_level->begin();
       child_region != prev_level->end();
       ++child_region, ++child_idx) {
    // TODO: We don't need child_idx here ...
    CHECK_EQ(child_idx, (*child_region)->index);

    Region* result_region = GetRegion(child_idx);
    DCHECK_EQ(&regions_[result_region->id], result_region);

    // Assign id to segmented resulted if not done yet.
    if (assigned_results.find(result_region->id) == assigned_results.end()) {
      // Do we require a copy?
      if (result_region->region_info != result_region->merged_info.get()) {
        const RegionInformation& old_info = *result_region->region_info;
        std::unique_ptr<RegionInformation> new_info(old_info.BasicCopy(
              merge_rasterization_));
        result_region->merged_info.swap(new_info);
        result_region->region_info = result_region->merged_info.get();
      }

      DCHECK(result_region->merged_info != nullptr);

      // Assign id, set constraint and allocate child index array.
      result_region->merged_info->index = next_assigned_idx++;
      result_region->merged_info->constrained_id = result_region->constraint_id;
      result_region->merged_info->child_idx.reset(new vector<int>);

      assigned_results[result_region->id] = result_region->merged_info.get();
      curr_level->push_back(std::move(result_region->merged_info));
      representative_id.push_back(result_region->id);
    }

    RegionInformation* result_info = assigned_results[result_region->id];
    DCHECK_GE(result_info->index, 0) << result_info->index;
    DCHECK(result_info->child_idx);
    result_info->child_idx->push_back(child_idx);
    (*child_region)->parent_idx = result_info->index;
  }

  if (weight_map) {
    weight_map->clear();
  }

  // After all representatives have their ids assigned, update neighbors and output
  // already computed edge weights.
  const float inv_edge_scale = 1.0f / edge_scale_;
  for (auto& region : *curr_level) {
    // Map neighbors ids to region_ids.
    vector<int> mapped_neighbors;
    for (int neighbor : region->neighbor_idx) {
      const Region* neighbor_region = GetRegion(neighbor);
      int neighbor_idx = neighbor_region->region_info->index;
      CHECK_GE(neighbor_idx, 0);

      if (weight_map) {
        DCHECK_EQ(region.get(), (*curr_level)[region->index].get());
        Edge graph_edge(representative_id[region->index], neighbor_region->id);
        Edge output_edge(region->index, neighbor_idx);
        (*weight_map)[output_edge] =
            inv_edge_scale * edge_position_map_[graph_edge].bucket;
      }

      InsertSortedUniquely(neighbor_idx, &mapped_neighbors);
    }
    region->neighbor_idx.swap(mapped_neighbors);
  }
}

void RegionAgglomerationGraph::AddRegionEdgesImpl(const RegionInfoList& region_list,
                                                  const vector<int>& constraint_ids,
                                                  const EdgeWeightMap* weight_map) {
  // Allocate nodes.
  const int region_num = region_list.size();
  CHECK_GT(region_num, 0);
  regions_.reserve(region_num);

  // Determine average load for EdgePositionMap.
  int num_neighbors = 0;

  for (const auto& region_ptr : region_list) {
    num_neighbors += region_ptr->neighbor_idx.size();
  }

  int av_neighbors_per_region = std::ceil(num_neighbors * (1.0f / region_list.size()));

  // Reset EdgePositionMap.
  edge_position_map_ = EdgePositionMap(2 * num_neighbors,  // twice the anticipated load.
                                       EdgeHasher(av_neighbors_per_region));

  vector<float> descriptor_distances(distance_->NumDescriptors());

  int region_idx = 0;
  for (const auto& region_ptr : region_list) {
    const RegionInformation& ri = *region_ptr;
    const int curr_id = ri.index;
    DCHECK_LE(curr_id, region_num);
    DCHECK_EQ(curr_id, region_idx);

    AddRegion(curr_id, constraint_ids[region_idx], 1, region_ptr.get());

    // Introduce an edge to each neighbor.
    for (int n : ri.neighbor_idx) {
      // Only add edge if not present yet.
      if (edge_position_map_.find(Edge(curr_id, n)) == edge_position_map_.end()) {
        float weight = 0;
        if (weight_map) {
          // Weight has been previously evaluated.
          const auto weight_iter = weight_map->find(Edge(curr_id, n));
          DCHECK(weight_iter != weight_map->end());
          weight = weight_iter->second;
        } else {
          // Explicitly evaluate distance.
          ri.DescriptorDistances(*region_list[n], &descriptor_distances);
          weight = distance_->Evaluate(descriptor_distances);
        }
        AddEdge(curr_id, n, weight);
      }
    }
    ++region_idx;
  }
}

inline void RegionAgglomerationGraph::AddRegion(int id,
                                                int constraint_id,
                                                int size,
                                                const RegionInformation* ptr) {
  regions_.push_back(Region(id, constraint_id, size, ptr));
}

inline bool RegionAgglomerationGraph::AddEdge(int region_1, int region_2, float weight) {
  int bucket = std::min(num_buckets_, (int)(weight * edge_scale_));
  const Edge e(region_1, region_2);

  // Only store mergable edges.
  auto insert_iter = edge_buckets_[bucket].end();
  bool mergable = AreRegionsMergable(regions_[region_1], regions_[region_2]);
  if (mergable) {
    // Add to end.
    insert_iter = edge_buckets_[bucket].insert(insert_iter, e);
  }

  // Only insert non-virtual edges to EdgePositionMap (no need to lookup virtual edges).
  if (bucket != num_buckets_) {
    auto insert_pos = edge_position_map_.insert(
        std::make_pair(e, EdgePosition(insert_iter, bucket)));

    CHECK_EQ(insert_pos.second, true)
      << "Edge " << region_1 << " to " << region_2 << " exists.";
  } else {
    CHECK(mergable) << regions_[region_1].constraint_id << " and "
                    << regions_[region_2].constraint_id;
  }

  return mergable;
}

inline RegionAgglomerationGraph::Region*
RegionAgglomerationGraph::GetRegion(int region_id) {
  DCHECK_GE(region_id, 0);
  DCHECK_LT(region_id, regions_.size());

  // Note, most regions directly point to root, so this will cause a single region
  // lookup.
  Region* r = &regions_[region_id];
  const int parent_id = r->id;
  Region* parent = &regions_[parent_id];

  if (parent->id == parent_id) {
    return parent;
  } else {
    // Path compression.
    parent = GetRegion(parent_id);
    r->id = parent->id;
    return parent;
  }
}

inline void RegionAgglomerationGraph::RemoveNeighboringEdges(
    int region_id,
    const vector<int>& neighbor_ids,
    int incident_region_id,
    vector<int>* removed_neighbors) {
  for (int n : neighbor_ids) {
    // Map to representative.
    const int neighbor_idx = GetRegion(n)->id;
    // TODO(grundman): remove this lookup.
    // CHECK_EQ(neighbor_idx, n);

    // Remove edge.
    auto edge_position = edge_position_map_.find(Edge(region_id, neighbor_idx));

    // Skip, if edge was already removed in previous iteration (possible due to mapping
    // to representative above.
    if (edge_position == edge_position_map_.end()) {
      continue;
    }

    // Only remove, if not an unmergable edge (otherwise not present in edge_buckets_).
    auto& bucket = edge_buckets_[edge_position->second.bucket];
    if (edge_position->second.iter != bucket.end()) {
      bucket.erase(edge_position->second.iter);
    }

    // Remove from edge position map as well.
    edge_position_map_.erase(edge_position);

    // Skip the two involved regions.
    if (neighbor_idx != incident_region_id) {
      InsertSortedUniquely(neighbor_idx, removed_neighbors);
    }
  }
}


float RegionAgglomerationGraph::MergeRegions(Region* rep_1,
                                             Region* rep_2,
                                             bool merge_rasterization) {
  const RegionInformation& rep_info_1 = *rep_1->region_info;
  const RegionInformation& rep_info_2 = *rep_2->region_info;

  const int region_1_id = rep_1->id;
  const int region_2_id = rep_2->id;

  // General algorithm: Remove all neighboring edges from each region, merge
  //                    representations and then re-evaluate edges and re-insert.
  // Record merged neighbors.
  vector<int> merged_neighbors;
  merged_neighbors.reserve(rep_info_1.neighbor_idx.size() +
                           rep_info_2.neighbor_idx.size());
  RemoveNeighboringEdges(region_1_id,
                         rep_info_1.neighbor_idx,
                         region_2_id,
                         &merged_neighbors);
  RemoveNeighboringEdges(region_2_id,
                         rep_info_2.neighbor_idx,
                         region_1_id,
                         &merged_neighbors);

  // Use region with most merged children as new representative. 
  Region* merged = rep_1->sz > rep_2->sz ? rep_1 : rep_2;
  merged->sz = rep_1->sz + rep_2->sz;
  rep_1->id = merged->id;
  rep_2->id = merged->id;
  merged->constraint_id = std::max(rep_1->constraint_id, rep_2->constraint_id);

  // Create new RegionInformation for merged region.
  std::unique_ptr<RegionInformation> new_info(new RegionInformation());
  new_info->size = rep_info_1.size + rep_info_2.size;
  new_info->neighbor_idx.swap(merged_neighbors);

  // Merge descriptors.
  new_info->MergeDescriptorsFrom(rep_info_1);
  new_info->MergeDescriptorsFrom(rep_info_2);

  // Merge rasterizations, if requested.
  if (merge_rasterization) {
    new_info->raster.reset(new Rasterization3D());
    MergeRasterization3D(*rep_info_1.raster,
                         *rep_info_2.raster,
                         new_info->raster.get());
  }


  // Re-evaluate edge weights of all incident edges.
  const int num_neighbors = new_info->neighbor_idx.size();
  std::vector<const RegionInformation*> neighbor_infos;
  neighbor_infos.reserve(num_neighbors);

  for (int neighbor_idx : new_info->neighbor_idx) {
    neighbor_infos.push_back(regions_[neighbor_idx].region_info);
  }

  // Records final distance for each neighbor.
  vector<float> region_distances(num_neighbors, 0.f);

  auto evaluator_fun = [this, &region_distances, &new_info, &neighbor_infos]
    (const base::BlockedRange& r) -> void {
    std::vector<float> descriptor_distances(distance_->NumDescriptors());
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const RegionInformation& neighbor = *neighbor_infos[i];
      new_info->DescriptorDistances(neighbor, &descriptor_distances);
      region_distances[i] = distance_->Evaluate(descriptor_distances);
    }
  };

  // TODO: Time measurements w.r.t. cutoff value.
  const int parallel_cutoff = 128;

  if (num_neighbors < parallel_cutoff) {
    evaluator_fun(base::BlockedRange(0, num_neighbors));
  } else {
    base::ParallelFor(base::BlockedRange(0, num_neighbors), evaluator_fun);
  }

  // Add edges back to graph, determine minimum added edge weight.
  float min_dist = 1.e6f;
  for (int n_idx = 0; n_idx < num_neighbors; ++n_idx) {
    const int neighbor_idx = new_info->neighbor_idx[n_idx];
    const float region_dist = region_distances[n_idx];
    if (AddEdge(merged->id, neighbor_idx, region_dist)) {
      // Already mapped to index, no GetRegion necessary.
      min_dist = std::min(min_dist, region_dist);
    }
  }

  merged->merged_info.swap(new_info);
  merged->region_info = merged->merged_info.get();
  return min_dist;
}

}  // namespace segmentation.
