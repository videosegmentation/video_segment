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

#ifndef VIDEO_SEGMENT_SEGMENTATION_SEGMENTATION_GRAPH_H__
#define VIDEO_SEGMENT_SEGMENTATION_SEGMENTATION_GRAPH_H__

#include "base/base.h"

#include <numeric>

#include "segmentation/segmentation_common.h"

namespace segmentation {

// FastSegmentationGraph is templated with traits/policy class that are required to
// supply the following information (see pixel_distance.h for specific implementations).
//
// class SegmentationTraits {
//   // Actual descriptor size per pixel.
//   static constexpr int region_descriptor_size();
//   // Size of actual pixel descriptor input (to be converted to region descriptor.
//   static constexpr int pixel_descriptor_size();
//
//   // Regions with distance below the specified threshold are allowed to merge.
//   float MergeDistanceThreshold() const { return .1 }
//
//   // Constrained regions above the specified threshold are split.
//   float SplitDistanceThreshold() const { return .15 }
//
//   // Return distance in interval [0, 1]   // not tested against.
//   inline float DescriptorDistance(const float* lhs,
//                                   const float* rhs,
//                                   float edge_distance) const;
//
//   // Merge descriptor lhs (region has size lhs_size) into descriptor rhs_dst
//   // (of size rhs_size).
//   inline void MergeDescriptor(int lhs_size,
//                               const float* lhs,
//                               int rhs_size,
//                               float* rhs_dst) const;
//
//  // Initializes the region descriptor of size region_descriptor_size() based on the
//  // pixel descriptor of size pixel_descriptor_size().
//  inline void InitializeDescriptor(const float* pixel_descriptor,    
//                                   float* region_descriptor) const; 
//
//  // Called if for two regions descriptor distance is above MergeDistanceThreshold.
//  // In that case return value of this function is used to determine if the merge test
//  // is considered to be failed. Once a region's merge test has failed, it is not
//  // considered for merges again!
//  // Common implementations are logical OR or AND of the *_above_min_region_sz flags.
//  // Can take actual descriptor into account as well (passed).
//  inline bool FlagMergeTestFailed(const float* lhs_desc,
//                                  bool lhs_above_min_region_sz,
//                                  const float* rhs_desc,
//                                  bool rhs_above_min_region_sz) const;
//
// };

// Fast O(n) implementation that replaces sorting stage of with bucket-sorting.
// Edge weight domain [0, max_weight] is partioned into equally sized buckets.
// In addition, each bucket representing a specific weight is partioned into a bucket
// list. This enables creation of the graph in parallel, as well as partial segmentation.
// 
// General algorithm:
// Single-link agglomerative clustering, i.e. process edges from smallest to largest.
// Stopping criterion: descriptor distance is above MergeDistanceThreshold()
//
// Support for constraints: Regions with different constraints are never merged,
//                          regions with same constraints are merged if descriptor
//                          distance is below SplitDistanceThreshold. Otherwise,
//                          region constraints are reset (depending on size, only
//                          the smaller region or both if of similar size).
//
// Example usage:
// Assuming 
// class ConcreteGraph : public FastSegmentationGraph<Traits> { };
// ConcreteGraph graph;
// 
// // Add regions.
// for (...) { graph.AddRegion(id, size, -1  /* no constraint */ ); }
// // Or:
// for (...) { graph.AddRegionWithDescriptor(id, size, -1,   // no constraint
//                                           ptr_to_pixel_descriptor); }
// 
// // Reserve nodes and edges based on upper bound w.r.t. regions.
// graph.ReserveNodes(num_nodes);
// graph.ReserveEdges(num_nodes * average_connections);
// 
// // Connect regions:
// for (...) { graph.AddEdge(region_id_1, region_id_2, weight); 
//
// // For constraint segmentation: Label some regions are virtual, i.e. their constraints
// // will not be reset.
// graph.AddVirtualNodeInterval(start_region_id, end_region_id);
//
// // Run actual segmentation.
// graph.SegmentGraph(min_region_size,   // Regions will have at least min_region_size
//                    true,              // Merge regions with the same constraint
//                    nullptr);          // Process all edges.
//
// // Get result. 
// RegionInfoList regions;       // Hold's created regions.
// RegionInfoPtrMap region_map;  // Map from each region's representative to corresponding
//                               // RegionInformation.
// graph.AssignRegionIdAndDetermineNeighbors(&region_list, &map);
template <class SegTraits>
class FastSegmentationGraph {
public:
  // Adds a new region of specified size. Region must be created in order, i.e.
  // the n'th call of this function, expects region_id == n.
  // Pass constraint_id >= 0 to constrain region to previous id or -1 for no constraint.
  inline void AddRegion(int region_id, int size, int constraint_id) {
    DCHECK_EQ(region_id, regions_.size());
    regions_.push_back(Region(region_id, size, constraint_id));
  }

  // Same as above but creates region with descriptor of
  // size SegTraits::pixel_descriptor().
  inline void AddRegionWithDescriptor(int region_id,
                                      int size,
                                      int constraint_id,
                                      float* descriptor) {
    regions_.push_back(Region(region_id, size, constraint_id, descriptor, seg_traits_));
  }

  // Connects specified regions with edge of weight weight.
  void AddEdge(int region_id_1, int region_id_2, float weight) {
    AddEdge(region_id_1, region_id_2, weight, 0);   // only one bucket list by standard.
  }

  // Same as above, but places the Edge in the specified bucket list.
  inline void AddEdge(int region_id_1, int region_id_2, float weight, int bucket_list) {
    const int bucket_index = (int)(std::min<float>(num_buckets_, weight * scale_));
    DCHECK_LT(bucket_list, bucket_lists_.size());
    bucket_lists_[bucket_list][bucket_index].push_back(Edge(region_id_1, region_id_2));
  }

  // Labels nodes in interval [start_idx, end_idx) as virtual. Virtual nodes constraints
  // are never reset. They mainly serve to establish correct temporal connections.
  // NOTE: All regions in the specified interval need to be regions with constraints!
  inline void AddVirtualNodeInterval(int start_idx, int end_idx) {
    virtual_nodes_.push_back(std::make_pair(start_idx, end_idx));
  }

  void ReserveNodes(int num_nodes);
  void ReserveEdges(int num_edges);

  // Performs the actual segmentation. Considers only edges in the specified bucket lists.
  void SegmentGraph(int min_region_size,
                    bool force_constraints,
                    const std::vector<int>* bucket_list_ids);   // Optional.

  // Read out results: Create RegionInformation for each segmented region and returns
  // a map that maps each regions representative id to the corresponding
  // RegionInformation.
  // Sets constraints as well as neighboring information.
  void AssignRegionIdAndDetermineNeighbors(RegionInfoList* region_list,
                                           RegionInfoPtrMap* map) {
    AssignRegionIdAndDetermineNeighbors(region_list, nullptr, map);
  }

  // Same as above, but limits determination of neighbors to specified bucket lists.
  void AssignRegionIdAndDetermineNeighbors(
      RegionInfoList* region_list,
      const std::vector<int>* neighbor_eval_bucket_list_ids,
      RegionInfoPtrMap* map);

  int NumBucketLists() const { return bucket_lists_.size(); }

  static int SizeOfEdge() { return sizeof(Edge); }
  static int SizeOfRegion() { return sizeof(Region); }
protected:
  // Don't use this class directly, use to implement FastSegmentationGraph's for concrete
  // use case.
  // Create FastSegmentationGraph discretizing the range of weights [0, max_weight] into
  // num_buckets. 
  // Edges can be added to different lists of buckets. During segmentation, only
  // a specific range of buckets can then be processed.
  FastSegmentationGraph(float max_weight,
                        int buckets,
                        const SegTraits& seg_traits,
                        int num_bucket_lists = 1);

  // Costly implementation test function. Checks for regions that are present but don't
  // edges with other regions (isolated regions). Those should not exists. Use only for
  // debugging purposes.
  bool CheckForIsolatedRegions();

  // Merges regions with the same constraint if descriptor distance is below
  // MergeThresholdDistance or splits them if needed.
  void MergeConstrainedRegions();

  // Traverses all regions, replacing the id of each region with the representative
  // id, therefore flatting the union find. If separate_representatives is set, each
  // representative will be re-created with an id >= regions_.size, therefore represent-
  // atives and nodes are separated. This should only be applied after some initial
  // segmentation otherwise too many representatives have to be re-created. This is
  // helpful to safely implement functions that carve parts away from regions to merge
  // them with other parts, without having to worry about overwriting representatives
  // or vital connections of the internal union find structure.
  // NOTE: If separate_representatives is set, descriptors are not preserved.
  void FlattenUnionFind(bool separate_representatives);

  // Prints statistics about how bucket lists are filled.
  void PrintBucketListStats();

protected:
  static constexpr int descriptor_size() { return SegTraits::region_descriptor_size(); };

  struct Edge {
    Edge (int r1, int r2) : region_1(r1), region_2(r2) {}
    int region_1;
    int region_2;
  };

  typedef std::vector<Edge> EdgeList;

  struct Region {
    Region(int _id, int _sz, int _constraint)
        : my_id(_id), sz(_sz), constraint_id(_constraint) {
    }

    Region(int _id, int _sz, int _constraint, float* _descriptor, const SegTraits& traits)
        : my_id(_id), sz(_sz), constraint_id(_constraint) {
      traits.InitializeDescriptor(_descriptor, descriptor.data());
    }

    Region() = default;

    int my_id = -1;           // id of representative of region (union find structure).
    int sz = 0;               // Size in pixels.
    int constraint_id = -1;   // unconstrained
    bool region_finalized = false;   // True, if region does not accept any more merges.
    std::array<float, descriptor_size()> descriptor;
  };

  // Returns representative for specified start_id.
  // GetRegion performs path compression, altering the union find structure during lookup.
  inline Region* GetRegion(int start_id);

  // Return if regions can be considered mergable based on constraints.
  inline bool AreRegionsMergable(const Region* lhs, const Region* rhs) {
    return (lhs->constraint_id < 0 ||
            rhs->constraint_id < 0 ||  // one is unconstraint.
            lhs->constraint_id == rhs->constraint_id);
  }

  // Merges two regions. Returns merged region (either pointer to rep_1 or rep_2,
  // based on size).
  inline Region* MergeRegions(Region* rep_1, Region* rep_2);

  const float max_weight_ = 0;
  const int num_buckets_ = 0;

  // Scale to be applied to each edge weight to map to bucket index.
  float scale_ = 1.0f;

  // Holds actual regions.
  std::vector<Region> regions_;

  // List of intervals denoting virtual nodes. Left bound is inclusive,
  // right bound exclusive.
  // Virtual nodes constraints are never reset.
  std::vector<std::pair<int, int>> virtual_nodes_;

  // Edge buckets are partioned into BucketLists, e.g. to separate different kinds
  // of connections.
  // Indexed via: [bucket_list][bucket index == edge weight]
  // Each bucket is of actual size num_buckets + 1, where last bucket is used
  // for virtual edges (not used during segmentation, only to estabilish neighboring
  // information).
  std::vector<std::vector<EdgeList>> bucket_lists_;

  bool flattened_ = false;

  SegTraits seg_traits_;
};

template <class SegTraits>
FastSegmentationGraph<SegTraits>::FastSegmentationGraph(float max_weight,
                                                        int num_buckets,
                                                        const SegTraits& seg_traits,
                                                        int num_bucket_lists)
    : max_weight_(max_weight),
      num_buckets_(num_buckets),
      seg_traits_(seg_traits) {
  bucket_lists_.resize(num_bucket_lists);
  for (int i = 0; i < num_bucket_lists; ++i) {
    // Last bucket is used for hard constraints (no-merges).
    bucket_lists_[i].resize(num_buckets + 1);
  }

  scale_ = num_buckets / (max_weight + 1e-6f);
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::SegmentGraph(
    int min_region_size,
    bool force_constraints,
    const std::vector<int>* bucket_list_ids) {
  // Run over buckets.
  // Record edges not considered for merging.
  LOG(INFO) << "Segmenting graph ...";
  int bucket_idx = 0;
  const float inv_scale = 1.0 / scale_;

  // Stat counters.
  int num_forced_merges = 0;
  int num_regular_merges = 0;
  int num_small_region_merges = 0;

  // Use all buckets if not specified.
  std::vector<int> all_bucket_lists;
  if (bucket_list_ids == nullptr) {
    all_bucket_lists.resize(bucket_lists_.size());
    std::iota(all_bucket_lists.begin(), all_bucket_lists.end(), 0);
    bucket_list_ids = &all_bucket_lists;
  }

  const float merge_distance_threshold = seg_traits_.MergeDistanceThreshold();
  const float split_distance_threshold = seg_traits_.SplitDistanceThreshold();

  // Traverse buckets in order of increasing edge weight.
  for (int bucket_idx = 0; bucket_idx < num_buckets_; ++bucket_idx) {
    const float weight = bucket_idx * inv_scale;
    for (int bucket_list_idx : *bucket_list_ids) {
      // Edges that were not used for merging. Retain to compute neighboring information.
      EdgeList remaining_edges;

      // Traverse edges in this bucket list.
      for (const auto& e : bucket_lists_[bucket_list_idx][bucket_idx]) {
        Region* rep_1 = GetRegion(e.region_1);
        Region* rep_2 = GetRegion(e.region_2);

        // Region rep's are different -> merge?
        if (rep_1 == rep_2) {
          continue;
        }

        // Regular test only if at least one node is unconstrained.
        if (rep_1->constraint_id < 0 || rep_2->constraint_id < 0) {
          // Only test if neither region is finalized, otherwise regions are not
          // considered again for merge.
          if (!rep_1->region_finalized && !rep_2->region_finalized) {
            const float desc_distance = seg_traits_.DescriptorDistance(
                rep_1->descriptor.data(), rep_2->descriptor.data(),  weight);

            if (desc_distance < merge_distance_threshold) {
              MergeRegions(rep_1, rep_2);
              ++num_regular_merges;
            } else {
              //if (seg_traits_.FlagMergeTestFailed(rep_1->descriptor.data(),
              //                                         rep_1->sz >= min_region_size,
               //                                        rep_2->descriptor.data(),
                //                                       rep_2->sz >= min_region_size)) {
              // Merge test failed, determine if regions should be flagged.
              rep_1->region_finalized = true;
              rep_2->region_finalized = true;
            }
          }

          // If regions are finalized or got finalized by above edge evaluation,
          // only perform the merge if minimum region size is not achieved yet.
          if (rep_1->region_finalized || rep_2->region_finalized) {
            if (rep_1->sz < min_region_size || rep_2->sz < min_region_size) {
              MergeRegions(rep_1, rep_2);
              ++num_small_region_merges;
            } else {
              // Keep edge for neighboring information.
              remaining_edges.push_back(e);
            }
          }
        } else if (rep_1->constraint_id == rep_2->constraint_id) {
          // Regions are constraint to the same id, check if they should be merged
          // or unconstrained.
          const float desc_distance = seg_traits_.DescriptorDistance(
              rep_1->descriptor.data(), rep_2->descriptor.data(), weight);
          if (desc_distance > split_distance_threshold) {
            // Unconstrain the smaller of both regions or both if of similar size.
            if (rep_1->sz < rep_2->sz * 0.3) {
              rep_1->constraint_id = -1;
            } else if (rep_2->sz < rep_1->sz * 0.3) {
              rep_2->constraint_id = -1;
            } else {
              rep_1->constraint_id = -1;
              rep_2->constraint_id = -1;
            }
            remaining_edges.push_back(e);
          } else {
            MergeRegions(rep_1, rep_2);
            ++num_forced_merges;
          }
        } else {
          // Both are constrained but have different ids: never merge!
          remaining_edges.push_back(e);
        }
      }  // end edge traversal.

      bucket_lists_[bucket_list_idx][bucket_idx].swap(remaining_edges);
    }  // end bucket lists ids.
  }  // end bucket idx.

  DCHECK(CheckForIsolatedRegions());

  if (force_constraints) {
    MergeConstrainedRegions();
  }

  const float total_merges = num_forced_merges +
                             num_regular_merges +
                             num_small_region_merges;

  LOG(INFO) << "Total merges: " << total_merges << "\n"
            << "Forced merges " << num_forced_merges << "("
            << num_forced_merges / total_merges * 100.f << "%), "
            << "Regular merges " << num_regular_merges << "("
            << num_regular_merges / total_merges * 100.f << "%), "
            << "Small region merges " << num_small_region_merges << "("
            << num_small_region_merges / total_merges * 100.f << "%)";
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::AssignRegionIdAndDetermineNeighbors(
  RegionInfoList* region_list,
  const std::vector<int>* neighbor_eval_bucket_list_ids,
  RegionInfoPtrMap* map) {
  // Traverse all remaining edges.
  int max_region_id = 0;

  // Convert bucket_list_idx to std::vector of flags.
  std::vector<bool> consider_for_neighbor_info(
    bucket_lists_.size(), neighbor_eval_bucket_list_ids == nullptr ? true : false);

  if (neighbor_eval_bucket_list_ids) {
    for (int nb : *neighbor_eval_bucket_list_ids) {
      if (nb > bucket_lists_.size()) {
        LOG(WARNING) << "Passed neighbor_eval_bucket_list_ids is out of bounds";
      } else {
        consider_for_neighbor_info[nb] = true;
      }
    }
  }

  auto insert_region =
    [&max_region_id, region_list, map](const Region* region) -> RegionInformation* {
      RegionInformation* new_region_info = new RegionInformation;
      new_region_info->index = max_region_id++;
      new_region_info->size = region->sz;
      new_region_info->constrained_id = region->constraint_id;

      // Add to list and hash_map.
      region_list->emplace_back(new_region_info);
      map->insert(std::make_pair(region->my_id, new_region_info));
      return new_region_info;
    };

  // Also process last bucket (virtual edges).
  for (int bucket_idx = 0; bucket_idx <= num_buckets_; ++bucket_idx) {
    for (int bucket_list_idx = 0; bucket_list_idx < bucket_lists_.size();
        ++bucket_list_idx) {
      const bool neighbor_eval = consider_for_neighbor_info[bucket_list_idx];
      for (const auto& e : bucket_lists_[bucket_list_idx][bucket_idx]) {
        const Region* r1 = GetRegion(e.region_1);
        const Region* r2 = GetRegion(e.region_2);

        const int r1_id = r1->my_id;
        const int r2_id = r2->my_id;

        if (r1_id == r2_id) {
          continue;
        }

        RegionInformation* r1_info;
        RegionInformation* r2_info;

        // Find region 1 in hash_map, insert if not present.
        auto hash_iter = map->find(r1->my_id);
        if (hash_iter == map->end()) {  // Does not exists yet --> insert.
          r1_info = insert_region(r1);
        } else {
          r1_info = hash_iter->second;
        }

        // Same for region 2.
        hash_iter = map->find(r2->my_id);
        if (hash_iter == map->end()) {
          r2_info = insert_region(r2);
        } else {
          r2_info = hash_iter->second;
        }

        if (neighbor_eval) {
          InsertSortedUniquely(r2_info->index, &r1_info->neighbor_idx);
          InsertSortedUniquely(r1_info->index, &r2_info->neighbor_idx);
        }
      }  // for edge loop.
    } // end bucket list loop.
  } // end bucket loop.

  if (max_region_id == 0) {
    LOG(WARNING) << "AssignRegionIdAndDetermineNeighbors: No boundary edges found. "
                 << "This results in only one region and is probably not what was "
                 << "desired. Check passed edge weights.";

    // Add one region so algorithm can proceed.
    const Region* r1 = GetRegion(0);    // Any region will do.
    insert_region(r1);
  }
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::ReserveEdges(int num_edges) {
  // Size each bucket to 10% of the average load.
  int edges_per_bucket = num_edges / num_buckets_ / bucket_lists_.size() / 10;
  for (auto& bucket_list : bucket_lists_) {
    for (auto& bucket : bucket_list) {
      bucket.reserve(edges_per_bucket);
    }
  }
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::ReserveNodes(int num_nodes) {
  regions_.reserve(num_nodes);
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::FlattenUnionFind(bool separate_representatives) {
  if (flattened_) {
    LOG(ERROR) << "Graph is already flattened. Ignoring request.";
    return;
  }

  flattened_ = true;

  const int region_offset = regions_.size();
  int new_region_id = region_offset;

  if (separate_representatives) {
    for (int i = 0; i < region_offset; ++i) {
      Region* r = GetRegion(i);
      int flattened_id = r->my_id;
      if (flattened_id < region_offset) {    // separate first.
        // Map representative.
        r->my_id = new_region_id;
        flattened_id = new_region_id;
        regions_.push_back(Region(new_region_id++, r->sz, r->constraint_id));
        // Note: push back might invalidate Region r!
      }

      regions_[i].my_id = flattened_id;
    }
    LOG(INFO) << "Inserted " << new_region_id - region_offset
              << " new representatives.";
  } else {
    for (auto& region : regions_) {
      region.my_id = GetRegion(region.my_id)->my_id;
    }
  }
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::PrintBucketListStats() {
  const int num_bucket_lists = bucket_lists_.size();
  for (int k = 0; k < num_bucket_lists; ++k) {
    int sum = 0;
    for (int l = 0; l < num_buckets_; ++l) {
      sum += bucket_lists_[k][l].size();
    }

    // Print out first 20.
    const float denom = 1.0f / sum;
    std::ostringstream s;
    s << "Bucket lists idx #" << k << " : " << " sum : " << sum << " .... ";
    for (int l = 0; l < 20; ++l) {
      s << (float)bucket_lists_[k][l].size() * denom << " ";
    }
    LOG(INFO) << s.str();
  }
}

template <class SegTraits>
inline typename FastSegmentationGraph<SegTraits>::Region*
FastSegmentationGraph<SegTraits>::GetRegion(int id) {
  DCHECK_GE(id, 0);
  DCHECK_LT(id, regions_.size());

  Region* r = &regions_[id];
  const int parent_id = r->my_id;
  Region* parent = &regions_[parent_id];

  if (parent->my_id == parent_id) {
    return parent;
  } else {
    // Path compression.
    parent = GetRegion(parent_id);
    r->my_id = parent->my_id;
    return parent;
  }
}

template <class SegTraits>
inline typename FastSegmentationGraph<SegTraits>::Region*
FastSegmentationGraph<SegTraits>::MergeRegions(Region* rep_1, Region* rep_2) {
  Region* merged;
  Region* other;

  // Assign smaller region new id.
  if (rep_1->sz > rep_2->sz) {
    merged = rep_1;
    other = rep_2;
  } else {
    merged = rep_2;
    other = rep_1;
  }

  seg_traits_.MergeDescriptor(other->sz,
                              other->descriptor.data(),
                              merged->sz,
                              merged->descriptor.data());

  // Update area.
  merged->sz += other->sz;

  // Constraint max: if one constraint the other not -> act as sticky constraint.
  merged->constraint_id = std::max(rep_1->constraint_id, rep_2->constraint_id);

  // Point to larger region.
  other->my_id = merged->my_id;

  return merged;
}

template <class SegTraits>
void FastSegmentationGraph<SegTraits>::MergeConstrainedRegions() {
  // Maps contraint id to corresponding representative.
  std::unordered_map<int, int> constraint_to_region_map;

  // Indicates ranges of virtual nodes.
  std::vector<std::pair<int, int>> virtual_nodes(virtual_nodes_);
  // Add begin and end as dummy ranges.
  virtual_nodes.push_back(std::make_pair(0, 0));
  virtual_nodes.push_back(std::make_pair(regions_.size(), regions_.size()));
  sort(virtual_nodes.begin(), virtual_nodes.end());

  // First process non-virtual nodes.
  // Non-virtual nodes are in-between nodes of virtual_nodes.
  const float split_distance_threshold = seg_traits_.SplitDistanceThreshold();
  for (int k = 1; k < virtual_nodes.size(); ++k) {
    for (auto region = regions_.cbegin() + virtual_nodes[k - 1].second,
         end_region = regions_.cbegin() + virtual_nodes[k].first;
         region != end_region;
         ++region) {
      // Only process constraint regions here.
      if (region->constraint_id < 0) {
        continue;
      }

      Region* my_rep = GetRegion(region->my_id);
      auto pos = constraint_to_region_map.find(my_rep->constraint_id);
      if (pos == constraint_to_region_map.end()) {
        // No representative yet.
        constraint_to_region_map.insert(std::make_pair(my_rep->constraint_id,
                                                       my_rep->my_id));
      } else {
        // Region for this constraint already exist.
        Region* constraint_rep = GetRegion(pos->second);
        if (constraint_rep != my_rep) {
          // Different from current region ->
          // We need to merge or separate the two regions.
          const float distance = seg_traits_.DescriptorDistance(
              my_rep->descriptor.data(),
              constraint_rep->descriptor.data(),
              1.0f);  // virtual edge weight.

          if (distance > split_distance_threshold) {
             if (my_rep->sz < constraint_rep->sz * 0.3) {
               my_rep->constraint_id = -1;
             } else if (constraint_rep->sz < my_rep->sz * 0.3) {
               constraint_rep->constraint_id = -1;
               pos->second = my_rep->my_id;
             } else {
               my_rep->constraint_id = -1;
               constraint_rep->constraint_id = -1;
               constraint_to_region_map.erase(pos);
             }
          } else {
            // Below threshold, merge these regions.
            MergeRegions(my_rep, constraint_rep);
          }
        }
      }
    }
  }

  // Process virtual nodes, this is just to establish correct neighboring information.
  for (int k = 0; k < virtual_nodes.size(); ++k) {
    for (auto region = regions_.cbegin() + virtual_nodes[k].first,
         end_region = regions_.cbegin() + virtual_nodes[k].second;
         region != end_region;
         ++region) {
      CHECK_GE(region->constraint_id, 0);  // Should always be constraint.
      Region* my_rep = GetRegion(region->my_id);
      auto pos = constraint_to_region_map.find(my_rep->constraint_id);
      if (pos == constraint_to_region_map.end()) {
        constraint_to_region_map.insert(std::make_pair(my_rep->constraint_id,
                                                       my_rep->my_id));
      } else {
        Region* constraint_rep = GetRegion(pos->second);
        if (constraint_rep != my_rep) {
          // Never reset, always merge.
          MergeRegions(my_rep, constraint_rep);
        }
      }
    }
  }
}

template <class SegTraits>
bool FastSegmentationGraph<SegTraits>::CheckForIsolatedRegions() {
  // Set of regions connected to an edge.
  std::unordered_set<int> connected_regions;

  // Traverse edges, set incident regions to true.
  for (const auto& bucket_list : bucket_lists_) {
    for (const auto& bucket : bucket_list) {
      for (const auto& e : bucket) {
        const Region* r1 = GetRegion(e.region_1);
        const Region* r2 = GetRegion(e.region_2);
        connected_regions.insert(r1->my_id);
        connected_regions.insert(r2->my_id);
      }
    }
  }

  bool is_constistent = true;
  for (int i = 0; i < regions_.size(); ++i) {
    // Only non-virtual regions should be considered isolated..
    Region* curr_region = GetRegion(i);
    if (curr_region->sz > 0 && connected_regions.find(curr_region->my_id) ==
        connected_regions.end()) {
      LOG(ERROR) << "Region " << curr_region->my_id << " is isolated! constraint: "
                 << curr_region->constraint_id << " sz : " << curr_region->sz;
      is_constistent = false;
    }
  }

  return is_constistent;
}

}  // namespace segmentation.

#endif // VIDEO_SEGMENT_SEGMENTATION_SEGMENTATION_GRAPH_H__
