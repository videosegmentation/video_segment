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

#ifndef VIDEO_SEGMENT_SEGMENTATION_REGION_SEGMENTATION_GRAPH_H__
#define VIDEO_SEGMENT_SEGMENTATION_REGION_SEGMENTATION_GRAPH_H__

#include "base/base.h"
#include "segmentation/segmentation_common.h"
#include "segmentation/segmentation_graph.h"

namespace segmentation {

// Maps a constraint id to list of region ids that get assigned this constraint.
typedef std::unordered_map<int, std::vector<int>> Skeleton;

// Implements hierarchical agglomerative clustering for graph initialized from
// RegionInfoLists.
class RegionAgglomerationGraph {
 public:
  // Internal classes, exposed for passing information between graphs for different
  // hierarchy levels.

  // An edge is always ordered by region ids, smallest first.
  struct Edge {
    inline Edge(int r1, int r2) {
      if (r1 < r2) {
        region_1 = r1;
        region_2 = r2;
      } else {
        region_1 = r2;
        region_2 = r1;
      }
    }

    bool operator==(const Edge& rhs) const {
      return region_1 == rhs.region_1 && region_2 == rhs.region_2;
    }

    int region_1 = 0;
    int region_2 = 0;
  };

  // Hashes edges (on average just one edge per hash bucket when average neighbors
  // per region is computed).
  struct EdgeHasher {
    EdgeHasher(int neighbors_per_region_ = 23)
        : neighbors_per_region(neighbors_per_region_) {
    }

    size_t operator()(const Edge& edge) const {
      return hasher(edge.region_1 * neighbors_per_region +
                    edge.region_2 % neighbors_per_region);
    }

    int neighbors_per_region;
   private:
    std::hash<int> hasher;
  };

  // Maps edge to its corresponding weight.
  typedef std::unordered_map<Edge, float, EdgeHasher> EdgeWeightMap;
 public:
  // Creates RegionAgglomerationGraph with edge weight domain [0, max_weight]
  // partitioned into num_buckets (allocates one more bucket for virtual edges).
  // Specify RegionDistance to evaluate distance between regions.
  RegionAgglomerationGraph(float max_weight,
                           int num_buckets,
                           const RegionDistance* distance);

  // Add regions and edges from RegionInfoList. If weight_map is specified, instead of
  // computing edge distance between regions from passed RegionDistance, cached weights
  // in weight_map are used.
  // Note that passed RegionInformations are not copied but stored internally as const
  // pointers. Therefore lifetime of passed RegionInfoList has to be at least as long as
  // results are obtained.
  void AddRegionEdges(const RegionInfoList& list,
                      const EdgeWeightMap* weight_map);  // optional.

  // Same as above, but expects constraint id (or -1 if unconstrained) for EACH
  // region in region_list (in order).
  // Also requires skeleton, i.e. map of constraint id to all neighboring regions.
  // TODO(grundman): Skeleton over-constrains, make this virtual edges as for
  // over-segmentation.
  // Add a set of virtual regions (frame slices? pre-segment, ie. merge all regions with
  // the same constraint) that are not output (keep descriptors?)
  // Add virtual edges between virtual and real regions (using direct neighbors?)
  void AddRegionEdgesConstrained(const RegionInfoList& region_list,
                                 const EdgeWeightMap* weight_map,
                                 const std::vector<int>& constraint_ids,
                                 const Skeleton& skeleton);

  // Performs iteratively lowest cost merges until number of regions, drops below
  // fractional cutoff_fraction (w.r.t. initial regions added). Due to constraints
  // the cutoff_fraction is only reached approximately.
  // Returns actual number of performed merges.
  int SegmentGraph(bool merge_rasterization, float cutoff_fraction);

  // Reads out segmentation result. Also returns edge weights between resulting regions
  // (to be used during initialization of the next level). 
  void ObtainSegmentationResult(RegionInfoList* prev_level,
                                RegionInfoList* curr_level,
                                EdgeWeightMap* weight_map);

 protected:
  // Adds a new region with specified id and constraint.
  void AddRegion(int id, int constraint, int size, const RegionInformation* info);
  void AddRegionEdgesImpl(const RegionInfoList& region_list,
                          const std::vector<int>& constraint_ids,
                          const EdgeWeightMap* weight_map);

  // Inserts edge into corresponding bucket and creates entry in EdgePositionMap.
  // Does not check if dual edge is already present (graph edges are undirected).
  // Returns if edge is mergable, i.e. was inserted into a bucket in edge_buckets_.
  bool AddEdge(int region_id_1, int region_id_2, float weight);

  // Removes all edges between region_id and its neighbors (specified by neighbor_ids).
  // Removed neighbors are added to output parameter removed_neighbors if not already
  // present or equal to incident_region_id.
  inline void RemoveNeighboringEdges(int region_id,
                                     const std::vector<int>& neighbor_ids,
                                     int incident_region_id,
                                     std::vector<int>* removed_neighbors);

  struct Region {
    Region(int id_,
           int constraint_id_,
           int sz_,
           const RegionInformation* region_info_) : id(id_),
                                                    constraint_id(constraint_id_),
                                                    sz(sz_),
                                                    region_info(region_info_) {
    }

    int id = 0;
    int constraint_id = -1;
    int sz = 0;
    // Points to external or merged_info.
    const RegionInformation* region_info;
    // Only set after a successful merge.
    std::unique_ptr<RegionInformation> merged_info;
  };

  // Returns representative for id (with path compression).
  Region* GetRegion(int region_id);

  // Merge two regions. Re-evalutates all edges with remaining neighbors and returns
  // minimum edge weight (w.r.t. neighboring regions). Invalidates all iterators to
  // edges connecting rep_1 or rep_2 with other regions.
  // If merge_rasterization is set, rasterizations are merged as well.
  float MergeRegions(Region* rep_1, Region* rep_2, bool merge_rasterization);

  // Returns true if not more than one region is constrained.
  inline bool AreRegionsMergable(const Region& region_1,
                                 const Region& region_2) {
      return region_1.constraint_id < 0 ||
             region_2.constraint_id < 0 ||
             region_1.constraint_id == region_2.constraint_id;
  }


 private:
  // Maximum edge weight and scale for number of buckets.
  float max_weight_  = 1.0f;
  int num_buckets_ = 0;
  float edge_scale_ = 1.0f;  // Scale applied to edge weight to yield bucket idx.

  // Distance used to evaluate region distances.
  const RegionDistance* distance_;

  // List of edges in buckets ordered by edge weight. Each bucket represents the same
  // edge weight. List is used to have iterators that don't invalidate during
  // inserts/deletes.
  typedef std::vector<std::list<Edge>> EdgeBuckets;
  EdgeBuckets edge_buckets_;   // size = num_buckets + 1 (last bucket for virtual
                               //                         edges).

  // Stores bucket index and iterator to list entry for an edge.
  // If the edge does not have a position (e.g. non-mergable edges), iter_ points
  // to list<Edge>::end().
  struct EdgePosition {
    EdgePosition() = default;
    EdgePosition(std::list<Edge>::iterator iter_, int bucket_)
      : iter(iter_), bucket(bucket_) {
    }

    std::list<Edge>::iterator iter;
    int bucket = -1;
  };

  // Maps an edge to its corresponding position. Does not store positions of virtual
  // edges.
  typedef std::unordered_map<Edge, EdgePosition, EdgeHasher> EdgePositionMap;
  EdgePositionMap edge_position_map_;

  // Stores all regions of this graph.
  std::vector<Region> regions_;

  // Kept from SegmentGraph for ObtainSegmentationResult.
  bool merge_rasterization_;
};

}  // namespace segmentation.

#endif // VIDEO_SEGMENT_SEGMENTATION_REGION_SEGMENTATION_GRAPH_H__
