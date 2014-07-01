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

#ifndef VIDEO_SEGMENT_SEGMENTATION_DENSE_SEGMENTATION_GRAPH_H__
#define VIDEO_SEGMENT_SEGMENTATION_DENSE_SEGMENTATION_GRAPH_H__

#include "base/base.h"

#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>

#include <thread>

#include "segmentation/dense_seg_graph_interface.h"
#include "segmentation/pixel_distance.h"
#include "segmentation/segmentation_graph.h"

DECLARE_bool(parallel_graph_construction);

namespace segmentation {

// Simplified edge representation used during graph construction.
struct EdgeTuple {
  EdgeTuple() = default;
  EdgeTuple(int node_1_, int node_2_, float weight_)
      : node_1(node_1_), node_2(node_2_), weight(weight_) {
  }

  bool operator<(const EdgeTuple& rhs) {
    return weight < rhs.weight;
  }

  int node_1 = -1;
  int node_2 = -1;
  float weight = 0;
};

// Templated with traits:
// - DistanceTraits (see dense_seg_graph_interface.h)
// - DescriptorTraits (see pixel_distance.h)
template<class DistanceTraits, class DescriptorTraits>
class DenseSegmentationGraph : public DenseSegGraphInterface,
                                      FastSegmentationGraph<DescriptorTraits> {
  typedef typename DistanceTraits::SpatialDistance SpatialDistance;
  typedef typename DistanceTraits::TemporalDistance TemporalDistance;
public:
  // Creates a new dense segmentation graph over the specified domain with max_frames
  // (you cannot add more than max_frames, CHECKED).
  // Spatial edges are put in even bucket lists, temporal in odd ones.
  DenseSegmentationGraph(int frame_width,
                         int frame_height,
                         int max_frames,
                         const DescriptorTraits& desc_traits);

  // Interface functions.
  virtual void InitializeGraph();
  virtual void AddNodesAndSpatialEdges(const AbstractPixelDistance& dist) {
    // Upcast abstract distance to the one specified via traits.
    CHECK(dynamic_cast<const SpatialDistance*>(&dist) != nullptr)
        << "Expected: " << typeid(SpatialDistance).name()
        << " found: " << typeid(dist).name();
    const SpatialDistance& forward_dist = dynamic_cast<const SpatialDistance&>(dist);
    AddNodesAndSpatialEdgesForward(forward_dist);
    ++num_frames_;
    CHECK_LE(num_frames_, max_frames_);
  }

  virtual void AddNodesAndSpatialEdgesConstrained(const AbstractPixelDistance& dist,
                                                  const SegmentationDesc& desc) {
    // Upcast abstract distance to the one specified via traits.
    CHECK(dynamic_cast<const SpatialDistance*>(&dist) != nullptr)
        << "Expected: " << typeid(SpatialDistance).name()
        << " found: " << typeid(dist).name();
    const SpatialDistance& forward_dist = dynamic_cast<const SpatialDistance&>(dist);
    AddNodesAndSpatialEdgesConstrainedForward(forward_dist, desc);
    ++num_frames_;
    CHECK_LE(num_frames_, max_frames_);
  }

  // Adds and connects virtual nodes (e.g. to establish correct neighboring
  // correspondences during constrained segmentation).
  virtual void AddVirtualNodesConstrained(const SegmentationDesc& desc);

  virtual void AddTemporalEdges(const AbstractPixelDistance& dist) {
    // Upcast abstract distance to the one specified via traits.
    CHECK(dynamic_cast<const TemporalDistance*>(&dist) != nullptr)
        << "Expected: " << typeid(TemporalDistance).name()
        << " found: " << typeid(dist).name();
    const TemporalDistance& forward_dist = dynamic_cast<const TemporalDistance&>(dist);
    AddTemporalEdgesForward(forward_dist);
  }

  virtual void AddTemporalFlowEdges(const AbstractPixelDistance& dist,
                                    const cv::Mat& flow) {
    // Upcast abstract distance to the one specified via traits.
    CHECK(dynamic_cast<const TemporalDistance*>(&dist) != nullptr)
        << "Expected: " << typeid(TemporalDistance).name()
        << " found: " << typeid(dist).name();
    const TemporalDistance& forward_dist = dynamic_cast<const TemporalDistance&>(dist);
    AddTemporalFlowEdgesForward(forward_dist, flow);
  }

  // Create virtual edges (without edge weight, not used for segmentation, but to
  // determine neighboring information).
  virtual void AddTemporalVirtualEdges();
  virtual void AddTemporalFlowVirtualEdges(const cv::Mat& flow);

  virtual void FinishBuildingGraph();
  virtual void SegmentGraphSpatially();
  virtual void SegmentFullGraph(int min_region_size, bool force_constraints);

  virtual void DetermineNeighborIds(RegionInfoList* region_list,
                                    RegionInfoPtrMap* map);

  // Traverses regions and fills their rasterization as well as assigns id.
  virtual void ObtainResults(RegionInfoList* region_list,
                             RegionInfoPtrMap* map,
                             bool remove_thin_structure,
                             bool enforce_n4_connections);

 protected:
  // Forward functions called after undoing type erasure via AbstractPixelDistance.
  void AddNodesAndSpatialEdgesForward(const SpatialDistance& dist);
  void AddNodesAndSpatialEdgesConstrainedForward(const SpatialDistance& dist,
                                                 const SegmentationDesc& desc);
  void AddTemporalEdgesForward(const TemporalDistance& distance);
  void AddTemporalFlowEdgesForward(const TemporalDistance& distance, const cv::Mat& flow);

  // Concrete implementation functions for specific frame index.
  // Used during parallel graph creation.
  void AddSpatialEdgesImpl(typename DistanceTraits::SpatialDistance dist,
                           int frame_idx);

  template<class Distance>
  void AddTemporalEdgesImpl(Distance distance,
                            int frame_idx);

  template<class Distance>
  void AddTemporalFlowEdgesImpl(Distance distance, const cv::Mat& flow, int frame_idx);

  // Returns edges between curr_idx (at location x, y) and prev_idx based on N-9.
  template<class Distance>
  void GetLocalEdges(int x,
                     int y,
                     int curr_idx,
                     int prev_idx,
                     Distance* distance,
                     std::vector<EdgeTuple>* local_edges);


  // Adds interval to region_id's corresponding RegionInformation* or creates new one
  // in region_list if not present in map yet.
  void AddIntervalToRasterization(int frame,
                                  int y,
                                  int left_x,
                                  int right_x,
                                  int region_id,
                                  RegionInfoList* region_list,
                                  RegionInfoPtrMap* map);

  int FrameNumber() const { return num_frames_; }
  int MaxFrames() const { return max_frames_; }

  class AddSpatialEdgesInvoker {
   public:
    AddSpatialEdgesInvoker(const SpatialDistance& distance, int frame_idx,
                           DenseSegmentationGraph* dense_graph)
        : distance_(distance), frame_idx_(frame_idx), dense_graph_(dense_graph) {
    }

    void operator()() {
      dense_graph_->AddSpatialEdgesImpl(distance_, frame_idx_);
    }

   private:
    SpatialDistance distance_;
    int frame_idx_;
    DenseSegmentationGraph* dense_graph_;
  };

  template<class Distance> class AddTemporalEdgesInvoker {
   public:
    AddTemporalEdgesInvoker(const Distance& distance, int frame_idx,
                            DenseSegmentationGraph* dense_graph)
        : distance_(distance), frame_idx_(frame_idx), dense_graph_(dense_graph) {
    }

    void operator()() {
      dense_graph_->AddTemporalEdgesImpl(distance_, frame_idx_);
    }

   private:
    Distance distance_;
    int frame_idx_;
    DenseSegmentationGraph* dense_graph_;
  };

  template<class Distance> class AddTemporalFlowEdgesInvoker {
   public:
    AddTemporalFlowEdgesInvoker(const Distance& distance,
                                const cv::Mat& flow,
                                int frame_idx,
                                DenseSegmentationGraph* dense_graph)
        : distance_(distance), flow_(flow), frame_idx_(frame_idx),
          dense_graph_(dense_graph) {
    }

    void operator()() {
      dense_graph_->AddTemporalFlowEdgesImpl(distance_, flow_, frame_idx_);
    }

   private:
    Distance distance_;
    cv::Mat flow_;
    int frame_idx_;
    DenseSegmentationGraph* dense_graph_;
  };

protected:
  // Create a new batch of nodes for current frame.
  void AddNodes();

  // Same as above but sets constraint from region_ids.
  void AddNodesConstrained(const SegmentationDesc& desc);
  void AddNodesWithDescriptors(SpatialDistance distance);
  void AddNodesConstrainedWithDescriptors(SpatialDistance distance,
                                          const SegmentationDesc& desc);

  // TODO(grundman): This function corrupts topological information. Fix this via 
  // more profound split / merge mechanism.
  void ThinStructureSuppression(cv::Mat* id_image,
                                std::unordered_map<int, int>* size_adjust_map);

  // Detects violating N8 only connectivity and performs resolving pixel swaps.
  void EnforceN4Connectivity(cv::Mat* id_image,
                             std::unordered_map<int, int>* size_adjust_map);

  // Ensures that all regions are spatially connected (within a radius of their
  // corresponding diameter).
  void EnforceSpatialConnectedness(
        RegionInfoList* region_list,
        RegionInfoPtrMap* region_map,
        std::unordered_map<int, int>* size_adjust_map);

  // Used to add constrained nodes.
  cv::Mat region_ids_;

  // Tracks constrained slices.
  std::vector<int> constrained_slices_;

  // No scanline results for virtual slices will be generated.
  std::vector<int> virtual_slices_;

  int num_frames_ = 0;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int max_frames_ = 0;

  std::vector<std::thread> add_edges_tasks_;
};

template<class DistanceTraits, class DescriptorTraits>
DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::DenseSegmentationGraph(
    int frame_width,
    int frame_height,
    int max_frames,
    const DescriptorTraits& desc_traits) :
        FastSegmentationGraph<DescriptorTraits>(1.0,            // normalized edges.
                                                2048,           // 2K buckets
                                                desc_traits,
                                                2 * max_frames - 1),  // spatial and
                                                                      // temporal.
        frame_width_(frame_width),
        frame_height_(frame_height),
        max_frames_(max_frames) {
  // Check correct combination of distance and descriptor traits.
  static_assert(DescriptorTraits::pixel_descriptor_size() ==
                DistanceTraits::SpatialDistance::descriptor_size(),
                "DescriptorTraits::pixel_descriptor_size must equal the size of data "
                "supplied by SpatialDistance::descriptor_size call.");

  // Create with one pixel boundary.
  region_ids_.create(frame_height_ + 2, frame_width_ + 2, CV_32S);
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::InitializeGraph() {
  const int nodes_per_frame = frame_width_ * frame_height_;

  // Allocate 2% more for flattening operation.
  this->ReserveNodes(nodes_per_frame * max_frames_ * 1.02f);
  const int neighbors = nodes_per_frame * 26 / 2;    // number of regions times average
                                                     // average number of edges.
                                                     // Undirected edges -> divide by 2
  this->ReserveEdges(neighbors * max_frames_);
}


template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddVirtualNodesConstrained(
    const SegmentationDesc& desc) {
  const int base_idx = num_frames_ * frame_height_ * frame_width_;
  CHECK_EQ(this->regions_.size(), base_idx);
  // Store as virtual slice.
  virtual_slices_.push_back(num_frames_);

  // Render to id image.
  cv::Mat id_view(region_ids_, cv::Rect(1, 1, frame_width_, frame_height_));
  SegmentationDescToIdImage(0,       // Level.
                            desc,
                            0,       // No Hierarchy.
                            &id_view);

  // Add nodes while performing merging all nodes with same constraint id
  // (pre-segmentation to avoid adding spatial edges).
  // Resulting regions have size zero (virtual).

  // Maps constrained id to representative for merging.
  std::unordered_map<int, int> constraint_to_rep_mat;
  for (int i = 0, region_idx = base_idx; i < frame_height_; ++i) {
    const int* id_ptr = id_view.ptr<int>(i);
    for (int j = 0; j < frame_width_; ++j, ++region_idx) {
      this->AddRegion(region_idx, 0, id_ptr[j]);
      auto pos = constraint_to_rep_mat.find(id_ptr[j]);
      if (pos == constraint_to_rep_mat.end()) {
        constraint_to_rep_mat.insert(std::make_pair(id_ptr[j], region_idx));
      } else {
        // Merge with representative (size is zero, same constraints, no descriptor).
        // Simply set representative id.
        this->regions_[region_idx].my_id = this->regions_[pos->second].my_id;
      }
    }
  }

  // Mark added nodes as virtual for underlying segmentation graph.
  this->AddVirtualNodeInterval(base_idx, base_idx + frame_height_ * frame_width_);
  ++num_frames_;
  CHECK_LE(num_frames_, max_frames_);
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddTemporalVirtualEdges() {
  // Place virtual edges into last bucket (highest edge weight).
  AddTemporalEdgesInvoker<ConstantPixelDistance> invoker(
      ConstantPixelDistance(1e10), num_frames_, this);

  if (FLAGS_parallel_graph_construction) {
    add_edges_tasks_.push_back(std::thread(invoker));
  } else {
    invoker();
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddTemporalFlowVirtualEdges(const cv::Mat& flow) {
  AddTemporalFlowEdgesInvoker<ConstantPixelDistance> invoker(
        ConstantPixelDistance(1e10),
        flow,
        num_frames_,
        this);
  if (FLAGS_parallel_graph_construction) {
    add_edges_tasks_.push_back(std::thread(invoker));
  } else {
    invoker();
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::FinishBuildingGraph() {
  if (FLAGS_parallel_graph_construction) {
    for (std::thread& thread : add_edges_tasks_) {
      thread.join();
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::SegmentGraphSpatially() {
  std::vector<int> spatial_bucket_lists(num_frames_);
  for (int i = 0; i < num_frames_; ++i) {
    spatial_bucket_lists[i] = 2 * i;
  }

  this->SegmentGraph(0,
                     false,
                     &spatial_bucket_lists);
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::SegmentFullGraph(
    int min_region_size,
    bool force_constraints) {
  this->SegmentGraph(min_region_size, force_constraints, nullptr);
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::DetermineNeighborIds(
    RegionInfoList* region_list,
    RegionInfoPtrMap* map) {
  this->DetermineNeighborIdsImpl(region_list, map);
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddIntervalToRasterization(int frame,
                               int y,
                               int left_x,
                               int right_x,
                               int region_id,
                               RegionInfoList* region_list,
                               RegionInfoPtrMap* map) {
  DCHECK_GE(region_id, 0);
  DCHECK_LT(region_id, this->regions_.size());
  DCHECK_EQ(region_id, this->regions_[region_id].my_id)
    << "Region is not a representative.";
  RegionInformation* ri = this->GetCreateRegionInformation(this->regions_[region_id],
                                                           false, region_list, map);

  // Allocate rasterization if necessary.
  if (ri->raster == nullptr) {
    ri->raster.reset(new Rasterization3D);
  }

  // Current frame-slice present?
  if (ri->raster->empty() || ri->raster->back().first < frame) {
    ri->raster->push_back(
        std::make_pair(frame, std::shared_ptr<Rasterization>(new Rasterization())));
  }


  Rasterization* raster = ri->raster->back().second.get();
  ScanInterval* scan_inter = raster->add_scan_inter();
  scan_inter->set_y(y);
  scan_inter->set_left_x(left_x);
  scan_inter->set_right_x(right_x);
}


template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    ObtainResults(RegionInfoList* region_list,
                  RegionInfoPtrMap* region_map,
                  bool remove_thin_structure,
                  bool enforce_n4_connections) {
  DCHECK(this->CheckForIsolatedRegions());

  // this->FlattenUnionFind(true);

  // Flag boundary with -1.
  region_ids_.row(0).setTo(-1);
  region_ids_.row(frame_height_ + 1).setTo(-1);
  region_ids_.col(0).setTo(-1);
  region_ids_.col(frame_width_ + 1).setTo(-1);

  cv::Mat id_view(region_ids_, cv::Rect(1, 1, frame_width_, frame_height_));

  // Maps region representative to change in region area due to thin structure
  // suppression and n4 connectivity.
  std::unordered_map<int, int> size_adjust_map;

  // Traverse all frames.
  for (int t = 0; t < num_frames_; ++t) {
    // Base region index for current frame.
    const int base_idx = frame_width_ * frame_height_ * t;
    const bool is_virtual_frame = std::binary_search(virtual_slices_.begin(),
                                                     virtual_slices_.end(),
                                                     t);
    if (is_virtual_frame) {
      continue;
    }

    bool is_constrained_frame = std::binary_search(constrained_slices_.begin(),
                                                   constrained_slices_.end(),
                                                   t);

    // Create id_view from the graph result.
    for (int i = 0, idx = base_idx; i < frame_height_; ++i) {
      int* region_ptr = id_view.ptr<int>(i);
      for (int j = 0; j < frame_width_; ++j, ++idx) {
        region_ptr[j] = this->GetRegion(idx)->my_id;
      }
    }

    // Assure constrained frames yield exactly the same result.
    if (!is_constrained_frame) {
      // TODO(grundman): This is more complicated, needs to be done after rasterization
      // as generic split operation.
      if (remove_thin_structure) {
        ThinStructureSuppression(&id_view, &size_adjust_map);
      }

      if (enforce_n4_connections) {
        EnforceN4Connectivity(&id_view, &size_adjust_map);
      }
    }

    // Traverse region_id_image.
    for (int i = 0, idx = base_idx; i < frame_height_; ++i) {
      const int* region_ptr = id_view.ptr<int>(i);
      int prev_id = region_ptr[0];
      int left_x = 0;
      for (int j = 1; j < frame_width_; ++j) {
        const int curr_id = region_ptr[j];
        // Test for new interval.
        if (prev_id != curr_id) {
          AddIntervalToRasterization(t,             // frame
                                     i,             // y coordinate
                                     left_x,        // start x
                                     j - 1,         // end x
                                     prev_id,       // region id
                                     region_list,
                                     region_map);
          // Reset current interval information.
          left_x = j;
          prev_id = curr_id;
        }

        // Add interval when end of scanline is reached.
        if (j + 1 == frame_width_) {
          AddIntervalToRasterization(t, i, left_x, j, prev_id, region_list, region_map);
        }
      }
    }
  }  // end image traversal.

  // Enforce spatial connectedness.
  // EnforceSpatialConnectedness(region_list, region_map, &size_adjust_map);

  // Adjust sizes.
  for (const auto& map_adjust_entry : size_adjust_map) {
    const auto pos = region_map->find(map_adjust_entry.first);
    if (pos == region_map->end()) {
      // Unlikely, but region could be completely erased, in which case it would have
      // no rasterization.
      DCHECK_EQ(0, this->regions_[map_adjust_entry.first].sz + map_adjust_entry.second);
      // Flag as virtual.
      this->regions_[map_adjust_entry.first].sz = 0;
      continue;
    }
    pos->second->size += map_adjust_entry.second;
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    EnforceSpatialConnectedness(
        RegionInfoList* region_list,
        RegionInfoPtrMap* region_map,
        std::unordered_map<int, int>* size_adjust_map) {
  const int num_regions = region_list->size();
  for (int k = 0; k < num_regions; ++k) {
    RegionInformation& ri = *(*region_list)[k];
    if (ri.raster == nullptr) {
      continue;
    }

    Rasterization3D& raster = *ri.raster;

    // Determine if for any frame rasterization is disconnected.
    
    for (const auto& raster_slice : raster) {
      std::vector<Rasterization> components;
      ConnectedComponents(*raster_slice.second, N4_CONNECT, &components);

      if (components.size() > 1) {
        // Disconnected. Determine by how much.
        std::vector<ShapeDescriptor> shapes(components.size());
        for (int l = 0; l < components.size(); ++l) {
          ShapeMoments moment;
          ShapeMomentsFromRasterization(components[l], &moment);
          GetShapeDescriptorFromShapeMoment(moment, &shapes[l]);
        }

        // Shape descriptor based criteria here.
      }
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddNodesAndSpatialEdgesForward(const SpatialDistance& distance) {
  AddNodesWithDescriptors(distance);
  AddSpatialEdgesInvoker invoker(distance, num_frames_, this);
  if (FLAGS_parallel_graph_construction) {
    add_edges_tasks_.push_back(std::thread(invoker));
  } else {
    invoker();
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddNodesAndSpatialEdgesConstrainedForward(const SpatialDistance& distance,
                                              const SegmentationDesc& desc) {
  AddNodesConstrainedWithDescriptors(distance, desc);
  AddSpatialEdgesInvoker invoker(distance, num_frames_, this);

  if (FLAGS_parallel_graph_construction) {
    add_edges_tasks_.push_back(std::thread(invoker));
  } else {
    invoker();
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddTemporalEdgesForward(
    const TemporalDistance& distance) {
  AddTemporalEdgesInvoker<TemporalDistance> invoker(distance, num_frames_, this);
  if (FLAGS_parallel_graph_construction) {
    add_edges_tasks_.push_back(std::thread(invoker));
  } else {
    invoker();
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddTemporalFlowEdgesForward(const TemporalDistance& distance,
                                const cv::Mat& flow) {
  AddTemporalFlowEdgesInvoker<TemporalDistance> invoker(
      distance, flow, num_frames_, this);
  if (FLAGS_parallel_graph_construction) {
    add_edges_tasks_.push_back(std::thread(invoker));
  } else {
    invoker();
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddSpatialEdgesImpl(
    typename DistanceTraits::SpatialDistance distance,
    int frame_idx) {
  // Add edges based on 8 neighborhood.
  const int base_idx = frame_idx * frame_height_ * frame_width_;
  const int bucket_list_idx = 2 * frame_idx;
  for (int i = 0, end_y = frame_height_ - 1, cur_idx = base_idx; i <= end_y; ++i) {
    distance.MoveAnchorTo(0, i);
    distance.MoveTestAnchorTo(0, i);

    for (int j = 0, end_x = frame_width_ - 1;
         j <= end_x;
         ++j, ++cur_idx, distance.IncrementAnchor(), distance.IncrementTestAnchor()) {

      if (j < end_x) { // Edge to right.
        this->AddEdge(cur_idx,
                      cur_idx + 1,
                      distance.PixelDistance(1, 0),
                      bucket_list_idx);
      }

      if (i < end_y) {  // Edge to bottom.
        this->AddEdge(cur_idx,
                      cur_idx + frame_width_,
                      distance.PixelDistance(0, 1),
                      bucket_list_idx);

        if (j > 0) {  // Edge to bottom left.
          this->AddEdge(cur_idx,
                        cur_idx + frame_width_ - 1,
                        distance.PixelDistance(-1, 1),
                        bucket_list_idx);
        }

        if (j < end_x) { // Edge to bottom right
          this->AddEdge(cur_idx,
                        cur_idx + frame_width_ + 1,
                        distance.PixelDistance(1, 1),
                        bucket_list_idx);
        }
      }
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
template<class Distance>
inline void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::GetLocalEdges(
    int x,              // (x,y) position in  *previous* frame.
    int y,
    int curr_idx,
    int prev_idx,
    Distance* distance,
    std::vector<EdgeTuple>* local_edges) {
  if (y > 0) { // Edges to top.
    const int local_prev_idx = prev_idx - frame_width_;
    if (x > 0) {
      local_edges->push_back(EdgeTuple(curr_idx,
                                       local_prev_idx - 1,
                                        distance->PixelDistance(-1, -1)));
    }

    local_edges->push_back(EdgeTuple(curr_idx,
                                     local_prev_idx,
                                     distance->PixelDistance(0, -1)));

    if (x + 1 < frame_width_) {
      local_edges->push_back(EdgeTuple(curr_idx,
                                       local_prev_idx + 1,
                                       distance->PixelDistance(1, -1)));
    }
  }

  // Edges left and right.
  const int local_prev_idx = prev_idx;
  if (x > 0) {
    local_edges->push_back(EdgeTuple(curr_idx,
                                     local_prev_idx - 1,
                                     distance->PixelDistance(-1, 0)));
  }

  local_edges->push_back(EdgeTuple(curr_idx,
                                   local_prev_idx,
                                   distance->PixelDistance(0, 0)));

  if (x + 1 < frame_width_) {
    local_edges->push_back(EdgeTuple(curr_idx,
                                     local_prev_idx + 1,
                                     distance->PixelDistance(1, 0)));
  }

  if (y + 1 < frame_height_) {  // Edges to bottom.
    const int local_prev_idx = prev_idx + frame_width_;
    if (x > 0) {
      local_edges->push_back(EdgeTuple(curr_idx,
                                       local_prev_idx - 1,
                                       distance->PixelDistance(-1, 1)));
    }

    local_edges->push_back(EdgeTuple(curr_idx,
                                     local_prev_idx,
                                     distance->PixelDistance(0, 1)));

    if (x + 1 < frame_width_) {
      local_edges->push_back(EdgeTuple(curr_idx,
                                       local_prev_idx + 1,
                                       distance->PixelDistance(1, 1)));
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
template<class Distance>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddTemporalEdgesImpl(
    Distance distance,
    int frame_idx) {
  // Add edges based on 9 neighborhood in time.
  const int base_idx = (frame_idx - 1) * frame_width_ * frame_height_;
  const int base_diff = frame_width_ * frame_height_;
  // One before previous spatial bucket.
  const int bucket_list_idx = 2 * (frame_idx - 1) - 1;
  int curr_idx = base_idx;
  std::vector<EdgeTuple> local_edges;

  for (int i = 0; i < frame_height_; ++i) {
    distance.MoveAnchorTo(0, i);
    distance.MoveTestAnchorTo(0, i);

    for (int j = 0;
         j < frame_width_;
         ++j, ++curr_idx, distance.IncrementAnchor(), distance.IncrementTestAnchor()) {
      local_edges.clear();
      const int prev_idx = curr_idx - base_diff;
      GetLocalEdges(j, i, curr_idx, prev_idx, &distance, &local_edges);

      for (const auto edge : local_edges) {
        this->AddEdge(edge.node_1, edge.node_2, edge.weight, bucket_list_idx);
      }
    }
  }
}

// Same as above, but displaces flow edges in time along flow.
template<class DistanceTraits, class DescriptorTraits>
template<class Distance>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddTemporalFlowEdgesImpl(Distance distance,
                             const cv::Mat& flow,
                             int frame_idx) {
  DCHECK(!flow.empty());
  DCHECK_EQ(flow.cols, frame_width_);
  DCHECK_EQ(flow.rows, frame_height_);
  // Nodes were added by AddNodesAndSpatialEdges run.
  // Add temporal edges to previous frame.
  const int base_diff = frame_width_ * frame_height_;
  const int base_idx = (frame_idx - 1) * frame_width_ * frame_height_;
  // One before previous spatial bucket.
  const int bucket_list_idx = 2 * (frame_idx - 1) - 1;
  std::vector<EdgeTuple> local_edges;

  // Add edges based on 9 neighborhood in time.
  for (int i = 0, curr_idx = base_idx; i < frame_height_; ++i) {
    distance.MoveAnchorTo(0, i);

    const float* flow_ptr = flow.ptr<float>(i);
    for (int j = 0; j < frame_width_;
         ++j, ++curr_idx, distance.IncrementAnchor(), flow_ptr += 2) {
      local_edges.clear();

      int prev_x = j + flow_ptr[0];
      int prev_y = i + flow_ptr[1];

      prev_x = std::max(0, std::min(frame_width_ - 1, prev_x));
      prev_y = std::max(0, std::min(frame_height_ - 1, prev_y));

      distance.MoveTestAnchorTo(prev_x, prev_y);

      int prev_idx = base_idx - base_diff + prev_y * frame_width_ + prev_x;
      GetLocalEdges(prev_x, prev_y, curr_idx, prev_idx, &distance, &local_edges);

      for (const auto edge : local_edges) {
        this->AddEdge(edge.node_1, edge.node_2, edge.weight, bucket_list_idx);
      }
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddNodes() {
  const int base_idx = num_frames_ * frame_height_ * frame_width_;
  CHECK_EQ(this->regions_.size(), base_idx);

  for (int i = 0; i < frame_height_; ++i) {
    int row_idx = base_idx + i * frame_width_;
    for (int j = 0; j < frame_width_; ++j) {
      this->AddRegion(row_idx + j, 1, -1);  // Size one region, no constrained.
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddNodesConstrained(
    const SegmentationDesc& desc) {
  const int base_idx = num_frames_ * frame_height_ * frame_width_;
  CHECK_EQ(this->regions_.size(), base_idx);
  constrained_slices_.push_back(num_frames_);

  // Render to id image.
  cv::Mat id_view(region_ids_, cv::Rect(1, 1, frame_width_, frame_height_));
  SegmentationDescToIdImage(0,       // Level.
                            desc,
                            0,       // No Hierarchy.
                            &id_view);

  for (int i = 0; i < frame_height_; ++i) {
    const int* id_ptr = id_view.ptr<int>(i);
    int row_idx = base_idx + i * frame_width_;
    for (int j = 0; j < frame_width_; ++j) {
      this->AddRegion(row_idx + j, 1, id_ptr[j]);
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::AddNodesWithDescriptors(
    SpatialDistance distance) {
  const int base_idx = num_frames_ * frame_height_ * frame_width_;
  CHECK_EQ(this->regions_.size(), base_idx);

  for (int i = 0; i < frame_height_; ++i) {
    int row_idx = base_idx + i * frame_width_;
    distance.MoveAnchorTo(0, i);
    float descriptor[DescriptorTraits::pixel_descriptor_size()];

    for (int j = 0; j < frame_width_; ++j, distance.IncrementAnchor()) {
      distance.SetPixelDescriptor(descriptor);
      this->AddRegionWithDescriptor(row_idx + j,
                                    1,              // Size.
                                    -1,             // No constraint.
                                    descriptor);
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    AddNodesConstrainedWithDescriptors(SpatialDistance distance,
                                       const SegmentationDesc& desc) {
  const int base_idx = num_frames_ * frame_height_ * frame_width_;
  CHECK_EQ(this->regions_.size(), base_idx);

   // Render to id image.
   cv::Mat id_view(region_ids_, cv::Rect(1, 1, frame_width_, frame_height_));
   SegmentationDescToIdImage(0,       // Level.
                             desc,
                             0,       // No Hierarchy.
                             &id_view);

  for (int i = 0; i < frame_height_; ++i) {
    const int* id_ptr = id_view.ptr<int>(i);
    int row_idx = base_idx + i * frame_width_;
    distance.MoveAnchorTo(0, i);
    float descriptor[DescriptorTraits::pixel_descriptor_size()];
    for (int j = 0; j < frame_width_; ++j, distance.IncrementAnchor()) {
      distance.SetPixelDescriptor(descriptor);
      this->AddRegionWithDescriptor(row_idx + j,
                                    1,             // Size.
                                    id_ptr[j],
                                    descriptor);
    }
  }
}


///////////////// NOTE: Corrupts topology. Algorithm is sound but introduced swaps
// can change neighboring information. TODO: Fix this to a more profound splitting
// algorithm.

namespace {

void ThinStructureSwap(const int n8_offsets[8],
                       const int n4_offsets[4],
                       int* region_ptr,
                       std::unordered_map<int, int>* size_adjust_map) {
  int num_neighbors = 0;
  const int region_id = *region_ptr;
  for (int k = 0; k < 8; ++k) {
    num_neighbors += (region_ptr[n8_offsets[k]] == region_id);
  }

  if (num_neighbors <= 2) {
    // Deemed a thin structure.
    std::vector<int> n4_vote;
    int swap_to = -1;
    // Compute N4-based swap.
    for (int k = 0; k < 4; ++k) {
      int n_id = region_ptr[n4_offsets[k]];
      if(n_id != -1 && n_id != region_id && !InsertSortedUniquely(n_id, &n4_vote)) {
        swap_to = n_id;
      }
    }

    if (swap_to >= 0) {
      // Perform swap.
      --(*size_adjust_map)[*region_ptr];
      ++(*size_adjust_map)[swap_to];
      *region_ptr = swap_to;

      // Recompute neighbors.
      int* neighbors[2];
      int n_id = 0;
      for (int k = 0; k < 8; ++k) {
        if (region_ptr[n8_offsets[k]] == region_id) {
          neighbors[n_id++] = region_ptr + n8_offsets[k];
        }
      }

      // Propagate to neighbors.
      for (int n = 0; n < num_neighbors; ++n) {
        ThinStructureSwap(n8_offsets, n4_offsets, neighbors[n], size_adjust_map);
      }
    }
  }
}

}  // namespace.

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::ThinStructureSuppression(
    cv::Mat* id_image,
    std::unordered_map<int, int>* size_adjust_map) {
  // Traverse image and find thin structures (connected to only two neighbors).
  const int lda = id_image->step1();
  const int n8_offsets[8] = { -lda - 1, -lda, -lda + 1,
                                    -1,              1,
                               lda - 1, lda, lda + 1 };
  const int n4_offsets[8] = { -lda, -1, 1, lda };

  for (int i = 0; i < frame_height_ - 1; ++i) {
    int* region_ptr = id_image->ptr<int>(i);
    for (int j = 0; j < frame_width_; ++j, ++region_ptr) {
      ThinStructureSwap(n8_offsets, n4_offsets, region_ptr, size_adjust_map);
    }
  }
}

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::EnforceN4Connectivity(
    cv::Mat* id_image,
    std::unordered_map<int, int>* size_adjust_map) {
  // General algorithm: Traverse image in order, if N4 connectivity is violated,
  // (bottom left or bottom right has same id as current, but left or right one is
  // different, simply swap bottom element to the same id as current one to connect).
  // Note: This does not change topological information, as neighbors are introduced
  // based on N8 connectivity, so a single N4 swap is OK.
  CHECK_NOTNULL(size_adjust_map);
  const int lda = id_image->step1();
  for (int i = 0; i < frame_height_ - 1; ++i) {
    int* region_ptr = id_image->ptr<int>(i);
    for (int j = 0; j < frame_width_; ++j, ++region_ptr) {
      const int region_id = *region_ptr;
      if (region_ptr[lda - 1] == region_id &&
          region_ptr[-1] != region_id &&
          region_ptr[lda] != region_id) {
        // Swap bottom to region_id.
        --(*size_adjust_map)[region_ptr[lda]];
        ++(*size_adjust_map)[region_id];
        region_ptr[lda] = region_id;
      }

      if (region_ptr[lda + 1] == region_id &&
          region_ptr[1] != region_id &&
          region_ptr[lda] != region_id) {
        // Swap bottom to region_id.
        --(*size_adjust_map)[region_ptr[lda]];
        ++(*size_adjust_map)[region_id];
        region_ptr[lda] = region_id;
      }
    }
  }
}

}  // namespace segmentation.

#endif // VIDEO_SEGMENT_SEGMENTATION_DENSE_SEGMENTATION_GRAPH_H__
