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
                             const std::vector<cv::Mat>* flows,                           
                             bool remove_thin_structure,
                             bool enforce_n4_connections,
                             bool enforce_spatial_connectedness);

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
        const std::vector<cv::Mat>* flows,                           
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
                  const std::vector<cv::Mat>* flows,                           
                  bool remove_thin_structure,
                  bool enforce_n4_connections,
                  bool enforce_spatial_connectedness) {
  DCHECK(this->CheckForIsolatedRegions());

  if (enforce_spatial_connectedness) {
    this->FlattenUnionFind(true);
    if (flows) {
      CHECK_EQ(num_frames_, flows->size()) << "Specify one flow field per frame.";
    }
  }

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

  if (enforce_spatial_connectedness) {
    EnforceSpatialConnectedness(region_list, region_map, flows, &size_adjust_map);
  }

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

// Captures information for one spatially connected tube in at a specific frame.
struct TubeSlice;
typedef std::vector<TubeSlice> Tube3D;

struct TubeSlice {
  int frame = -1;
  Rasterization raster;
  ShapeDescriptor shape;

  TubeSlice() = default;
  TubeSlice(const TubeSlice&) = default;
  TubeSlice& operator=(const TubeSlice&) = default;

  TubeSlice(TubeSlice&& rhs) : frame(rhs.frame), shape(rhs.shape) {
    raster.Swap(&rhs.raster);
  }

  // Finds the corresponding tube in the previous frame - 1. Optionally uses optical
  // flow information if supplied.
  // Returns pair (previous index, distance to previous index).
  std::pair<int, float>
  FindPreviousTube(const std::vector<Tube3D>& slices,
                   int frame,
                   const cv::Mat* flow) const {
    // Find closest prev centroid.
    cv::Point2f prev_center = shape.center;
    if (flow) {
      DCHECK(prev_center.x >= 0 && prev_center.x < flow->cols);
      DCHECK(prev_center.y >= 0 && prev_center.y < flow->rows);
      const float* flow_ptr = flow->ptr<float>(prev_center.y) + 2 * (int)prev_center.x;
      prev_center += cv::Point2f(flow_ptr[0], flow_ptr[1]);
    }

    float closest_dist = std::numeric_limits<float>::max();
    float closest_idx = -1;
    for (int k = 0; k < slices.size(); ++k) {
      if (slices[k].empty() || slices[k].back().frame >= frame) {
        // Got already used by another matching tube.
        continue;
      }
      const cv::Point2f diff = slices[k].back().shape.center - prev_center;
      const float dist = hypot(diff.y, diff.x);
      if (dist < closest_dist) {
        closest_dist = dist;
        closest_idx = k;
      }
    }
    return std::make_pair(closest_idx, closest_dist);
  }

  void MergeFrom(const TubeSlice& other) {
    CHECK_EQ(frame, other.frame);
    MergeRasterization(raster, other.raster, &raster);
    ComputeShapeDescriptor();
  }

  void ComputeShapeDescriptor() {
    ShapeMoments moment;
    ShapeMomentsFromRasterization(raster, &moment);
    GetShapeDescriptorFromShapeMoment(moment, &shape);
  }
};

// Returns average area of a slice across time.
float AverageTubeSliceSize(const Tube3D& ts);

// Merges lhs + rhs into result. Does not allow for inplace operation.
void MergeTube3D(const Tube3D& lhs, const Tube3D& rhs, Tube3D* result);

// Returns true if tubes are neighboring in time, and matching slices are of similiar
// area and share same centroid.
bool AreTubesTemporalNeighbors(const Tube3D& lhs, const Tube3D& rhs);

// Returns average tube distance (of centroids) between lhs and rhs.
float AverageTubeDistance(const Tube3D& lhs, const Tube3D& rhs);

// Returns fraction (in [0, 1]) for which lhs and rhs intersect.
float Tube3DIntersection(const Tube3D& lhs, const Tube3D& rhs);

// Returns index of closest tube in vectors tubes, -1 if tubes is empty. Specify index
// to skip in tubes via ignore_index.
int GetClosestTube3D(const Tube3D& tube,
                     const std::vector<Tube3D>& tubes,
                     int ignore_index = -1);

template<class DistanceTraits, class DescriptorTraits>
void DenseSegmentationGraph<DistanceTraits, DescriptorTraits>::
    EnforceSpatialConnectedness(
        RegionInfoList* region_list,
        RegionInfoPtrMap* region_map,
        const std::vector<cv::Mat>* flows,                           
        std::unordered_map<int, int>* size_adjust_map) {

  const int num_regions = region_list->size();
  for (int r = 0; r < num_regions; ++r) {
    RegionInformation& ri = *(*region_list)[r];
    if (ri.raster == nullptr) {
      continue;
    }

    Rasterization3D& raster = *ri.raster;

    // Split the spatio-temporal rasterization into connected tubes.
    std::vector<Tube3D> result_tubes;

    // Set of active tubes, that are currently considered.
    std::vector<Tube3D> active_tubes;

    const float inv_frame_diam = 1.0f / hypot(frame_width_, frame_height_);
      
    // Determine if for any frame rasterization is disconnected.
    for (const auto& raster_slice : raster) {
      const int frame = raster_slice.first;
      std::vector<Rasterization> components;
      ConnectedComponents(*raster_slice.second, N4_CONNECT, &components);

      // Create one tube slice for each component.
      std::vector<TubeSlice> slices;
      slices.reserve(components.size());

      for (auto& comp_raster : components) {
        TubeSlice slice;
        slice.frame = frame;
        slice.raster.Swap(&comp_raster);
        slice.ComputeShapeDescriptor();
        slices.push_back(std::move(slice));
      }

      // Ensure no-one is using components anymore.
      components.clear();

      if (active_tubes.empty()) {
        // Initialize a new one frame Tube3D from each slice.
        for (auto& slice : slices) {
          active_tubes.push_back(Tube3D{std::move(slice)});
        }
      } else {
        // Determine new set of active tubes. 
        std::vector<Tube3D> new_active_tubes;
        
        // Find which slices have a corrsponding slice and connect.
        // Indicates which slices have been used.
        std::vector<int> used_indices(active_tubes.size(), 0);
        for (auto& slice : slices) {
          const auto match = slice.FindPreviousTube(active_tubes, frame,
                                                    flows ? &flows->at(frame) : nullptr);

          const int prev_idx = match.first;
          if (prev_idx < 0) {
            // No corresponding tube found, start new one.
            new_active_tubes.push_back(Tube3D{std::move(slice)});
            continue;
          }

          const float diff_dist = match.second;
          const float area_ratio =
            std::min(active_tubes[prev_idx].back().shape.size, slice.shape.size) /
            (std::max(active_tubes[prev_idx].back().shape.size, slice.shape.size) + 1e-6);


          if (area_ratio > 0.75 &&
              diff_dist * inv_frame_diam < 0.04f) {
            // Found matching tube, continue.
            DCHECK_EQ(used_indices[prev_idx], 0);
            ++used_indices[prev_idx];

            active_tubes[prev_idx].push_back(std::move(slice));

            new_active_tubes.push_back(Tube3D());
            new_active_tubes.back().swap(active_tubes[prev_idx]);
          } else {
            // No corresponding tube found, start new one.
            new_active_tubes.push_back(Tube3D{std::move(slice)});
          }
        }

        // Copy all previous active tubes that are not continued to the result.
        for(int k = 0; k < active_tubes.size(); ++k) {
          if (used_indices[k] == 0) {
            result_tubes.push_back(std::move(active_tubes[k]));
          }
        }
 
        // Replace with new set.
        new_active_tubes.swap(active_tubes);
      }
    }  // end raster slice processing.

    // Push remaining active tubes to results.
    for(auto& slice : active_tubes) {
      result_tubes.push_back(std::move(slice));
    }

    if (result_tubes.size() <= 1) {
      continue;
    }

    // Post-processing: Merge tubes of small size and close distance in space
    //                  and time.
    auto merge_with_closest_tube =
      [&result_tubes](int k) -> bool {
      int idx = GetClosestTube3D(result_tubes[k], result_tubes, k);
      if (idx < 0) {
        return false;
      }
       
      Tube3D merged;
      MergeTube3D(result_tubes[idx], result_tubes[k], &merged);
      result_tubes[idx].swap(merged);
      result_tubes.erase(result_tubes.begin() + k);
      return true;
    };

    // Merge tubes that are small, or close to each other.
    for (int k = 0; k < result_tubes.size();  /* advanced below */ ) {
      bool merge = AverageTubeSliceSize(result_tubes[k]) < 20;
      if (!merge) {
        // Is tube close to any other tube?
        for (int l = 0; l < result_tubes.size(); ++l) {
          if (l == k) {
            continue;
          }
          if (Tube3DIntersection(result_tubes[k], result_tubes[l]) > 0.8) {
            merge = true;
            break;
          }
        }
      }

      if (merge && merge_with_closest_tube(k)) {
        // Nothing to do on successful merge.
      } else {
        // Advance.
        ++k;
      }
    }

    // Merge tubes that are temporal neighbors (and of compatible size / position).
    for (int k = 0; k < result_tubes.size();  /* advanced below */ ) {
      bool is_merged = false;
      for (int l = 0; l < result_tubes.size(); ++l) {
        if (l == k) {
          continue;
        }
        if (AreTubesTemporalNeighbors(result_tubes[k], result_tubes[l])) {
          Tube3D merged;
          MergeTube3D(result_tubes[k], result_tubes[l], &merged);
          result_tubes[l].swap(merged);
          result_tubes.erase(result_tubes.begin() + k);
          is_merged = true;
          // Only perform one merge per Tube (if both ends are mergeable, we will
          // visit the other corresponding end later).
          break;
        }
      }
      if (!is_merged) {
        ++k;
      }
    }

    // Largest / Longest tube keeps id, rest is assigned a new id.
    // Determine largest/longest tube.
    int tube_to_keep = -1;
    int tube_to_keep_score = 0;
    std::vector<float> tube_areas(result_tubes.size());
    for (int k = 0; k < result_tubes.size(); ++k) {
      // Compute tube area and length.
      int length = 0;
      float area = 0;
      for (const auto& slice : result_tubes[k]) {
        area += slice.shape.size;
        ++length;
      }
      tube_areas[k] = area;

      const float tube_score = area;
      if (tube_score > tube_to_keep_score) {
        tube_to_keep_score = tube_score;
        tube_to_keep = k;
      }
    }

    // Re-assign ids.
    for (int k = 0; k < result_tubes.size(); ++k) {
      int first_idx = result_tubes[k][0].frame * frame_width_ * frame_height_;
      const auto& first_scanline = result_tubes[k][0].raster.scan_inter(0);
      first_idx += first_scanline.y() * frame_width_ + first_scanline.left_x();
     
      typedef typename FastSegmentationGraph<DescriptorTraits>::Region Region;
      Region* rep = this->GetRegion(first_idx);
      if (k != tube_to_keep) {
        (*size_adjust_map)[rep->my_id] -= tube_areas[k];

        // Create new region.
        this->regions_.push_back(Region(this->regions_.size(),
                                        tube_areas[k],
                                        -1));    // unconstrained.
        rep = &this->regions_.back();
        const int region_id = rep->my_id;
      
        // Label all nodes with new id.
        for (const auto& slice : result_tubes[k]) {
          const int base_idx = slice.frame * frame_width_ * frame_height_;
          for (const auto& scan_inter : slice.raster.scan_inter()) {
            const int row_idx = base_idx + scan_inter.y() * frame_width_;
            for (int x = scan_inter.left_x(); x <= scan_inter.right_x(); ++x) {
              this->regions_[row_idx + x].my_id = region_id;
            }
          }
        }
      }

      // Create RegionInformation and reset rasterization.
      RegionInformation* ri = this->GetCreateRegionInformation(*rep, false,
                                                               region_list, region_map);
      ri->raster.reset(new Rasterization3D);
      for (auto& slice : result_tubes[k]) {
        std::shared_ptr<Rasterization> new_raster(new Rasterization());
        new_raster->Swap(&slice.raster);
        ri->raster->push_back(std::make_pair(slice.frame, new_raster));
      }
    }
  } // end regions.
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
