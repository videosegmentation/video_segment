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

#ifndef VIDEO_SEGMENT_SEGMENTATION_DENSE_SEG_GRAPH_INTERFACE_H__
#define VIDEO_SEGMENT_SEGMENTATION_DENSE_SEG_GRAPH_INTERFACE_H__

#include "base/base.h"

namespace cv {
  class Mat;
}

namespace segmentation {

class DenseSegGraphInterface;

// Creates dense segmentation graph templated with distance and descriptor traits.
class DenseSegGraphCreatorInterface {
 public:
  virtual DenseSegGraphInterface* CreateDenseSegGraph(int frame_width,
                                                      int frame_height_,
                                                      int max_frames) const = 0;
  virtual ~DenseSegGraphCreatorInterface() = 0;
};

// Bundles spatial and temporal distance.
template<class Spatial, class Temporal>
struct DistanceTraits {
  typedef Spatial SpatialDistance;
  typedef Temporal TemporalDistance;
};

// Usage:
// DenseSegGraphCreatorInterface* interface;   // Supplied by client.
//
// DenseSegGraphInterface* graph = interface.CreateDenseSegGraph(960, 540, 100);
// 
// SpatialDistance spatial_distance[N];      // supplied by user.
// TemporalDistance temporal_distance[N-1];  // supplied by user.
//
// graph->InitializeGraph();
//
// // Add first nodes and spatial connections for two frames.
// graph.AddNodesAndSpatialEdges(&spatial_distance[0]);
//
// graph.AddNodesAndSpatialEdges(&spatial_distance[1]);
// 
// // Connect in time.
// graph.AddTemporalEdges(&temporal_distance[0]);
// // Add another slice.
// graph.AddNodesAndSpatialEdges(&spatial_distance[2]);
// // Connect in time.
// graph.AddTemporalEdges(&temporal_distance[1]);
//
// ...
//
// // Graph is created in parallel: wait.
// graph.FinishBuildingGraph();
//
// graph.SegmentFullGraph(200,     // Min region size in voxels.
//                        false);  // No constraints being used here.
//
// // Obtain results.
// RegionInfoList regions;
// RegionInfoPtrMap id_to_region_map;    // Maps representative id to RegionInformation*
// graph.AssignRegionIds(&regions, &id_to_region_map);
//
// // Read out rasterized results.
// graph.ObtainScanlineRepFromResults(id_to_region_map,
//                                    false,    // Thin structure suppression.
//                                    true);    // N4 connectivity.


// Forward decl's.
class AbstractPixelDistance;
class SegmentationDesc;
class RegionInformation;
typedef std::unordered_map<int, RegionInformation*> RegionInfoPtrMap;
typedef std::vector<std::unique_ptr<RegionInformation> > RegionInfoList;

class DenseSegGraphInterface {
 public:
  virtual ~DenseSegGraphInterface() = 0;

  // Pre-allocates memory.
  virtual void InitializeGraph() = 0;

  // Creates nodes in a graph, connects them in space.
  virtual void AddNodesAndSpatialEdges(const AbstractPixelDistance& dist) = 0;
  virtual void AddNodesAndSpatialEdgesConstrained(const AbstractPixelDistance& dist,
                                                  const SegmentationDesc& desc) = 0;

  // Adds and spatially connects virtual nodes (e.g. to establish correct neighboring
  // correspondences during constrained segmentation).
  virtual void AddVirtualNodesConstrained(const SegmentationDesc& desc) = 0;

  // Connects last two added node layers with temporal edges.
  virtual void AddTemporalEdges(const AbstractPixelDistance& distance) = 0;

  // Same as above along dense flow.
  virtual void AddTemporalFlowEdges(const AbstractPixelDistance& distance,
                                    const cv::Mat& flow) = 0;

  // Add virtual edges (no edge weight, just for topological neighboring information).
  virtual void AddTemporalVirtualEdges() = 0;
  virtual void AddTemporalFlowVirtualEdges(const cv::Mat& flow) = 0;

  // If graph is build in parallel, locks until creation is done.
  virtual void FinishBuildingGraph() = 0;

  // Segments graph only in space (not in time).
  virtual void SegmentGraphSpatially() = 0;

  // Full spatio-temporal segmentation.
  virtual void SegmentFullGraph(int min_region_size, bool force_constraints) = 0;

  // After segmentation: Assigns region ids and neighbors.
  virtual void AssignRegionIds(RegionInfoList* region_list,
                               RegionInfoPtrMap* map) = 0;

  // Creates corresponding rasterization for each region.
  virtual void ObtainScanlineRepFromResults(const RegionInfoPtrMap& map,
                                            bool remove_thin_structure,
                                            bool enforce_n4_connections) = 0;
};

}  // namespace segmentation.

#endif  // VIDEO_SEGMENT_SEGMENTATION_DENSE_SEG_GRAPH_INTERFACE_H__
