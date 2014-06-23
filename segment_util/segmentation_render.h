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

#ifndef SEGMENTATION_RENDER_H__
#define SEGMENTATION_RENDER_H__

#include "base/base.h"

#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "segment_util/segmentation_util.h"

namespace segmentation {

// Tuple representing a region at a specific level. Used to represent local
// hierarchy levels.
struct RegionID {
  RegionID(int id_ = -1, int level_ = 0) : id(id_), level(level_) {};
  int id;
  int level;

  // Lexicographic ordering.
  bool operator<(const RegionID& region) const {
    return level < region.level || (level == region.level && id < region.id);
  }

  bool operator==(const RegionID& region) const {
    return id == region.id && level == region.level;
  }

  bool operator!=(const RegionID& region) const {
    return !(*this == region);
  }
};

struct RegionIDHasher {
 public:
  size_t operator()(const RegionID& r_id) const {
    // We don't expect more than 128 = 2^8 hierarchy levels.
    return ((r_id.level % 128) << (31-8)) | ((r_id.id << 8) >> 8); }
};

// Functional object to customize RenderRegionsRandomColor. Uses global hierarchy level
// to map oversegmented regions to their corresponding parent id at hierarchy_level.
class HierarchyColorGenerator {
 public:
  // Parameter hierarchy can be zero if hierarchy_level is set to zero.
  // If hierarchy_level is larger than the number of levels in hierarchy,
  // hierarchy_level is truncated to the maximum possible level and a WARNING
  // is issued.
  HierarchyColorGenerator(int hierarchy_level,
                          int channels,
                          const Hierarchy* hierarchy);

  // Is called by RenderRegions with each oversegmented region id.
  // Id is mapped to corresponding parent id at the desired hierarchy level
  // and this id is used as color seed to random generate colors.
  // Mapped id needs also to be returned (for draw shape descriptors, etc.)
  // If false is returned, region is not rendered.
  bool operator()(int overseg_region_id,
                  RegionID* mapped_id,
                  unsigned char* colors) const;

protected:
  const Hierarchy* hierarchy_;
  int hierarchy_level_;
  const int channels_;
};

inline int ColorDiff_L1(const uint8_t* first, const uint8_t* second) {
  return abs((int)first[0] - (int)second[0]) +
         abs((int)first[1] - (int)second[1]) +
         abs((int)first[2] - (int)second[2]);
}

// Customizable by ColorGenerator.
// ColorGenerator requires the interface below to be implemented.
// Specifically, ColorGenerator is called for each oversegmented region id
// which is assumed to be mapped by ColorGenerator to a region id at a specific level
// captured by RegionID (the level can be local w.r.t. each oversegmented region).
// In particular, oversegmented ids that are mapped to the same RegionID are aggregated
// to facilitate the display of joint descriptors (e.g. shape decriptors).
// In addition, the colors the over-segmented region should be painted with should be
// returned in colors. For an example, see above HierarchyColorGenerator.
//
// bool operator(int oversegmented_region_id,
//               RegionID* mapped_id,
//               unsigned char* colors) const;
//
// Note: Return value indicates if region is to be rendered (return false to skip the
//       region).

template<class ColorGenerator>
void RenderRegions(bool highlight_boundary,
                   bool draw_shape_descriptors,
                   const SegmentationDesc& seg,
                   const ColorGenerator& generator,
                   cv::Mat* output) {
  CHECK_NOTNULL(output);

  // Parent map: map parent id to vector of children id.
  typedef std::unordered_map<RegionID, std::vector<int>, RegionIDHasher> RegionIDMap;
  RegionIDMap parent_map;
  const int channels = output->channels();
  std::vector<uint8_t> color(channels);

  // Traverse regions.
  for (const auto& r : seg.region()) {
    // Get color.
    RegionID mapped_id;
    if (!generator(r.id(), &mapped_id, &color[0])) {
      continue;
    }

    parent_map[mapped_id].push_back(r.id());

    for (const auto s : r.raster().scan_inter()) {
      const int curr_y = s.y();
      uint8_t* out_ptr = output->ptr<uint8_t>(curr_y) + channels * s.left_x();
      for (int j = 0, len = s.right_x() - s.left_x() + 1;
           j < len;
           ++j, out_ptr += channels) {
        for (int c = 0; c < channels; ++c) {
          out_ptr[c] = color[c];
        }
      }
    }
  }

  // Edge highlight post-process.
  if (highlight_boundary) {
    const int height = output->rows;
    const int width = output->cols;
    const int width_step = output->step[0];
    for (int i = 0; i < height - 1; ++i) {
      uint8_t* row_ptr = output->ptr<uint8_t>(i);
      for (int j = 0; j < width - 1; ++j, row_ptr += channels) {
        if (ColorDiff_L1(row_ptr, row_ptr + channels) != 0 ||
            ColorDiff_L1(row_ptr, row_ptr + width_step) != 0)
          row_ptr[0] = row_ptr[1] = row_ptr[2] = 0;
      }

      // Last column.
      if (ColorDiff_L1(row_ptr, row_ptr + width_step) != 0)
        row_ptr[0] = row_ptr[1] = row_ptr[2] = 0;
    }

    // Last row.
    uint8_t* row_ptr = output->ptr<uint8_t>(height - 1);
    for (int j = 0; j < width - 1; ++j, row_ptr += channels) {
      if (ColorDiff_L1(row_ptr, row_ptr + channels) != 0)
        row_ptr[0] = row_ptr[1] = row_ptr[2] = 0;
    }
  }

  if (draw_shape_descriptors) {
    for (RegionIDMap::const_iterator parent = parent_map.begin();
         parent != parent_map.end();
         ++parent) {
      int parent_id = parent->first.id;
      std::string parent_string = boost::lexical_cast<std::string>(parent_id);
      RenderShapeDescriptor(parent->second, seg, output, &parent_string);
    }
  }
}


// Renders each region with a random color for 3-channel 8-bit input image.
// If highlight_boundary is set, region boundary will be colored black.
inline void RenderRegionsRandomColor(int hierarchy_level,
                                     bool highlight_boundary,
                                     bool draw_shape_descriptors,
                                     const SegmentationDesc& desc,
                                     const Hierarchy* seg_hier,
                                     cv::Mat* output) {
  CHECK_NOTNULL(output);
  output->setTo(0);
  RenderRegions(highlight_boundary, draw_shape_descriptors, desc,
                HierarchyColorGenerator(hierarchy_level, output->channels(), seg_hier),
                output);
}

}  // namespace segmentation.

#endif // SEGMENTATION_RENDER_H__
