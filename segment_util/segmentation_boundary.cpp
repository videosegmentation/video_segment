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


#include "segmentation_boundary.h"
#include "base/base_impl.h"

namespace segmentation {

namespace {

// Sets buffer to one for each pixel of the scanline.
// Returns x-range of rendered pixels.
pair<int, int> RenderScanInterval(const ScanInterval& scan_inter,
                                  uint8_t* buffer) {
  pair<int, int> range(1e6, -1e6);
  const int interval_sz = scan_inter.right_x() - scan_inter.left_x() + 1;
  range.first = std::min<int>(range.first, scan_inter.left_x());
  range.second = std::max<int>(range.second, scan_inter.right_x());
  memset(buffer + scan_inter.left_x(), 1, interval_sz);
  return range;
}

// Renders all scanlines for a specific y coordinate for all passed rasterizations.
// Returns x-range of rendered pixels.
pair<int, int> RenderScanlines(const vector<const Rasterization*>& rasters,
                               int y,
                               uint8_t* buffer) {
  pair<int, int> range(1e6, -1e6);
  for (const auto r_ptr : rasters) {
    auto s = LocateScanLine(y, *r_ptr);
    while (s != r_ptr->scan_inter().end() && s->y() == y) {
      pair<int, int> ret_range = RenderScanInterval(*s++, buffer);
      range.first = std::min<int>(range.first, ret_range.first);
      range.second = std::max<int>(range.second, ret_range.second);
    }
  }
  return range;
}

}  // namespace.

void GetBoundary(const Rasterization& raster,
                 int frame_width,
                 bool inner_boundary,
                 vector<uchar>* buffer,
                 RegionBoundary* boundary) {
  vector<const SegmentationDesc::Rasterization*> rasters;
  rasters.push_back(&raster);
  GetBoundary(rasters, frame_width, inner_boundary, buffer, boundary);
}

void GetBoundary(const vector<const Rasterization*>& rasters,
                 int frame_width,
                 bool inner_boundary,
                 vector<uint8_t>* buffer,
                 RegionBoundary* boundary) {
  CHECK_NOTNULL(boundary);
  CHECK_NOTNULL(buffer);

  if (rasters.empty()) {
    return;
  }

  const int rad = 1;
  const int width_step = frame_width + 2 * rad;
  if (buffer->size() < 3 * width_step) {
    LOG(WARNING) << "Buffer too small, resize to " << 3 * width_step;
    buffer->resize(3 * width_step);
  }

  // Zero buffer.
  std::fill(buffer->begin(), buffer->end(), 0);

  // Set up scanline pointers.
  uint8_t* prev_ptr = &(*buffer)[rad];
  uint8_t* curr_ptr = &(*buffer)[rad + width_step];
  uint8_t* next_ptr = &(*buffer)[rad + 2 * width_step];

  // Determine min and max y.
  int min_y = 1e7;
  int max_y = -1e7;

  for (const auto raster_ptr : rasters) {
    CHECK_NOTNULL(raster_ptr);
    if (raster_ptr->scan_inter_size() == 0) {
      LOG(WARNING) << "Empty rasterization passed to GetBoundary.";
    } else {
      min_y = std::min<int>(raster_ptr->scan_inter(0).y(), min_y);
      max_y = std::max<int>(raster_ptr->scan_inter(raster_ptr->scan_inter_size() - 1).y(),
                                                   max_y);
    }
  }

  // Render first scanline.
  vector<pair<int, int> > ranges;
  if (inner_boundary) {
    ranges.push_back(RenderScanlines(rasters, min_y, curr_ptr));
  } else {
    ranges.push_back(RenderScanlines(rasters, min_y, next_ptr));
  }

  int shift = inner_boundary ? 0 : 1;

  const int range_size = inner_boundary ? 1 : 3;
  for (int y = min_y - shift; y <= max_y + shift; ++y) {
    // Clear buffer.
    memset(next_ptr, 0, frame_width);

    if (y < max_y) {
      ranges.push_back(RenderScanlines(rasters, y + 1, next_ptr));
    }

    int min_range = 1e7;
    int max_range = -1e7;

    for (int k = 0; k < ranges.size(); ++k) {
      min_range = std::min(min_range, ranges[k].first);
      max_range = std::max(max_range, ranges[k].second);
    }

    if (min_range <= max_range) {
      const uint8_t* prev_x_ptr = prev_ptr + min_range - shift;
      const uint8_t* curr_x_ptr = curr_ptr + min_range - shift;
      const uint8_t* next_x_ptr = next_ptr + min_range - shift;

      for (int x = min_range;
           x <= max_range + 2 * shift;
           ++x, ++prev_x_ptr, ++curr_x_ptr, ++next_x_ptr) {

        // A point is a boundary point if the current value for it is set,
        // but one of its 4 neighbors is not (inner_boundary, else inverted criteria).
        if (inner_boundary) {
          if (curr_x_ptr[0] &&
              (!curr_x_ptr[-1] || !curr_x_ptr[1] || !prev_x_ptr[0] || !next_x_ptr[0])) {
            boundary->push_back(BoundaryPoint(x, y));
          }
        } else {
          if (!curr_x_ptr[0] &&
              (curr_x_ptr[-1] || curr_x_ptr[1] || prev_x_ptr[0] || next_x_ptr[0])) {
            boundary->push_back(BoundaryPoint(x, y));
          }
        }
      }
    }

    // Swap with wrap around.
    std::swap(prev_ptr, curr_ptr);
    std::swap(curr_ptr, next_ptr);
    if (ranges.size() >= range_size) {
      ranges.erase(ranges.begin());
    }
  }
}

}  // namespace segmentation.
