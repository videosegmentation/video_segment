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

#include "segmentation_render.h"
#include "base/base_impl.h"

namespace segmentation {

HierarchyColorGenerator::HierarchyColorGenerator(int hierarchy_level,
                                                 int channels,
                                                 const Hierarchy* hierarchy)
    : hierarchy_(hierarchy),
      hierarchy_level_(hierarchy_level),
      channels_(channels) {
  if (hierarchy_level > 0 && hierarchy == nullptr) {
    hierarchy_level_ = 0;
    LOG(WARNING) << "Requested level > 0, but hierarchy is NULL. Truncated to zero.";
  }

  // Might truncate desired level.
  if (hierarchy != nullptr && hierarchy_level_ >= hierarchy_->size()) {
    hierarchy_level_ = hierarchy_->size() - 1;
    LOG(WARNING) << "Requested level " << hierarchy_level << " not present in "
                 << "hierarchy. Truncated to " << hierarchy_level_ << "\n";
  }
}

bool HierarchyColorGenerator::operator()(int overseg_region_id,
                                         RegionID* mapped_id,
                                         uint8_t* colors) const {
  CHECK_NOTNULL(mapped_id);
  CHECK_NOTNULL(colors);
  int region_id = overseg_region_id;

  if (hierarchy_level_ > 0) {
    region_id = GetParentId(overseg_region_id, 0, hierarchy_level_, *hierarchy_);
  }

  *mapped_id = RegionID(region_id, hierarchy_level_);

  srand(region_id);
  for (int c = 0; c < channels_; ++c) {
    colors[c] = (uint8_t) (rand() % 255);
  }

  return true;
}

}  // namespace segmentation.
