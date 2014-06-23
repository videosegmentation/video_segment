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

#include "base/base.h"

#include <gflags/gflags.h>

#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_util.h"
#include "segmentation/segmentation_unit.h"
#include "video_framework/video_reader_unit.h"
#include "video_framework/video_display_unit.h"
#include "video_framework/video_writer_unit.h"

using namespace segmentation;
using namespace video_framework;

DEFINE_string(input_file, "", "Input filename");
DEFINE_double(render_level, 0, "Render level in [0, 1]");
DEFINE_string(output_file, "", "Output filename");
DEFINE_int32(min_dim, 0, "If set > 0, scales minimum dimension to specified value.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Get filename from command prompt.
  if (FLAGS_input_file.empty() || FLAGS_output_file.empty()) {
    LOG(ERROR) << "Specify --input_file and --output_file";
    return 1;
  }

  CHECK(FLAGS_render_level >= 0 && FLAGS_render_level <= 1)
    << "Specify fractional level in [0, 1]";

  SegmentationReaderUnitOptions reader_options;
  reader_options.filename = FLAGS_input_file;

  SegmentationReaderUnit seg_reader(reader_options);

  SegmentationRenderUnitOptions render_options;
  render_options.blend_alpha = 1.0f;
  render_options.out_stream_name = "VideoStream";
  render_options.video_stream_name = "";
  render_options.hierarchy_level = FLAGS_render_level;

  SegmentationRenderUnit seg_render(render_options);
  seg_render.AttachTo(&seg_reader);

  VideoWriterOptions writer_options;
  writer_options.bit_rate = 1200000000;
  if (FLAGS_min_dim > 0) {
    writer_options.scale_min_dim = FLAGS_min_dim;
  }

  VideoWriterUnit writer_unit(writer_options, FLAGS_output_file);
  writer_unit.AttachTo(&seg_render);

  if (!seg_reader.PrepareAndRun()) {
    std::cerr << "Error during processing.";
  }

  return 0;
}
