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



#include <gflags/gflags.h>
#include <iostream>

#include "base/base_impl.h"
#include "video_framework/video_display_unit.h"
#include "video_framework/conversion_units.h"
#include "video_framework/flow_reader.h"    // Rename.
#include "video_framework/video_reader_unit.h"
#include "video_framework/video_pipeline.h"

using namespace video_framework;

DEFINE_string(input, "", "The input file to process.");
DEFINE_bool(logging, false, "If set output various logging information.");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_input.empty()) {
    std::cout << "Usage: video_example --input=VIDEO_FILE\n";
    return -1;
  }

  if (FLAGS_logging) {
    FLAGS_logtostderr = 1;
  }

  std::string video_file = FLAGS_input;
  std::cout << "Using video file: " << video_file << "\n";

  // Single threaded usage.
  /////////////////////////
  VideoReaderUnit reader(VideoReaderOptions(), video_file);
  VideoDisplayUnit display((VideoDisplayOptions()));
  display.AttachTo(&reader);

  if (!reader.PrepareProcessing()) {
    LOG(ERROR) << "Video framework setup failed.";
    return -1;
  }

  // Run with rate limitation.
  RatePolicy rate_policy;
  // Speed up playback for fun :)
  rate_policy.max_rate = 45;

  // This call will block and return when the whole has been displayed.
  if (!reader.RunRateLimited(rate_policy)) {
    LOG(ERROR) << "Could not process video file.";
    return -1;
  }

  // Multithreaded pipeline usage.
  ////////////////////////////////

  // First pipeline stage.
  VideoReaderUnit reader_multi(VideoReaderOptions(), video_file);

  // Convert to luminance.
  LuminanceUnit luminance((LuminanceOptions()));
  luminance.AttachTo(&reader_multi);

  // Second pipeline stage.
  // Compute flow.
  VideoPipelineSink first_sink;
  first_sink.AttachTo(&luminance);

  VideoPipelineSource second_source(&first_sink);
  DenseFlowOptions flow_options;
  flow_options.flow_iterations = 10;
  flow_options.num_warps = 4;
  flow_options.video_out_stream_name = "RenderedFlow";
  DenseFlowUnit dense_flow(flow_options);
  dense_flow.AttachTo(&second_source);

  // Third pipeline stage.
  // Output video and flow.
  VideoPipelineSink second_sink;
  second_sink.AttachTo(&dense_flow);

  VideoPipelineSource third_source(&second_sink);
  VideoDisplayUnit display_video((VideoDisplayOptions()));
  display_video.AttachTo(&third_source);
  VideoDisplayOptions display_options;
  display_options.stream_name = "RenderedFlow";
  VideoDisplayUnit display_flow(display_options);
  display_flow.AttachTo(&display_video);

  // Run framework in multi-threaded fashion.
  if (!reader_multi.PrepareProcessing()) {
    LOG(ERROR) << "Setup failed.";
  }

  // Run pipeline with 3 sources to be run in 3 different threads:
  // reader_multi, second_source, third_source.
  VideoPipelineInvoker invoker;
  RatePolicy pipeline_policy;

  // Setup rate policy with 15 fps max processing, startup of 5 frames (flow uses two),
  // and dynamic updates to pipeline every 3 frames.
  pipeline_policy.max_rate = 15;
  pipeline_policy.dynamic_rate = true;
  pipeline_policy.startup_frames = 5;
  pipeline_policy.update_interval = 3;

  // Start the threads.
  // First source is run rate limited.
  invoker.RunRootRateLimited(pipeline_policy, &reader_multi);
  // Second is run not rate limited (other pipeline sources rates are
  // controlled via root).
  invoker.RunPipelineSource(&second_source);

  // Visualization has to be run in main thread (so don't use invoker).
  // would crash: invoker.RunPipelineSource(&third_source);
  // as it is creating a new thread.
  third_source.Run();

  invoker.WaitUntilPipelineFinished();

  return 0;
}
