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
#include "base/base_impl.h"

#include <gflags/gflags.h>
#include <opencv2/highgui/highgui.hpp>

#include "segmentation/dense_segmentation.h"
#include "segmentation/region_segmentation.h"
#include "segmentation/segmentation_unit.h"
#include "video_framework/conversion_units.h"
#include "video_framework/flow_reader.h"
#include "video_framework/video_capture_unit.h"
#include "video_framework/video_reader_unit.h"
#include "video_framework/video_pipeline.h"
#include "video_framework/video_writer_unit.h"

#ifndef NO_X_SUPPORT
#ifdef WITH_QT
#include "video_display_qt/video_display_qt_unit.h"
#endif  // WITH_QT
#include "video_framework/video_display_unit.h"
#endif

DEFINE_bool(flow, true, "Determine if optical flow should be computed.");
DEFINE_string(input_file, "", "The input video file to segment. Use 'CAMERA' to read "
                              "from input camera. Adopt chunk_size and chunk_set_size "
                              "in that case to smaller values.");
DEFINE_bool(use_pipeline, true, "If set processing will be done in parallel as pipeline");
DEFINE_bool(pipeline_status, false, "If set outputs pipeline status");
DEFINE_bool(over_segment, false, "If set only dense segmentation will be performed.");
DEFINE_double(display, -1, "If set >=0 displays segmentation at the specified level "
                           "to stream.");

DEFINE_bool(render_and_save, false, "If set renders the resulting segmentation "
                                    "at levels 0.75, 0.4 and 0.1.");
DEFINE_bool(write_to_file, false, "If set write segmentation result to specified "
                                  "input_file + .pb");
DEFINE_bool(logging, false, "If set output various logging information.");
DEFINE_bool(save_flow, false, "If set, buffers flow to file if it does not exist.");
DEFINE_bool(display_flow, false, "If set, buffers flow to file if it does not exist.");
DEFINE_bool(run_on_server, false, "If set uses special settings for server processing. "
                                  "Note: This will override some settings!");
DEFINE_int32(downscale_min_size, 0, "If set > 0, downsamples input video to specifed "
                                    "dimension.");

namespace seg = segmentation;
namespace vf = video_framework;

#ifdef WITH_QT
typedef vf::VideoDisplayQtUnit DisplayUnit;
typedef vf::VideoDisplayQtOptions DisplayUnitOptions;
#else
typedef vf::VideoDisplayUnit DisplayUnit;
typedef vf::VideoDisplayOptions DisplayUnitOptions;
#endif  // WITH_QT

int main(int argc, char** argv) {
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_run_on_server) {
    FLAGS_logging = true;
    FLAGS_use_pipeline = true;
    FLAGS_display = -1;
    FLAGS_render_and_save = false;
    FLAGS_write_to_file= true;
    FLAGS_display_flow = false;
    FLAGS_downscale_min_size = 360;
  }

  if (FLAGS_logging) {
    FLAGS_logtostderr = 1;
  }

  if (FLAGS_input_file.empty()) {
    LOG(ERROR) << "Specified input file is empty.";
    return -1;
  }

  std::vector<std::unique_ptr<vf::VideoPipelineSource>> sources;
  std::vector<std::unique_ptr<vf::VideoPipelineSink>> sinks;

  // For pipeline stats if requested.
  std::vector<std::pair<const vf::VideoPipelineSink*, std::string>> named_sinks;

  std::unique_ptr<vf::VideoReaderUnit> reader;
  std::unique_ptr<vf::VideoCaptureUnit> camera_reader;
  vf::VideoUnit* root = nullptr;
  vf::VideoUnit* input = nullptr;  // Updated throughout graph construction.

  // Determine if flow is being used and if it is being read from file.
  std::string flow_file =
      FLAGS_input_file.substr(0, FLAGS_input_file.find_last_of(".")) + ".flow";
  bool use_flow_from_file = false;
  if (FLAGS_flow) {
    use_flow_from_file = base::FileExists(flow_file);
  }

  // Setup the actual video units.
  if (FLAGS_input_file == "CAMERA") {
    vf::VideoCaptureOptions options;
    options.downscale = 4.0f;
    camera_reader.reset(new vf::VideoCaptureUnit(options));
    root = input = camera_reader.get();
  } else {
    vf::VideoReaderOptions reader_options;
    if (FLAGS_downscale_min_size > 0) {
      reader_options.downscale_size = FLAGS_downscale_min_size;
      reader_options.downscale = vf::VideoReaderOptions::DOWNSCALE_TO_MIN_SIZE;
    }
    reader.reset(new vf::VideoReaderUnit(reader_options, FLAGS_input_file));
    root = input = reader.get();
  }

  if (FLAGS_use_pipeline) {
    sinks.emplace_back(new vf::VideoPipelineSink());
    sinks.back()->AttachTo(input);
    vf::SourceRatePolicy srp;
    if (FLAGS_flow && !use_flow_from_file) {
      // Ensure we are not computing too many flow files.
      srp.respond_to_limit_rate = true;
      srp.rate_scale = 1.25f;
      srp.sink_max_queue_size = 10;
    }
    sources.emplace_back(new vf::VideoPipelineSource(sinks.back().get(),
                                                     nullptr, srp));
    input = sources.back().get();
    named_sinks.push_back(std::make_pair(sinks.back().get(), "Input"));
  }

  std::unique_ptr<vf::LuminanceUnit> lum_unit;
  std::unique_ptr<vf::DenseFlowUnit> dense_flow_unit;
  std::unique_ptr<vf::DenseFlowReaderUnit> dense_flow_reader;

  if (FLAGS_flow) {
    if (use_flow_from_file) {
      dense_flow_reader.reset(
          new vf::DenseFlowReaderUnit(vf::DenseFlowReaderOptions(), flow_file));
      dense_flow_reader->AttachTo(input);
      input = dense_flow_reader.get();
    } else {
      lum_unit.reset(new vf::LuminanceUnit(vf::LuminanceOptions()));
      lum_unit->AttachTo(input);
      vf::DenseFlowOptions flow_options;
      flow_options.flow_iterations = 10;
      flow_options.num_warps = 2;
      flow_options.video_out_stream_name = FLAGS_display_flow ? "RenderedFlow" : "";
      if (FLAGS_save_flow) {
        flow_options.flow_output_file = flow_file;
      }
      dense_flow_unit.reset(new vf::DenseFlowUnit(flow_options));
      dense_flow_unit->AttachTo(lum_unit.get());
      input = dense_flow_unit.get();

      if (FLAGS_use_pipeline) {
        sinks.emplace_back(new vf::VideoPipelineSink());
        sinks.back()->AttachTo(input);
        sources.back()->SetMonitorSink(sinks.back().get());
        sources.emplace_back(new vf::VideoPipelineSource(sinks.back().get()));
        input = sources.back().get();
        named_sinks.push_back(std::make_pair(sinks.back().get(), "Flow"));
      }
    }
  }

  seg::DenseSegmentationUnitOptions dense_seg_unit_options;
  if (!FLAGS_flow) {
    dense_seg_unit_options.flow_stream_name.clear();
  }

  seg::DenseSegmentationOptions dense_seg_options;
  // Last one creates vectorization.
  if (FLAGS_over_segment) {
    dense_seg_options.compute_vectorization = true;
  }
  std::unique_ptr<seg::DenseSegmentationUnit> dense_segment(
      new seg::DenseSegmentationUnit(dense_seg_unit_options, &dense_seg_options));

  dense_segment->AttachTo(input);
  input = dense_segment.get();

  if (FLAGS_use_pipeline) {
    sinks.emplace_back(new vf::VideoPipelineSink());
    sinks.back()->AttachTo(input);
    sources.emplace_back(new vf::VideoPipelineSource(sinks.back().get()));
    input = sources.back().get();
    named_sinks.push_back(std::make_pair(sinks.back().get(), "DenseSeg"));
  }

  std::unique_ptr<seg::RegionSegmentationUnit> region_segment;
  if (!FLAGS_over_segment) {
    seg::RegionSegmentationUnitOptions region_unit_options;
    if (!FLAGS_flow) {
      region_unit_options.flow_stream_name.clear();
    }
    // Memory preserving run on server.
    if (FLAGS_run_on_server) {
      region_unit_options.free_video_frames = true;
      region_unit_options.free_flow_frames = true;
    }

    region_segment.reset(new seg::RegionSegmentationUnit(region_unit_options, nullptr));
    region_segment->AttachTo(input);
    input = region_segment.get();

    if (FLAGS_use_pipeline) {
      sinks.emplace_back(new vf::VideoPipelineSink());
      sinks.back()->AttachTo(input);
      sources.emplace_back(new vf::VideoPipelineSource(sinks.back().get()));
      input = sources.back().get();
      named_sinks.push_back(std::make_pair(sinks.back().get(), "HierSeg"));
    }
  }

  std::unique_ptr<seg::SegmentationRenderUnit> display_render;
  std::unique_ptr<DisplayUnit> segment_display;

  if (FLAGS_display >= 0) {
    seg::SegmentationRenderUnitOptions render_options;
    render_options.blend_alpha = 0.9;
    render_options.concat_with_source = true;
    render_options.hierarchy_level = FLAGS_display;

    display_render.reset(new seg::SegmentationRenderUnit(render_options));
    display_render->AttachTo(input);

    DisplayUnitOptions display_options;
    display_options.stream_name = render_options.out_stream_name;
    segment_display.reset(new DisplayUnit(display_options));
    segment_display->AttachTo(display_render.get());

    input = segment_display.get();
  }

  std::unique_ptr<DisplayUnit> flow_display;

  if (FLAGS_display_flow) {
    DisplayUnitOptions display_options;
    display_options.stream_name = "RenderedFlow";
    flow_display.reset(new DisplayUnit(display_options));
    flow_display->AttachTo(input);

    input = flow_display.get();
  }
  std::vector<std::unique_ptr<seg::SegmentationRenderUnit>> out_render;
  std::vector<std::unique_ptr<vf::VideoWriterUnit>> out_writer;

  if (FLAGS_render_and_save) {
    vector<float> out_levels{0.1f, 0.4f, 0.75f};
    int idx = 0;
    for (float level : out_levels) {
      seg::SegmentationRenderUnitOptions render_options;
      render_options.blend_alpha = 0.9;
      render_options.hierarchy_level = level;
      render_options.out_stream_name = base::StringPrintf("RenderStream%d", idx);
      out_render.emplace_back(new seg::SegmentationRenderUnit(render_options));
      out_render.back()->AttachTo(input);

      vf::VideoWriterOptions writer_options;
      writer_options.stream_name = render_options.out_stream_name;
      writer_options.bit_rate = 40000000;
      writer_options.fraction = 16;
      std::string out_file = base::StringPrintf("%s_render_%0.2f.mp4",
                                                FLAGS_input_file.c_str(), level);
      out_writer.emplace_back(new vf::VideoWriterUnit(writer_options, out_file));
      out_writer.back()->AttachTo(out_render.back().get());

      input = out_writer.back().get();
      ++idx;
    }
  }

  std::unique_ptr<seg::SegmentationWriterUnit> seg_writer;
  if (FLAGS_write_to_file) {
    std::string out_file = FLAGS_input_file + ".pb";
    LOG(INFO) << "Writing result to file " << out_file;
    seg::SegmentationWriterUnitOptions writer_options;
    writer_options.filename = out_file;
    writer_options.remove_rasterization = true;
    seg_writer.reset(new seg::SegmentationWriterUnit(writer_options));
    seg_writer->AttachTo(input);
    input = seg_writer.get();
  }

  std::unique_ptr<vf::VideoPipelineStats> pipeline_stat;
  std::unique_ptr<DisplayUnit> pipeline_stat_display;
  if (FLAGS_use_pipeline && FLAGS_pipeline_status) {
    pipeline_stat.reset(new vf::VideoPipelineStats(vf::VideoPipelineStatsOptions(),
                                                   named_sinks));
    sources.back()->SetIdleUnit(pipeline_stat.get());

    DisplayUnitOptions display_options;
    display_options.stream_name = "VideoPipelineStats";
    pipeline_stat_display.reset(new DisplayUnit(display_options));
    pipeline_stat_display->AttachTo(pipeline_stat.get());
  }

  LOG(INFO) << "Tree layout:";
  root->PrintTree();

  if (!FLAGS_use_pipeline) {
    if (!root->PrepareProcessing()) {
      LOG(ERROR) << "Setup failed.";
    }
    root->Run();
  } else {
    if (!root->PrepareProcessing()) {
      LOG(ERROR) << "Setup failed.";
    }
    vf::VideoPipelineInvoker invoker;
    vf::RatePolicy pipeline_policy;

    // Setup rate policy with 20 fps max processing, startup of 2 frames (flow uses two),
    // and dynamic updates to pipeline every 30 frames.
    const bool use_camera = FLAGS_input_file == "CAMERA";
    pipeline_policy.max_rate = 20;
    pipeline_policy.dynamic_rate = true;
    pipeline_policy.startup_frames = 10;
    pipeline_policy.update_interval = 1;
    pipeline_policy.queue_throttle_threshold = use_camera ? 3 : 10;
    // Guarantee that buffers never go empty in non-camera mode.
    pipeline_policy.dynamic_rate_scale = use_camera ? 0.9 : 1.1;

    // Start the threads.
    // First source is run rate limited.
    invoker.RunRootRateLimited(pipeline_policy, root);

    // Run last source in main thread.
    for (int k = 0; k < sources.size() - 1; ++k) {
      invoker.RunPipelineSource(sources[k].get());
    }

    sources.back()->Run();

    invoker.WaitUntilPipelineFinished();
  }

  LOG(INFO) << "__SEGMENTATION_FINISHED__";
  return 0;
}
