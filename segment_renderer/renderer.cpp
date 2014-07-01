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

#include <fstream>
#include <boost/lexical_cast.hpp>
#include <gflags/gflags.h>
#include <json/json.h>
#include <opencv2/highgui/highgui.hpp>

#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_render.h"
#include "segment_util/segmentation_util.h"
#include "segmentation/segmentation_unit.h"
#include "video_framework/video_reader_unit.h"
#include "video_framework/video_display_unit.h"
#include "video_framework/video_writer_unit.h"

using namespace segmentation;
using namespace video_framework;

DEFINE_string(input_file, "", "Input filename");
DEFINE_double(render_level, 0, "Render level in [0, 1]");
DEFINE_string(output_video_file, "", "Output video filename");
DEFINE_string(output_image_dir, "", "Output image directory.");
DEFINE_string(json_file, "", "Optional Json project file.");
DEFINE_int32(min_output_dim, 0, "If set > 0, scales minimum dimension to specified "
                                "value.");
DEFINE_int32(max_frames, 0, "If set > 0 stops rendering after outputting #max_frames "
                            "frames");
DEFINE_bool(logging, false, "If set outputs logging information");

// Represents a region in project format.
struct ProjectRegion {
  // Attached label.
  std::string label;

  // Color of this region
  cv::Scalar color;

  // Sorted list of region ids.
  std::vector<int> ids;
};

// Parser for project JSON file. Determines set of ProjectRegions.
// Usage:
// JsonProjectParser parser;
// parser.ReadFromFile(FILE_NAME);
// std::vector<ProjectRegion> regions = parser.ProjectRegions();
//
class JsonProjectParser {
 public:
  JsonProjectParser() = default;

  // Returns true if successfully parsed JSON project file.
  bool ReadFromFile(const std::string& file) {
    // Read json to string.
    std::ifstream ifs;
    ifs.open(FLAGS_json_file.c_str(), std::ios_base::in);

    // Crappy quick read in.
    std::string json_data;
    char buffer[1024];
    while (ifs) {
      ifs.read(buffer, 1024);
      json_data += std::string(&buffer[0], &buffer[ifs.gcount()]);
    }

    if (json_data.empty()) {
      LOG(ERROR) << "Could not read from file " << file;
      return false;
    }

    Json::Value root;
    Json::Reader json_reader;
    if(!json_reader.parse(json_data, root)) {
      LOG(ERROR) << "Could not parse json input file.";
      return false;
    }

    Json::Value labels = root["labels"];
    for (int k = 0; k < labels.size(); ++k) {
      ProjectRegion region;
      Json::Value label = labels[k];
      region.label = label["name"].asString();
      
      // Color ordering is alpha = 0xff, red, green, blue;
      int color = boost::lexical_cast<int>(label["color"].asString());
      int red = (color & 0x00ff0000) >> 16;
      int green = (color & 0x0000ff00) >> 8;
      int blue = (color & 0x000000ff);

      region.color = cv::Scalar(red, green, blue);

      Json::Value region_ids = label["region_ids"];
      for (int l = 0; l < region_ids.size(); ++l) {
        Json::Value entry = region_ids[l];
        region.ids.push_back(boost::lexical_cast<int>(entry["id"].asString()));
      }
      std::sort(region.ids.begin(), region.ids.end());
      project_regions_.push_back(region);
    }

    return true;
  }

  const std::vector<ProjectRegion>& ProjectRegions() const { return project_regions_; }

 private:
  std::vector<ProjectRegion> project_regions_;
};

// Color generator for Project regions.
class ProjectRegionColorGenerator {
 public:
  ProjectRegionColorGenerator(const std::vector<ProjectRegion>& project_regions)
    : project_regions_(project_regions) { 
    // Build map of region id to label.
    for (const auto& region : project_regions_) {
      const cv::Scalar* color_ptr = &region.color;
      for (int id : region.ids) {
        region_color_map_[id] = color_ptr;
      }
    }
  }
   
  bool operator()(int overseg_region_id,
                  RegionID* mapped_id,
                  uint8_t* colors) const {
    auto map_iter = region_color_map_.find(overseg_region_id);
    if (map_iter == region_color_map_.end()) {
      return false;
    }

    *mapped_id = RegionID(overseg_region_id);
    const cv::Scalar& color = *map_iter->second;
    colors[0] = (int)color[0];
    colors[1] = (int)color[1];
    colors[2] = (int)color[2];
    return true;
  }

 private: 
  std::vector<ProjectRegion> project_regions_;
  // Maps overseg. region id to corresponding color (storred in project_regions_).

  std::unordered_map<int, const cv::Scalar*> region_color_map_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_logging) {
    FLAGS_logtostderr = true;
  }

  // Get filename from command prompt.
  if (FLAGS_input_file.empty() || 
      (FLAGS_output_video_file.empty() && FLAGS_output_image_dir.empty())) {
    LOG(ERROR) << "Specify --input_file and (--output_video_file or output_image_dir)";
    return 1;
  }

  const bool video_output = !FLAGS_output_video_file.empty();
  const bool image_output = !FLAGS_output_image_dir.empty();

  CHECK(FLAGS_render_level >= 0 && FLAGS_render_level <= 1)
    << "Specify fractional level in [0, 1]";

  std::vector<ProjectRegion> project_regions;
  if (!FLAGS_json_file.empty()) {
    JsonProjectParser parser;
    if (!parser.ReadFromFile(FLAGS_json_file)) {
      LOG(ERROR) << "Could not parse json project file!";
      return -1;
    }
    project_regions = parser.ProjectRegions();
  }

  // Init reader with file and determine resolution.
  SegmentationReader segment_reader(FLAGS_input_file);
  segment_reader.OpenFileAndReadHeaders();

  int frame_width = 0;
  int frame_height = 0;

  segment_reader.SegmentationResolution(&frame_width, &frame_height);
  int width_step = frame_width * 3;
  if (width_step % 4 != 0) {
    width_step += 4 - width_step % 4;
  }

  std::unique_ptr<VideoWriterUnit> writer_unit;
  if (video_output) {
    // Create VideoStream manually to use video renderer if needed.
    StreamSet stream_set;
    stream_set.emplace_back(new VideoStream(frame_width,
                                            frame_height,
                                            width_step,
                                            30,       // fps.
                                            PIXEL_FORMAT_BGR24,
                                            "VideoStream"));

    VideoWriterOptions writer_options;
    // Target bit rate.
    writer_options.bit_rate = frame_width * frame_height * 300;

    if (FLAGS_min_output_dim > 0) {
      writer_options.scale_min_dim = FLAGS_min_output_dim;
    }

    writer_unit.reset(new VideoWriterUnit(writer_options, FLAGS_output_video_file));
    if (!writer_unit->OpenStreams(&stream_set)) {
      LOG(ERROR) << "Could not setup video writer.";
      return -1;
    }
  }

  // Keeps current hierarchy per chunk.
  Hierarchy hierarchy;
  // Absolute render level (constant across all chunks).
  int absolute_level = -1;

  int max_frames = FLAGS_max_frames > 0 ? FLAGS_max_frames : segment_reader.NumFrames();

  for (int f = 0; f < max_frames; ++f) {
    segment_reader.SeekToFrame(f);

    // Read from file.
    SegmentationDesc segmentation;
    segment_reader.ReadNextFrame(&segmentation);

    if (segmentation.hierarchy_size() > 0) {
      hierarchy = segmentation.hierarchy();
      // Convert fractional to constant absolute level.
      if (absolute_level < 0) {
        absolute_level = FLAGS_render_level * (float)hierarchy.size();
      }
    }

    std::unique_ptr<VideoFrame> curr_frame;
    cv::Mat frame_view;
    if (video_output) {              // Use VideoFrame as default backing store.
      curr_frame.reset(new VideoFrame(frame_width,
                                      frame_height,
                                      3,
                                      width_step,
                                      f));
      curr_frame->MatView(&frame_view);
    } else {
      frame_view.create(frame_height, frame_width, CV_8UC3);
    }

    if (project_regions.empty()) {
      RenderRegionsRandomColor(FLAGS_render_level,
                               true,     // With boundaries.
                               false,    // No shape moments.
                               segmentation,
                               &hierarchy,
                               &frame_view);
    } else {
      RenderRegions(false,       // No boundaries.
                    false,       // No shape moments.
                    segmentation,
                    ProjectRegionColorGenerator(project_regions),
                    &frame_view);
    }

    if (image_output) {
      std::string output_file = base::StringPrintf("%s/frame%05d.png",
                                                   FLAGS_output_image_dir.c_str(),
                                                   f);
      cv::imwrite(output_file, frame_view);
    }

    if (video_output) {
      FrameSetPtr frame_set(new FrameSet());
      frame_set->emplace_back(curr_frame.release());
      std::list<FrameSetPtr> unused_output;
      writer_unit->ProcessFrame(frame_set, &unused_output);
    }
  }

  if (video_output) {
    // Close writer.
    std::list<FrameSetPtr> unused_output;
    while (writer_unit->PostProcess(&unused_output));
  }

  return 0;
}
