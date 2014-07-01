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

#include "base/base_impl.h"
#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_util.h"
#include "segment_util/segmentation_render.h"

using namespace segmentation;

DEFINE_string(input, "", "The input segmentation protobuffer (.pb). REQUIRED");
DEFINE_string(window_name, "", "Use different window names to support "
                               "multiple viewer sessions.");

void ViewerFramePosChanged(int pos, void* viewer);
void ViewerHierarchyLevelChanged(int pos, void* viewer);

class Viewer {
 public:
  Viewer(const std::string& filename, const std::string& window_name)
      : filename_(filename), window_name_(window_name) {
  }

  ~Viewer() {
  }

  bool ReadFile() {
    // Read segmentation file.
    segment_reader_.reset(new SegmentationReader(filename_));
    if (!segment_reader_->OpenFileAndReadHeaders()) {
      return false;
    }
    frame_pos_ = 0;

    // Read first frame, it contains the hierarchy.
    seg_hierarchy_.reset(new SegmentationDesc);
    segment_reader_->ReadNextFrame(seg_hierarchy_.get());
    hierarchy_pos_ = 0;
    frame_num_ = segment_reader_->NumFrames();
    frame_width_ = seg_hierarchy_->frame_width();
    frame_height_ = seg_hierarchy_->frame_height();

    frame_buffer_.create(frame_height_, frame_width_, CV_8UC3);

    LOG(INFO) << "Video resolution: " << frame_width_ << "x" << frame_height_;
    LOG(INFO) << "Segmentation contains " << frame_num_ << " frames.";
    return true;
  }

  void CreateUI() {
    // Create OpenCV window.
    cv::namedWindow(window_name_.c_str(), CV_WINDOW_NORMAL);

    cv::createTrackbar("frame_pos",
                       window_name_,
                       &frame_pos_,
                       segment_reader_->NumFrames() - 1,
                       &ViewerFramePosChanged,
                       this);

    cv::createTrackbar("hier_level",
                       window_name_,
                       &hierarchy_level_,
                       seg_hierarchy_->hierarchy_size() - 1,
                       &ViewerHierarchyLevelChanged,
                       this);
    RenderCurrentFrame(0);
    cv::imshow(window_name_, frame_buffer_);
  }

  void Run() {
    int key_value = 0;

    // Yotam Doron recommended this kind of loop.
    while (1) {
      key_value = cv::waitKey(30) & 0xFF;
      if (key_value == 27) {   // Esc.
        break;
      }

      if (playing_) {
        FramePosChanged((frame_pos_ + 1) % frame_num_);
      }

      switch (key_value) {
        case 3:
        case 110:        // Right key.
          FramePosChanged((frame_pos_ + 1) % frame_num_);
          break;
        case 2:
        case 112:        // Left key.
          FramePosChanged((frame_pos_ - 1 + frame_num_) % frame_num_);
          break;
        case 32:         // Space.
          playing_ = !playing_;
        default:
          break;
      }
    }
  }

  void FramePosChanged(int pos) {
    frame_pos_ = pos;
    RenderCurrentFrame(frame_pos_);
    cv::imshow(window_name_, frame_buffer_);
    cv::setTrackbarPos("frame_pos", window_name_, pos);
  }

  void HierarchyLevelChanged(int level) {
    hierarchy_level_ = level;
    RenderCurrentFrame(frame_pos_);
    cv::imshow(window_name_, frame_buffer_);
    cv::setTrackbarPos("hier_level", window_name_, level);
  }

  // Returns number of hierarchy_levels.
  void RenderCurrentFrame(int frame_number) {
    segment_reader_->SeekToFrame(frame_number);

    // Read from file.
    SegmentationDesc segmentation;
    segment_reader_->ReadNextFrame(&segmentation);

    // Update hierarchy is necessary.
    if (hierarchy_pos_ != segmentation.hierarchy_frame_idx()) {
      hierarchy_pos_ = segmentation.hierarchy_frame_idx();
      segment_reader_->SeekToFrame(hierarchy_pos_);

      // Read from file.
      segment_reader_->ReadNextFrame(seg_hierarchy_.get());
    }

    // Render segmentation at specified level.
    RenderRegionsRandomColor(hierarchy_level_,
                             true,
                             false,
                             segmentation,
                             &seg_hierarchy_->hierarchy(),
                             &frame_buffer_);
  }

 private:
  std::string filename_;

  // Frame width and height.
  int frame_width_ = 0;
  int frame_height_ = 0;
  // Number of frames in segmentation.
  int frame_num_ = 0;

  // Slider positions.
  int hierarchy_level_ = 0;
  int frame_pos_ = 0;

  std::unique_ptr<SegmentationReader> segment_reader_;
  std::unique_ptr<SegmentationDesc> seg_hierarchy_;
  int hierarchy_pos_ = 0;

  // Render target.
  cv::Mat frame_buffer_;

  // Indicates if automatic playing is set.
  bool playing_ = false;

  // The name used to identify the GTK window
  std::string window_name_;
};

// Callbacks for OpenCV.
void ViewerFramePosChanged(int pos, void* viewer) {
  reinterpret_cast<Viewer*>(viewer)->FramePosChanged(pos);
}
void ViewerHierarchyLevelChanged(int pos, void* viewer) {
  reinterpret_cast<Viewer*>(viewer)->HierarchyLevelChanged(pos);
}

int main(int argc, char** argv) {
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_input.empty()) {
    std::cout << "Usage: segment_viewer -input=FILE_NAME\n";
    return 1;
  }

  std::string window_name;
  if (FLAGS_window_name.empty()) {
    window_name = argv[2];
  } else {
    // if not set as the filename
    size_t temp = FLAGS_input.find_last_of("/\\");
    window_name = FLAGS_input.substr(temp+1);
  }

  Viewer viewer(FLAGS_input, window_name);
  if (!viewer.ReadFile()) {
    LOG(ERROR) << "Could not read file!";
    return 1;
  }

  viewer.CreateUI();
  viewer.Run();

   return 0;
}
