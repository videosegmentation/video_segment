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


#ifndef VIDEO_SEGMENT_VIDEO_DISPLAY_QT_VIDEO_DISPLAY_QT_UNIT_H__
#define VIDEO_SEGMENT_VIDEO_DISPLAY_QT_VIDEO_DISPLAY_QT_UNIT_H__

#include <boost/date_time/posix_time/posix_time.hpp>
#include <queue>

#include "base/base.h"
#include "video_framework/video_unit.h"

class MainWindow;

namespace cv {
  class Mat;
}

namespace segmentation {
  class SegmentationDesc;
}

namespace video_framework {

using boost::posix_time::ptime;

// __ __ __  __ _   _ _   _ _   (_)  _ _    __ _
// \ V  V / / _` | | '_| | ' \  | | | ' \  / _` |
//  \_/\_/  \__,_| |_|   |_|_|  |_| |_|_|  \__, |
//                                         |___/
//
// VideoDisplayQtUnit can only be run in main the thread!

struct VideoDisplayQtOptions {
  std::string stream_name = "VideoStream";

  // Scale input video by this amount for display.
  float upscale = 1.0f;
};

class VideoDisplayQtUnit : public VideoUnit {

public:
  VideoDisplayQtUnit(const VideoDisplayQtOptions& options);
  ~VideoDisplayQtUnit();

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

private:
  VideoDisplayQtOptions options_;

  int video_stream_idx_;
  int display_unit_id_;

  int frame_width_;
  int frame_height_;

  std::unique_ptr<MainWindow> main_window_;
  std::unique_ptr<cv::Mat> scaled_image_;

  // FPS control.
  ptime last_update_time_;
};

struct SegmentationDisplayOptions {
  // Input video stream name. Optional, set to "" for none.
  std::string video_stream_name = "VideoStream";
  // Input segmentation stream. Required.
  std::string segment_stream_name = "SegmentationStream";

  // Specify how much segmentation and input video should be blended together.
  float blend_alpha = 0.5;

  // Hierarchy level to be rendered initially.
  // Specify a fraction f in [0, 1] to select f * max_hierarchy_level.
  float hierarchy_level = 0;
  bool highlight_edges = true;
  bool draw_shape_descriptors = false;

  // Concatenate with original video?
  bool concat_with_source = false;

  // Scale input/segmentation by this amount.
  float upscale = 1.0f;
};


class SegmentationDisplayUnit : public VideoUnit {
public:
  SegmentationDisplayUnit(const SegmentationDisplayOptions& options);
  ~SegmentationDisplayUnit();

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

  void SetFractionalLevel(float level) { fractional_level_ = level; }
  int GetHierarchyLevel();

  std::pair<int, int> GetMouseLoc();
  bool SpaceKeyPressed();

private:
  SegmentationDisplayOptions options_;
  int vid_stream_idx_ = -1;
  int seg_stream_idx_ = -1;

  int frame_width_ = 0;
  int frame_height_ = 0;

  float fractional_level_ = -1;
  int absolute_level_ = -1;

  std::shared_ptr<segmentation::SegmentationDesc> seg_hier_;
  std::unique_ptr<cv::Mat> render_frame_;
  std::unique_ptr<cv::Mat> output_frame_;

  std::unique_ptr<MainWindow> main_window_;
  std::unique_ptr<cv::Mat> scaled_image_;

  // FPS control.
  ptime last_update_time_;
};

}  // namespace video_framework.

#endif  // VIDEO_SEGMENT_VIDEO_DISPLAY_QT_VIDEO_DISPLAY_QT_UNIT_H__
