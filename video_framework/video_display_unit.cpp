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

#include "video_display_unit.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "base/base_impl.h"

namespace video_framework {

int VideoDisplayUnit::display_unit_count;

VideoDisplayUnit::VideoDisplayUnit(const VideoDisplayOptions& options)
    : options_(options) {
  display_unit_id_ = display_unit_count++;
}

VideoDisplayUnit::~VideoDisplayUnit() {
}

bool VideoDisplayUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  video_stream_idx_ = FindStreamIdx(options_.stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find Video stream!\n";
    return false;
  }

  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  const int frame_width = vid_stream.frame_width();
  const int frame_height = vid_stream.frame_height();

  if (options_.output_scale != 1.0f) {
    frame_buffer_.reset(new cv::Mat(frame_height * options_.output_scale,
                                    frame_width * options_.output_scale,
                                    CV_8UC3));
  }

  // Open display window.
  std::ostringstream os;
  os << "VideoDisplayUnit_" << display_unit_id_;
  window_name_ = os.str();

  cv::namedWindow(window_name_);
  cv::waitKey(10);
  return true;
}

void VideoDisplayUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  const VideoFrame* frame = input->at(video_stream_idx_)->AsPtr<VideoFrame>();

  cv::Mat image;
  frame->MatView(&image);

  if (frame_buffer_) {
    cv::resize(image, *frame_buffer_, frame_buffer_->size());
    cv::imshow(window_name_.c_str(), *frame_buffer_);
  } else {
    cv::imshow(window_name_.c_str(), image);
  }

  output->push_back(input);
  cv::waitKey(1);
}

bool VideoDisplayUnit::PostProcess(list<FrameSetPtr>* append) {
  cv::destroyWindow(window_name_);
  return false;
}

}  // namespace video_framework.
