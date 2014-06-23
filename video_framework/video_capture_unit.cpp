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

#include "video_capture_unit.h"
#include "base/base_impl.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace video_framework {

bool VideoCaptureUnit::OpenStreams(StreamSet* set) {
  capture_.reset(new cv::VideoCapture(0));  // First camera.
  if (!capture_->isOpened()) {
    capture_->open(0);
  }

  // Get capture properties.
  frame_width_ = capture_->get(CV_CAP_PROP_FRAME_WIDTH) / options_.downscale;
  frame_height_ = capture_->get(CV_CAP_PROP_FRAME_HEIGHT) / options_.downscale;

  LOG(INFO) << "Capturing from Camera frames of size "
            << frame_width_ << "x" << frame_height_;

  frame_width_step_ = frame_width_ * 3;
  if (frame_width_step_ % 4) {
    frame_width_step_ += (4 - frame_width_step_ % 4);
  }

  VideoStream* vid_stream = new VideoStream(frame_width_,
                                            frame_height_,
                                            frame_width_step_,
                                            24,       // dummy default value.
                                            PIXEL_FORMAT_BGR24,
                                            options_.video_stream_name);

  set->push_back(shared_ptr<VideoStream>(vid_stream));

  frame_count_ = 0;
  return true;
}

void VideoCaptureUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  // This is a source, ProcessFrame is never called.
}

bool VideoCaptureUnit::PostProcess(list<FrameSetPtr>* append) {
  // Get frame from camera and output.
  FrameSetPtr frame_set(new FrameSet());
  VideoFrame* curr_frame = new VideoFrame(frame_width_,
                                          frame_height_, 3,
                                          frame_width_step_);
  cv::Mat retrieved_frame;
  CHECK(capture_->grab());
  CHECK(capture_->retrieve(retrieved_frame));

  cv::Mat frame_view;
  curr_frame->MatView(&frame_view);
  cv::resize(retrieved_frame, frame_view, frame_view.size());

  frame_set->push_back(shared_ptr<VideoFrame>(curr_frame));
  append->push_back(frame_set);

  // TODO: Listen for ESC to stop processing.

  ++frame_count_;
  return true;
}

}  // namespace video_framework.
