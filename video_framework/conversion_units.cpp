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

#include "conversion_units.h"

#include "base/base_impl.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "imagefilter/image_util.h"
using namespace imagefilter;

namespace video_framework {

bool LuminanceUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  video_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);
  CHECK_GE(video_stream_idx_, 0) << "Could not find video stream!\n";

  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();
  pixel_format_ = vid_stream.pixel_format();

  if (pixel_format_ != PIXEL_FORMAT_RGB24 &&
      pixel_format_ != PIXEL_FORMAT_BGR24 &&
      pixel_format_ != PIXEL_FORMAT_RGBA32) {
    LOG(ERROR) << "Unsupported input pixel format!\n";
    return false;
  }

  width_step_ = frame_width_;
  if (width_step_ % 4) {
    width_step_ = width_step_ + (4 - width_step_ % 4);
    DCHECK_EQ(width_step_ % 4, 0);
  }

  VideoStream* lum_stream = new VideoStream(frame_width_,
                                            frame_height_,
                                            width_step_,
                                            vid_stream.fps(),
                                            PIXEL_FORMAT_LUMINANCE,
                                            options_.luminance_stream_name);
  set->push_back(shared_ptr<VideoStream>(lum_stream));
  return true;
}

void LuminanceUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  const VideoFrame* frame = input->at(video_stream_idx_)->AsPtr<VideoFrame>();

  cv::Mat image;
  frame->MatView(&image);

  VideoFrame* lum_frame = new VideoFrame(frame_width_, frame_height_, 1, width_step_,
                                         frame->pts());

  cv::Mat lum_image;
  lum_frame->MatView(&lum_image);

  // Map from our pixel format to OpenCV format.
  int convert_flags;

  switch (pixel_format_) {
    case PIXEL_FORMAT_RGB24:
      convert_flags = CV_RGB2GRAY;
      break;
    case PIXEL_FORMAT_BGR24:
      convert_flags = CV_BGR2GRAY;
      break;
    case PIXEL_FORMAT_RGBA32:
      convert_flags = CV_RGBA2GRAY;
      break;
    default:
      LOG(FATAL) << "Unsupported input pixel format.\n";
      return;
  }

  cv::cvtColor(image, lum_image, convert_flags);

  input->push_back(shared_ptr<DataFrame>(lum_frame));
  output->push_back(input);
}

bool LuminanceUnit::PostProcess(list<FrameSetPtr>* append) {
  return false;
}

bool FlipBGRUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  video_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

  CHECK_GE(video_stream_idx_, 0) << "Could not find Video stream!\n";

  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();
  width_step_ = vid_stream.width_step();

  VideoPixelFormat pixel_format = vid_stream.pixel_format();

  if (pixel_format != PIXEL_FORMAT_RGB24 &&
      pixel_format != PIXEL_FORMAT_BGR24) {
    LOG(ERROR) << "Unsupported input pixel format!\n";
    return false;
  }

  VideoPixelFormat output_format = pixel_format == PIXEL_FORMAT_RGB24 ?
  PIXEL_FORMAT_BGR24 : PIXEL_FORMAT_RGB24;

  VideoStream* output_stream = new VideoStream(frame_width_,
                                               frame_height_,
                                               width_step_,
                                               vid_stream.fps(),
                                               output_format,
                                               options_.output_stream_name);

  set->push_back(shared_ptr<VideoStream>(output_stream));
  return true;
}

void FlipBGRUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  const VideoFrame* frame = input->at(video_stream_idx_)->AsPtr<VideoFrame>();
  CHECK_NOTNULL(frame);

  cv::Mat image;
  frame->MatView(&image);

  VideoFrame* output_frame = new VideoFrame(frame_width_, frame_height_,
                                            frame->channels(), width_step_,
                                            frame->pts());

  cv::Mat out_image;
  output_frame->MatView(&out_image);

  cv::cvtColor(image, out_image, CV_RGB2BGR);

  input->push_back(shared_ptr<DataFrame>(output_frame));
  output->push_back(input);
}

bool FlipBGRUnit::PostProcess(list<FrameSetPtr>* append) {
  return false;
}

bool ColorTwist::OpenStreams(StreamSet* set) {
  video_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find Video stream!\n";
    return false;
  }

  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();
  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();

  return true;
}

namespace {

float clamp_uint8(float c) {
  return std::max(0.f, std::min(255.f, c));
}

}  // namespace.

void ColorTwist::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  VideoFrame* frame = input->at(video_stream_idx_)->AsMutablePtr<VideoFrame>();

  cv::Mat image;
  frame->MatView(&image);
  const vector<float>& w = options_.weights;
  const vector<float>& o = options_.offsets;

  for (int i = 0; i < frame_height_; ++i) {
    uint8_t* color_ptr = image.ptr<uint8_t>(i);
    for (int j = 0; j < frame_width_; ++j, color_ptr +=3) {
      color_ptr[0] = (uint8_t)clamp_uint8((float)color_ptr[0] * w[0] + o[0]);
      color_ptr[1] = (uint8_t)clamp_uint8((float)color_ptr[1] * w[1] + o[1]);
      color_ptr[2] = (uint8_t)clamp_uint8((float)color_ptr[2] * w[2] + o[2]);
    }
  }

  output->push_back(input);
}

}  // namespace video_framework.
