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

#ifndef VIDEO_SEGMENT_VIDEO_FRAMEWORK_CONVERSION_UNITS_H__
#define VIDEO_SEGMENT_VIDEO_FRAMEWORK_CONVERSION_UNITS_H__

#include "video_framework/video_unit.h"
#include "base/base.h"

namespace video_framework {

struct LuminanceOptions {
  std::string video_stream_name = "VideoStream";
  std::string luminance_stream_name = "LuminanceStream";
};

class LuminanceUnit : public VideoUnit {
public:
  LuminanceUnit(const LuminanceOptions& options) : options_(options) { }

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);
private:
  LuminanceOptions options_;

  int video_stream_idx_;
  VideoPixelFormat pixel_format_;

  int frame_width_;
  int frame_height_;
  int width_step_;
};

struct FlipBGROptions {
  std::string video_stream_name = "VideoStream";
  std::string output_stream_name = "FlippedVideoStream";
};

class FlipBGRUnit : public VideoUnit {
public:
  FlipBGRUnit(const FlipBGROptions& options) : options_(options) { }

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);
private:
  FlipBGROptions options_;
  int video_stream_idx_;

  int frame_width_;
  int frame_height_;
  int width_step_;
};

// Applies a global color twist weight[color] * color_value + offset[value],
// clamped to [0, 255]. Operates in-place on the specified video stream.
struct ColorTwistOptions {
  std::vector<float> weights;
  std::vector<float> offsets;
  std::string video_stream_name = "VideoStream";
};

class ColorTwist : public VideoUnit {
public:
  ColorTwist(const ColorTwistOptions& options) : options_(options) {}

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append) { return false; }

protected:
  ColorTwistOptions options_;
  int video_stream_idx_;

  int frame_width_;
  int frame_height_;
  int frame_num_;
};

}  // namespace video_framework.

#endif // VIDEO_SEGMENT_VIDEO_FRAMEWORK_CONVERSION_UNITS_H__
