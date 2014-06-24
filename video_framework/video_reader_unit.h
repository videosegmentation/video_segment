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

#ifndef VIDEO_READER_UNIT_H__
#define VIDEO_READER_UNIT_H__

#include "base/base.h"
#include "video_unit.h"

struct AVCodec;
struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace video_framework {

struct VideoReaderOptions {
  int trim_frames = 0;
  std::string stream_name = "VideoStream";
  VideoPixelFormat pixel_format = PIXEL_FORMAT_BGR24;

  // For settings below only downscale will be performed, ie. never upscaling.
  enum DOWNSCALE {
    DOWNSCALE_NONE,
    DOWNSCALE_BY_FACTOR,     // Resizes each dimension by downscale_factor.

    DOWNSCALE_TO_MIN_SIZE,   // Resizes minimum dimension to downscale_size.
    DOWNSCALE_TO_MAX_SIZE,   // Resizes maximum dimension to downscale_size.
  };

  DOWNSCALE downscale = DOWNSCALE_NONE;

  float downscale_factor = 1.0f;
  int downscale_size = 0;
};

class VideoReaderUnit : public VideoUnit {
 public:
  VideoReaderUnit(const VideoReaderOptions& options,
                  const std::string& video_file);
  ~VideoReaderUnit();

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

  // Experimental (might not seek to correct locations).
  virtual bool SeekImpl(int64_t pts);
  bool PrevFrame();

  // Can be positive or negative.
  bool SkipFrames(int frame_offset);

 private:
  // Returns allocated VideoFrame (ownership passed to caller).
  // Returns NULL if end of file is reached.
  VideoFrame* ReadNextFrame();
  // Used by above function is a frame could be decoded to get the actual VideoFrame.
  VideoFrame* ReadNextFrameImpl(const AVPacket& packet);

 private:
  VideoReaderOptions options_;
  std::string video_file_;

  int video_stream_idx_;
  int bytes_per_pixel_;

  int frame_num_ = 0;

  int frame_width_ = 0;
  int frame_height_ = 0;

  int output_width_ = 0;
  int output_height_ = 0;
  int output_width_step_ = 0;
  float downscale_factor_ = 1.0;

  AVCodec*  codec_;
  AVCodecContext* codec_context_;
  AVFormatContext* format_context_;
  SwsContext* sws_context_;

  AVFrame* frame_yuv_;
  AVFrame* frame_bgr_;

  // This is used for seeking to a specific frame.
  // If next_packet_ is set, it is read instead of decode being called.
  std::unique_ptr<AVPacket> next_packet_;

  uint64_t current_pos_ = 0;
  double fps_;

  bool used_as_root_ = true;
  static bool ffmpeg_initialized_;
};

}  // namespace video_framework.

#endif
