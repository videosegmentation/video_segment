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

#include "video_framework/video_reader_unit.h"

#include <gflags/gflags.h>

#include "base/base_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#ifdef __cplusplus
}
#endif

DEFINE_int32(trim_to, 0, "If set > 0, processing stops after trim_to frames have been "
                         "processed.");

namespace video_framework {

bool VideoReaderUnit::ffmpeg_initialized_ = false;

VideoReaderUnit::VideoReaderUnit(const VideoReaderOptions& options,
                                 const string& video_file)
    : options_(options),
      video_file_(video_file) {
  codec_context_ = 0;
  format_context_ = 0;
  frame_bgr_ = 0;
  frame_yuv_ = 0;

  if (FLAGS_trim_to > 0) {
    options_.trim_frames = FLAGS_trim_to;
  }
}

VideoReaderUnit::~VideoReaderUnit() {
  if (frame_bgr_) {
    av_free(frame_bgr_->data[0]);
    av_free(frame_bgr_);
  }

  if (frame_yuv_)
    av_free(frame_yuv_);

  if (codec_context_)
    avcodec_close(codec_context_);
  if (format_context_)
    avformat_close_input(&format_context_);
}

bool VideoReaderUnit::OpenStreams(StreamSet* set) {
  // Setup FFMPEG.
  if (!ffmpeg_initialized_) {
    ffmpeg_initialized_ = true;
    av_register_all();
  }

  // Open video file.
  AVFormatContext* format_context = nullptr;
  if (avformat_open_input (&format_context, video_file_.c_str(), nullptr, nullptr) != 0) {
    LOG(ERROR) << "Could not open file: " << video_file_;
    return false;
  }

  if (avformat_find_stream_info(format_context, nullptr) < 0) {
    LOG(ERROR) << video_file_ << " is not a valid movie file.";
    return false;
  }

  // Get video stream index.
  video_stream_idx_ = -1;

  for (uint i = 0; i < format_context->nb_streams; ++i) {
    if (format_context->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_idx_ = i;
      break;
    }
  }

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find video stream in " << video_file_;
    return false;
  }

  AVCodecContext* codec_context = format_context->streams[video_stream_idx_]->codec;
  AVCodec* codec = avcodec_find_decoder (codec_context->codec_id);

  if (!codec) {
    LOG(ERROR) << "Unsupported codec for file " << video_file_;
    return false;
  }

  if (avcodec_open2(codec_context, codec, nullptr) < 0) {
    LOG(ERROR) << "Could not open codec";
    return false;
  }

  AVStream* av_stream = format_context->streams[video_stream_idx_];
  fps_ = av_q2d(av_stream->avg_frame_rate);
  LOG(INFO) << "Frame rate: " << fps_;

  // if av_q2d wasn't able to figure out the frame rate, set it 24
  if (fps_ != fps_) {
    LOG(WARNING) << "Can't figure out frame rate - Defaulting to 24";
    fps_ = 24;
  }

  // Limit to meaningful values. Sometimes avg_frame_rate.* holds garbage.
  if (fps_ < 5) {
    LOG(WARNING) << "Capping video fps_ of " << fps_ << " to " << 5;
    fps_ = 5;
  }

  if (fps_ > 60) {
    LOG(WARNING) << "Capping video fps_ of " << fps_ << " to " << 60;
    fps_ = 60;
  }

  bytes_per_pixel_ = PixelFormatToNumChannels(options_.pixel_format);
  frame_width_ = codec_context->width;
  frame_height_ = codec_context->height;

  switch (options_.downscale) {
    case VideoReaderOptions::DOWNSCALE_NONE:
      output_width_ = frame_width_;
      output_height_ = frame_height_;
      downscale_factor_ = 1.0f;
      break;

    case VideoReaderOptions::DOWNSCALE_BY_FACTOR:
      if (options_.downscale_factor > 1.0f) {
        LOG(ERROR) << "Only downscaling is supported.";
        return false;
      }

      downscale_factor_ = options_.downscale_factor;
      output_width_ = std::ceil(frame_width_ * downscale_factor_);
      output_height_ = std::ceil(frame_height_ * downscale_factor_);
      break;

    case VideoReaderOptions::DOWNSCALE_TO_MIN_SIZE:
      downscale_factor_ = std::max(options_.downscale_size * (1.0f / frame_width_),
                                   options_.downscale_size * (1.0f / frame_height_));
      // Cap to downscaling.
      downscale_factor_ = std::min(1.0f, downscale_factor_);
      output_width_ = std::ceil(frame_width_ * downscale_factor_);
      output_height_ = std::ceil(frame_height_ * downscale_factor_);
      break;

    case VideoReaderOptions::DOWNSCALE_TO_MAX_SIZE:
      downscale_factor_ = std::min(options_.downscale_size * (1.0f / frame_width_),
                                   options_.downscale_size * (1.0f / frame_height_));
      // Cap to downscaling.
      downscale_factor_ = std::min(1.0f, downscale_factor_);
      output_width_ = std::ceil(frame_width_ * downscale_factor_);
      output_height_ = std::ceil(frame_height_ * downscale_factor_);
      break;
  }

  if (downscale_factor_ != 1.0) {
    LOG(INFO) << "Downscaling by factor " << downscale_factor_
              << " from " << frame_width_ << ", " << frame_height_
              << " to " << output_width_ << ", " << output_height_;
  }

  // Force even resolutions.
  output_width_ += output_width_ % 2;
  output_width_step_ = output_width_ * bytes_per_pixel_;

  // Pad width_step to be a multiple of 4.
  if (output_width_step_ % 4 != 0) {
    output_width_step_ += 4 - output_width_step_ % 4;
    DCHECK_EQ(output_width_step_ % 4, 0);
  }

  // Save some infos for later use.
  codec_ = codec;
  codec_context_ = codec_context;
  format_context_ = format_context;

  // Allocate temporary structures.
  frame_yuv_ = av_frame_alloc();
  frame_bgr_ = av_frame_alloc();

  if (!frame_yuv_ || !frame_bgr_) {
    LOG(ERROR) << "Could not allocate AVFrames.";
    return false;
  }

  int pix_fmt;
  switch (options_.pixel_format) {
    case PIXEL_FORMAT_RGB24:
      pix_fmt = PIX_FMT_RGB24;
      break;
    case PIXEL_FORMAT_BGR24:
      pix_fmt = PIX_FMT_BGR24;
      break;
    case PIXEL_FORMAT_ARGB32:
      pix_fmt = PIX_FMT_ARGB;
      break;
    case PIXEL_FORMAT_ABGR32:
      pix_fmt = PIX_FMT_ABGR;
      break;
    case PIXEL_FORMAT_RGBA32:
      pix_fmt = PIX_FMT_RGBA;
      break;
    case PIXEL_FORMAT_BGRA32:
      pix_fmt = PIX_FMT_BGRA;
      break;
    case PIXEL_FORMAT_YUV422:
      pix_fmt = PIX_FMT_YUYV422;
      break;
    case PIXEL_FORMAT_LUMINANCE:
      pix_fmt = PIX_FMT_GRAY8;
      break;
  }

  uint8_t* bgr_buffer = (uint8_t*)av_malloc(avpicture_get_size((::PixelFormat)pix_fmt,
                                                               output_width_,
                                                               output_height_));

  avpicture_fill((AVPicture*)frame_bgr_,
                 bgr_buffer,
                 (::PixelFormat)pix_fmt,
                 output_width_,
                 output_height_);

  // Setup SwsContext for color conversion.
  sws_context_ = sws_getContext(frame_width_,
                                frame_height_,
                                codec_context_->pix_fmt,
                                output_width_,
                                output_height_,
                                (::PixelFormat)pix_fmt,
                                SWS_BICUBIC,
                                nullptr,
                                nullptr,
                                nullptr);
  if(!sws_context_) {
    LOG(ERROR) << "Could not setup SwsContext for color conversion.";
    return false;
  }

  current_pos_ = 0;
  used_as_root_ = set->empty();
  VideoStream* vid_stream = new VideoStream(output_width_,
                                            output_height_,
                                            output_width_step_,
                                            fps_,
                                            options_.pixel_format,
                                            options_.stream_name);

  vid_stream->set_original_width(frame_width_);
  vid_stream->set_original_height(frame_height_);

  set->push_back(shared_ptr<VideoStream>(vid_stream));
  frame_num_ = 0;
  return true;
}

void VideoReaderUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  if (!used_as_root_) {
    input->push_back(shared_ptr<VideoFrame>(ReadNextFrame()));
    output->push_back(input);
    ++frame_num_;
  }
}

bool VideoReaderUnit::PostProcess(list<FrameSetPtr>* append) {
  if (used_as_root_) {
    VideoFrame* next_frame = ReadNextFrame();
    if (next_frame != nullptr) {
      // Make new frameset and push VideoFrame.
      FrameSetPtr frame_set (new FrameSet());
      frame_set->push_back(shared_ptr<VideoFrame>(next_frame));
      append->push_back(frame_set);
      ++frame_num_;
      return true;
    } else {
      return false;
    }
  }
  return false;
}

VideoFrame* VideoReaderUnit::ReadNextFrame() {
  // Read frame by frame.
  AVPacket packet;

  if (options_.trim_frames > 0 && frame_num_ == options_.trim_frames) {
    return nullptr;
  }

  // Test if we already have a read a package (e.g. Seek command)
  if (next_packet_) {
    packet = *next_packet_;
    next_packet_.reset();
  } else {
    // No packages left?
    if (av_read_frame(format_context_, &packet) < 0) {
      // Test if some are queued up in internal buffers.
      packet.data = nullptr;
      packet.size = 0;
      int frame_finished = 0;
      avcodec_decode_video2(codec_context_,
                            frame_yuv_,
                            &frame_finished,
                            &packet);
      if (frame_finished) {
        // More frames to decode.
        VideoFrame* next_frame = ReadNextFrameImpl(packet);
        av_free_packet(&packet);
        return next_frame;
      }

      return nullptr;
    }
  }

  do {
    if (packet.stream_index == video_stream_idx_) {
      int frame_finished;
      int len = avcodec_decode_video2(codec_context_,
                                      frame_yuv_,
                                      &frame_finished,
                                      &packet);
      if (frame_finished) {
        VideoFrame* next_frame = ReadNextFrameImpl(packet);
        av_free_packet(&packet);
        return next_frame;
      }
    }
    av_free_packet(&packet);
  } while (av_read_frame(format_context_, &packet) >= 0); // Single frame could consists
                                                          // of multiple packages.
  LOG(ERROR) << "Unexpected end of ReadNextFrame()";
  return nullptr;
}

VideoFrame* VideoReaderUnit::ReadNextFrameImpl(const AVPacket& packet) {
  // Convert to requested format.
  sws_scale(sws_context_, frame_yuv_->data, frame_yuv_->linesize, 0,
            frame_height_, frame_bgr_->data, frame_bgr_->linesize);

  AVRational target_timebase = {1, 1000000};  // micros.
  int64_t pts =
    av_rescale_q(packet.pts, target_timebase,
                 format_context_->streams[video_stream_idx_]->time_base);
  current_pos_ = pts;
  VideoFrame* curr_frame = new VideoFrame(output_width_,
                                          output_height_,
                                          bytes_per_pixel_,
                                          output_width_step_,
                                          pts);

  const uint8_t* src_data = frame_bgr_->data[0];
  uint8_t* dst_data = curr_frame->mutable_data();

  // Copy to our image.
  for (int i = 0;
       i < output_height_;
       ++i, src_data += frame_bgr_->linesize[0], dst_data += output_width_step_) {
    memcpy(dst_data, src_data, bytes_per_pixel_ * output_width_);
  }

  return curr_frame;
}

bool VideoReaderUnit::SeekImpl(int64_t time) {   // in micros.
  if (time == current_pos_)
    return false;

  // Flush buffers.
  int seek_flag = AVSEEK_FLAG_BACKWARD;
  AVRational source_timebase = {1, 1000000};  // micros.
  int64_t target_pts =
    av_rescale_q(time,
                 format_context_->streams[video_stream_idx_]->time_base,
                 source_timebase);

  LOG(INFO) << "Seeking to " << time << " from curr_pos @ " << current_pos_
            << " (target_pts: " << target_pts << ")";

  av_seek_frame(format_context_, -1, target_pts, seek_flag);
  avcodec_flush_buffers(codec_context_);

  next_packet_.reset(new AVPacket);
  codec_context_->skip_frame = AVDISCARD_NONKEY;
  int runs = 0;
  while (runs++ < 1000) {     // Correct frame should not be more than 1000 frames away.
    if (av_read_frame(format_context_, next_packet_.get()) < 0)
      break;

    int64_t pts = next_packet_->pts;

    if (pts >= target_pts) {
      // Store current pts in our timebase
      current_pos_ = av_rescale_q(pts, source_timebase,
                                  format_context_->streams[video_stream_idx_]->time_base);
      break;
    }

    int frame_finished;
    avcodec_decode_video2(codec_context_, frame_yuv_, &frame_finished,
                          next_packet_.get());
    av_free_packet(next_packet_.get());
  }
  codec_context_->skip_frame =  AVDISCARD_DEFAULT;

  return true;
}

bool VideoReaderUnit::SkipFrames(int frames) {
  int64_t seek_time = current_pos_ + (double)frames / fps_ * 1e6;

  if (seek_time > 0) {
    if (Seek(seek_time))
      return NextFrame();
    else
      return false;
  }
  else {
    return false;
  }
}

bool VideoReaderUnit::PrevFrame() {
  return SkipFrames(-1);
}

}  // namespace video_framework.
