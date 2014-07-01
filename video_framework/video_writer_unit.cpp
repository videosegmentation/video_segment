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

#include "video_framework/video_writer_unit.h"
#include "base/base_impl.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavutil/mathematics.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#ifdef __cplusplus
}
#endif

#ifndef CODEC_TYPE_VIDEO // Gone since libavcodec53
#define CODEC_TYPE_VIDEO AVMEDIA_TYPE_VIDEO
#endif

namespace video_framework {

bool VideoWriterUnit::ffmpeg_initialized_ = false;

VideoWriterUnit::VideoWriterUnit(const VideoWriterOptions& options,
                                 const std::string& video_file)
    : options_(options), video_file_(video_file) {
}

bool VideoWriterUnit::OpenStreams(StreamSet* set) {
  // Setup FFMPEG.
  if (!ffmpeg_initialized_) {
    ffmpeg_initialized_ = true;
    av_register_all();
  }

  // Find video stream index.
  video_stream_idx_ = FindStreamIdx(options_.stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find Video stream!\n";
    return false;
  }

  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();
  if (!options_.fps) {
    options_.fps = vid_stream.fps();
  }

  if (!options_.output_format.empty()) {
    output_format_ = av_guess_format(options_.output_format.c_str(), NULL, NULL);
  } else {
    output_format_ = av_guess_format(NULL, video_file_.c_str(), NULL);
  }

  output_width_ = frame_width_;
  output_height_ = frame_height_;

  if (options_.scale != 1) {
    if (options_.scale_max_dim || options_.scale_min_dim) {
      LOG(WARNING) << "Scale set, ignoring scale_[max|min]_dim.";
    }
    output_width_ *= options_.scale;
    output_height_ *= options_.scale;
  } else {
    if (options_.scale_max_dim) {
      float max_dim = std::max(frame_width_, frame_height_);
      output_width_ = (float)frame_width_ / max_dim * options_.scale_max_dim;
      output_height_ = (float)frame_height_ / max_dim * options_.scale_max_dim;
    } else if (options_.scale_min_dim) {
      float min_dim = std::min(frame_width_, frame_height_);
      output_width_ = (float)frame_width_ / min_dim * options_.scale_min_dim;
      output_height_ = (float)frame_height_ / min_dim * options_.scale_min_dim;
    }
  }

  int w_reminder = output_width_ % options_.fraction;
  if (w_reminder > 0) {
    if (w_reminder < options_.fraction / 2) {
      output_width_ -= w_reminder;
    } else {
      output_width_ += (options_.fraction - w_reminder);
    }
  }

  int h_reminder = output_height_ % options_.fraction;
  if (h_reminder > 0) {
    if (h_reminder < options_.fraction / 2) {
      output_height_ -= h_reminder;
    } else {
      output_height_ += (options_.fraction - h_reminder);
    }
  }

  avformat_alloc_output_context2(&format_context_, output_format_, NULL,
                                 video_file_.c_str());
  if(!format_context_) {
    LOG(ERROR) << "Could not open format context.\n";
    return false;
  }

  // Add video stream.
  video_stream_ = avformat_new_stream(format_context_, NULL);
  if (!video_stream_) {
    LOG(ERROR) << "Could not allocate video stream.\n";
    return false;
  }

  // Set standard parameters.
  codec_context_ = video_stream_->codec;
  const string file_ending = video_file_.substr(video_file_.size() - 3);
  if (file_ending == "mp4" || file_ending == "mov") {
    codec_context_->codec_id = CODEC_ID_H264;
  } else {
    codec_context_->codec_id = output_format_->video_codec;
  }

  codec_context_->codec_type = CODEC_TYPE_VIDEO;
  codec_context_->bit_rate = options_.bit_rate;
  codec_context_->bit_rate_tolerance = options_.bit_rate / 5;
  codec_context_->width = output_width_;
  codec_context_->height = output_height_;

  LOG(INFO) << "Encoding with " << options_.fps << " fps.";
  codec_context_->time_base = av_d2q(1.0 / options_.fps, 1000);
  
  LOG(INFO) << "time base : " << codec_context_->time_base.num
            << " / " << codec_context_->time_base.den;

  codec_context_->pix_fmt = PIX_FMT_YUV420P;

  if (codec_context_->codec_id == CODEC_ID_MPEG2VIDEO) {
    codec_context_->max_b_frames = 2;
  }

  if (codec_context_->codec_id == CODEC_ID_MPEG1VIDEO) {
    codec_context_->mb_decision = 2;
  }

  if (codec_context_->codec_id == CODEC_ID_H264) {
    // H264 settings.
    codec_context_->coder_type = FF_CODER_TYPE_AC;
    codec_context_->flags |= CODEC_FLAG_LOOP_FILTER | CODEC_FLAG_GLOBAL_HEADER;
    codec_context_->profile=FF_PROFILE_H264_BASELINE;
    codec_context_->scenechange_threshold = 40;
    codec_context_->gop_size = 10;
    codec_context_->max_b_frames = 0;
    codec_context_->max_qdiff = 4;
    codec_context_->me_method = ME_HEX;
    codec_context_->me_range = 16;
    codec_context_->me_subpel_quality = 6;
    codec_context_->qmin = 10;
    codec_context_->qmax = 51;
    codec_context_->qcompress = 0.6;
    codec_context_->keyint_min = 10;
    codec_context_->trellis = 0;
    codec_context_->level = 13;
    codec_context_->refs = 1;
  }

  // Find and open codec.
  codec_ = avcodec_find_encoder(codec_context_->codec_id);
  if (!codec_) {
    LOG(ERROR) << "Codec not found.";
    return false;
  }

  if (avcodec_open2(codec_context_, codec_, NULL) < 0) {
    LOG(ERROR) << "Could not open codec.";
    return false;
  }

  frame_encode_ = av_frame_alloc();
  frame_bgr_ = av_frame_alloc();

  if (!frame_bgr_ || !frame_encode_) {
    LOG(ERROR) << "Could not alloc tmp. images.\n";
    return false;
  }

  uint8_t* encode_buffer =
      (uint8_t*)av_malloc(avpicture_get_size(codec_context_->pix_fmt,
                                             codec_context_->width,
                                             codec_context_->height));

  avpicture_fill((AVPicture*)frame_encode_, encode_buffer, codec_context_->pix_fmt,
                 codec_context_->width, codec_context_->height);

  uint8_t* bgr_buffer = (uint8_t*)av_malloc(avpicture_get_size(PIX_FMT_BGR24,
                                                               frame_width_,
                                                               frame_height_));
  avpicture_fill((AVPicture*)frame_bgr_,
                 bgr_buffer,
                 PIX_FMT_BGR24,
                 frame_width_,
                 frame_height_);

  // Open output file, if needed.
  if(!(output_format_->flags & AVFMT_NOFILE)) {
    if (avio_open(&format_context_->pb, video_file_.c_str(), AVIO_FLAG_WRITE) < 0) {
      LOG(ERROR) << " Could not open" << video_file_;
      return false;
    }
  }

  avformat_write_header(format_context_,0);

  // Setup color conversion.
  sws_context_ = sws_getContext(frame_width_,
                                frame_height_,
                                PIX_FMT_BGR24,
                                codec_context_->width,
                                codec_context_->height,
                                codec_context_->pix_fmt,
                                SWS_BICUBIC,
                                NULL,
                                NULL,
                                NULL);

  if (!sws_context_) {
    LOG(ERROR) << "Could initialize sws_context.";
    return false;
  }

  frame_num_ = 0;
  return true;
}

void VideoWriterUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  // Write single frame.
  const VideoFrame* frame = input->at(video_stream_idx_)->AsPtr<VideoFrame>();

  // Copy video_frame to frame_bgr_.
  const uint8_t* src_data = frame->data();
  uint8_t* dst_data = frame_bgr_->data[0];

  for (int i = 0;
       i < frame_height_;
       ++i, src_data += frame->width_step(), dst_data += frame_bgr_->linesize[0]) {
   memset(dst_data, 0, LineSize());
   memcpy(dst_data, src_data, 3 * frame_width_);
  }

  // Convert bgr picture to codec.
  sws_scale(sws_context_, frame_bgr_->data, frame_bgr_->linesize, 0, frame_height_,
            frame_encode_->data, frame_encode_->linesize);
  int got_frame;
  EncodeFrame(frame_encode_, &got_frame);

  ++frame_num_;
  output->push_back(input);
}

int VideoWriterUnit::EncodeFrame(AVFrame* frame, int* got_frame) {
   // Encode.
  int ret_val;
  AVPacket packet;
  packet.data = nullptr;
  packet.size = 0;
  av_init_packet(&packet);

  ret_val = avcodec_encode_video2(codec_context_, &packet, frame, got_frame);
  if (ret_val < 0) {
    LOG(ERROR) << "Error encoding frame.";
    return ret_val;
  }
  if (*got_frame == 0) {
    return 0;
  }

  // TODO: CHECK!
  if (codec_context_->coded_frame->pts != AV_NOPTS_VALUE) {
    packet.pts = av_rescale_q(codec_context_->coded_frame->pts,
                              codec_context_->time_base,
                              video_stream_->time_base);
  }

  packet.stream_index = video_stream_->index;
  ret_val = av_interleaved_write_frame(format_context_, &packet);

  if (ret_val != 0) {
    LOG(ERROR) << "Error while writing frame.";
  }
  return ret_val;
}

bool VideoWriterUnit::PostProcess(list<FrameSetPtr>* append) {
  if (format_context_->streams[video_stream_->index]->codec->codec->capabilities &
      CODEC_CAP_DELAY) {
    while (true) {
      int got_frame;
      if (EncodeFrame(nullptr, &got_frame) < 0) {
        break;
      }
      if (!got_frame) {
        break;
      }
    }
  }

  // Close file.
  av_write_trailer(format_context_);

  if(!output_format_->flags & AVFMT_NOFILE) {
    avio_close(format_context_->pb);
  }

  // Free resources.
  avcodec_close(codec_context_);
  av_free(frame_encode_->data[0]);
  av_free(frame_encode_);

  av_free(frame_bgr_->data[0]);
  av_free(frame_bgr_);

  for (uint i = 0; i < format_context_->nb_streams; ++i) {
    av_freep(&format_context_->streams[i]->codec);
    av_freep(&format_context_->streams);
  }

  av_free(format_context_);

  return false;
}

int VideoWriterUnit::LineSize() const {
  return frame_bgr_->linesize[0];
}

}  // namespace video_framework.
