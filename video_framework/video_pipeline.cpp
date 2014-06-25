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

#include "video_framework/video_pipeline.h"

#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base/base_impl.h"

namespace video_framework {

VideoPipelineSink::VideoPipelineSink() {
  source_exhausted_ = false;
  frame_number_ = 0;
}

void VideoPipelineSink::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  frameset_queue_.push(input);
  ++frame_number_;
}

bool VideoPipelineSink::TryFetchingFrameSet(FrameSetPtr* ptr) {
  FrameSetPtr local_ptr;
  const bool success = frameset_queue_.try_pop(&local_ptr);
  if (success) {
    ptr->swap(local_ptr);
    return true;
  } else {
    return false;
  }
}

VideoPipelineSource::VideoPipelineSource(VideoPipelineSink* sink,
                                         VideoUnit* idle_unit,
                                         const SourceRatePolicy& policy,
                                         float max_fps)
    : sink_(sink),
      idle_unit_(idle_unit),
      source_rate_policy_(policy),
      max_fps_(max_fps) {
  AttachTo(sink);
  frame_num_ = 0;
}

bool VideoPipelineSource::OpenStreams(StreamSet* set) {
  if (idle_unit_) {
    return idle_unit_->PrepareProcessing();
  }

  return true;
}

bool VideoPipelineSource::Run() {
  while (!(sink_->IsExhausted() && sink_->GetQueueSize() == 0)) {
    FrameSetPtr frame_set_ptr;
    float timeout = 200;  // 0.2 ms

    if (max_fps_ > 0) {
      timeout = 1.0f / max_fps_ * 1e6;  // in micros.
    }

    if (sink_->TryFetchingFrameSet(&frame_set_ptr)) {
      // Measure time difference.
      boost::posix_time::ptime curr_time =
          boost::posix_time::microsec_clock::local_time();

      if (frame_num_ > 0) {
        float micros_passed = boost::posix_time::time_period(
            prev_process_time_, curr_time).length().total_microseconds();
        int wait_time = timeout - micros_passed;
        while (wait_time > 10) {
          OnIdle();

          // Update wait time.
          curr_time = boost::posix_time::microsec_clock::local_time();
          micros_passed = boost::posix_time::time_period(
              prev_process_time_,
              curr_time).length().total_microseconds();
          wait_time = timeout - micros_passed;
          // We only sleep a fraction of the wait time to keep OnIdle going.
          if (wait_time > 100) {
            boost::thread::sleep(boost::get_system_time() +
                                 boost::posix_time::microseconds(wait_time / 5));
          }
        }
      }

      // Update processing time.
      prev_process_time_ = boost::posix_time::microsec_clock::local_time();

      // Pass to children.
      for (auto child : children_) {
        child->ProcessFrameImpl(frame_set_ptr, this);
      }

      ++frame_num_;
    } else {
      OnIdle();
      boost::thread::sleep(boost::get_system_time() +
                           boost::posix_time::microseconds(timeout / 5));
    }
  }

  PostProcessImpl(this);
  return true;
}

void VideoPipelineSource::OnIdle() {
  if (idle_unit_) {
    // Call idle_unit PostProcess.
    idle_unit_->ProcessFrameImpl(FrameSetPtr(new FrameSet()), this);
  }
}

void VideoPipelineSource::LimitRateImpl(float fps) {
  if (source_rate_policy_.respond_to_limit_rate) {
    max_fps_ = fps * source_rate_policy_.rate_scale;

    if (monitor_sink_ != nullptr &&
        source_rate_policy_.sink_max_queue_size > 0 &&
        monitor_sink_->GetQueueSize() > source_rate_policy_.sink_max_queue_size) {
      // Stall, but still process.
      max_fps_ = fps * 0.1;
    }
  }
}

VideoPipelineInvoker::VideoPipelineInvoker() : threads_(new boost::thread_group()) {
}

VideoPipelineInvoker::~VideoPipelineInvoker() {
}

void VideoPipelineInvoker::RunRoot(VideoUnit* root) {
  thread_ptrs_.push_back(
    threads_->create_thread(boost::bind(&VideoUnit::Run, root)));
}

void VideoPipelineInvoker::RunRootRateLimited(const RatePolicy& rate_policy,
                                              VideoUnit* root) {
  thread_ptrs_.push_back(
      threads_->create_thread(boost::bind(&VideoUnit::RunRateLimited, root,
                                          rate_policy)));
}

void VideoPipelineInvoker::RunPipelineSource(VideoPipelineSource* source) {
  thread_ptrs_.push_back(
      threads_->create_thread(boost::bind(&VideoPipelineSource::Run, source)));
}

void VideoPipelineInvoker::WaitUntilPipelineFinished() {
  threads_->join_all();
}

bool VideoPipelineStats::OpenStreams(StreamSet* set) {
  frame_width_step_ = options_.frame_width * 3;
  // Ensure can be encoded as video.
  if (frame_width_step_ % 4) {
    frame_width_step_ += (4 - frame_width_step_ % 4);
  }

  VideoStream* vid_stream = new VideoStream(options_.frame_width,
                                            options_.frame_height,
                                            frame_width_step_,
                                            0,   // fps.
                                            PIXEL_FORMAT_BGR24,
                                            options_.video_stream_name);
  set->push_back(shared_ptr<VideoStream>(vid_stream));
  start_time_ = boost::posix_time::microsec_clock::local_time();
  return true;
}

void VideoPipelineStats::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  const int bins = sinks_.size();
  const int border = 40;
  const float scale = (options_.frame_height - border) /
    (options_.max_queue_height * 1.3);

  // Create a new frameset and video-frame.
  shared_ptr<VideoFrame> curr_frame(new VideoFrame(options_.frame_width,
                                                   options_.frame_height,
                                                   3,
                                                   frame_width_step_));

  cv::Mat image;
  curr_frame->MatView(&image);
  image.setTo(180);

  // Draw histogram.
  const int spacing = image.cols / (bins + 2);
  for (int i = 0; i < bins; ++i) {
    // Locations.
    int val = sinks_[i].first->GetQueueSize();
    int col = (i + 1) * spacing;
    int row = options_.frame_height - border - (val * scale);
    cv::Point pt1(col - spacing / 3, row);
    cv::Point pt2(col + spacing / 3, image.rows - border);
    // Bar plot.
    cv::rectangle(image, pt1, pt2, CV_RGB(0, 0, 200), -2);   // Neg. last arg for filled.

    // Value text.
    cv::Point txt_pt(col - spacing / 3, options_.frame_height - border / 3 - 10);
    cv::putText(image,
                base::StringPrintf("%02d", val),
                txt_pt,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                CV_RGB(0, 0, 0),
                2);

    // Name text.
    txt_pt.y += 15;
    cv::putText(image,
                sinks_[i].second,
                txt_pt,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                CV_RGB(0, 0, 0),
                1);

    // Fps text.
    txt_pt.y = border / 3;
    cv::putText(image,
                base::StringPrintf("%3.1f", sinks_[i].first->MinTreeRate()),
                txt_pt,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                CV_RGB(0, 0, 0),
                1);

  }

  // Print running time.
  boost::posix_time::ptime curr_time = boost::posix_time::microsec_clock::local_time();
  float secs_passed = boost::posix_time::time_period(
      start_time_, curr_time).length().total_milliseconds() * 1.e-3f;
  cv::putText(image,
              base::StringPrintf("up: %5.1f", secs_passed),
              cv::Point(options_.frame_width - 75, border / 3),
              cv::FONT_HERSHEY_SIMPLEX,
              0.4,
              CV_RGB(0, 0, 0),
              1);

  // Pass to output.
  input->push_back(curr_frame);
  output->push_back(input);
}

}  // namespace video_framework.
