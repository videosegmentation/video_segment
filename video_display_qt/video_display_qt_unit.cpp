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


#include "video_display_qt_unit.h"

#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>

#include <QtGui/QApplication>
#include <QtGui/QMainWindow>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "base/base_impl.h"
#include "main_window.h"
#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_render.h"
#include "segment_util/segmentation_util.h"

using namespace segmentation;

static constexpr int kMinFrameTime = 20;   // Min milliseconds between frame updates.
static constexpr int kMaxWaitTime = 10;

namespace video_framework {

// Global var to be used across classes to check for only one qt application running.
static QApplication* g_main_app = nullptr;
static char* g_main_argv[1] = { "dummy_file_name" };
static int g_main_argc = 1;

// Summed border width to be use across classes to ensure windows don't overlap.
static int g_summed_window_width = 50;  // Initial 50 pixel border.

VideoDisplayQtUnit::VideoDisplayQtUnit(const VideoDisplayQtOptions& options)
  : options_(options) {
  if (g_main_app == nullptr) {
    g_main_app = new QApplication(g_main_argc, g_main_argv);
  }

  main_window_.reset(new MainWindow(options_.stream_name, false));  // No slider.
}

VideoDisplayQtUnit::~VideoDisplayQtUnit() {
  if (g_main_app != nullptr) {
    g_main_app->exit();
    g_main_app = nullptr;
  }
}

bool VideoDisplayQtUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  video_stream_idx_ = FindStreamIdx(options_.stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find Video stream!\n";
    return false;
  }

  // Get video stream.
  const VideoStream* vid_stream =
    dynamic_cast<const VideoStream*>(set->at(video_stream_idx_).get());
  CHECK_NOTNULL(vid_stream);

  // Window setup.
  frame_width_ = vid_stream->frame_width();
  frame_height_ = vid_stream->frame_height();

  if (options_.upscale != 1.0f) {
    frame_width_ *= options_.upscale;
    frame_height_ *= options_.upscale;
    scaled_image_.reset(new cv::Mat(frame_height_, frame_width_, CV_8UC3));
  }

  // Qt Display.
  main_window_->SetSize(frame_width_, frame_height_);
  main_window_->move(g_summed_window_width, 0);
  main_window_->show();

  // Gap between windows.
  g_summed_window_width += frame_width_ + 50;

  // Fps control.
  last_update_time_ = boost::posix_time::microsec_clock::local_time();

  // Force refresh now.
  QApplication::processEvents();
  return true;
}

void VideoDisplayQtUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  // Fps control -> early return if updated too frequently.
  ptime curr_time = boost::posix_time::microsec_clock::local_time();
  float msecs_passed = boost::posix_time::time_period(
                         last_update_time_, curr_time).length().total_milliseconds();

  output->push_back(input);
  if (msecs_passed < kMinFrameTime) {
    // Ok if we wait a bit?
    if (msecs_passed + kMaxWaitTime < kMinFrameTime) {
        boost::thread::sleep(boost::get_system_time() +
                             boost::posix_time::milliseconds(
                               kMinFrameTime - msecs_passed));
    } else {
      return;
    }
  }

  const VideoFrame* frame =
    dynamic_cast<const VideoFrame*>(input->at(video_stream_idx_).get());
  CHECK_NOTNULL(frame);

  // Draw.
  std::unique_ptr<QImage> image;
  if (options_.upscale != 1.0f) {
    // Resize.
    cv::Mat mat_view;
    frame->MatView(&mat_view);
    cv::resize(mat_view, *scaled_image_, scaled_image_->size());

    // Make Qt image.
    image.reset(new QImage((const uint8_t*)scaled_image_->data,
                           scaled_image_->cols,
                           scaled_image_->rows,
                           scaled_image_->step[0],
                           QImage::Format_RGB888));
  } else {
    // Make Qt image.
    image.reset(new QImage((const uint8_t*)frame->data(),
                           frame->width(),
                           frame->height(),
                           frame->width_step(),
                           QImage::Format_RGB888));
  }

  // Show Qt image
  main_window_->DrawImage(image->rgbSwapped());

  // Force refresh
  QApplication::processEvents();
  // fps control
  last_update_time_ = boost::posix_time::microsec_clock::local_time();
}


bool VideoDisplayQtUnit::PostProcess(std::list<FrameSetPtr>* append) {
  return false;
}

SegmentationDisplayUnit::SegmentationDisplayUnit(
    const SegmentationDisplayOptions& options) {
  if (g_main_app == nullptr) {
    g_main_app = new QApplication(g_main_argc, g_main_argv);
  }
  main_window_.reset(new MainWindow(options_.segment_stream_name, true));  // with slider.
}


SegmentationDisplayUnit::~SegmentationDisplayUnit() {
  if (g_main_app != nullptr) {
    g_main_app->exit();
  }
}

bool SegmentationDisplayUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  if (options_.video_stream_name.empty()) {
    if (options_.blend_alpha != 1.f) {
      options_.blend_alpha = 1.f;
      LOG(WARNING) << "No video stream requested. Setting blend alpha to 1.";
    }

    vid_stream_idx_ = -1;
  } else {
    vid_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

    if (vid_stream_idx_ < 0) {
      LOG(ERROR) << "Could not find Video stream!\n";
      return false;
    }

    // Get video stream info.
    const VideoStream* vid_stream =
      dynamic_cast<const VideoStream*>(set->at(vid_stream_idx_).get());
    CHECK_NOTNULL(vid_stream);

    frame_width_ = vid_stream->frame_width();
    frame_height_ = vid_stream->frame_height();
  }

  // Get segmentation stream.
  seg_stream_idx_ = FindStreamIdx(options_.segment_stream_name, set);

  if (seg_stream_idx_ < 0) {
    LOG(ERROR) << "SegmentationDisplayUnit::OpenStreams: "
               << "Could not find Segmentation stream!\n";
    return false;
  }

  const SegmentationStream* seg_stream =
    dynamic_cast<const SegmentationStream*>(set->at(seg_stream_idx_).get());
  if (frame_width_ == 0) {
    // Read dimensions from segmentation stream.
    frame_width_ = seg_stream->frame_width();
    frame_height_ = seg_stream->frame_height();
  } else {
    CHECK_EQ(frame_width_, seg_stream->frame_width());
    CHECK_EQ(frame_height_, seg_stream->frame_height());
  }

  // Allocate render buffer.
  render_frame_.reset(new cv::Mat(frame_height_, frame_width_, CV_8UC3));
  const int actual_height = frame_height_ * (options_.concat_with_source ? 2 : 1);
  const int actual_width = frame_width_;

  // Qt Display.
  main_window_->SetSize(actual_width * options_.upscale,
                        actual_height * options_.upscale);
  main_window_->move(g_summed_window_width, 0);
  main_window_->show();

  output_frame_.reset(new cv::Mat(actual_height, actual_width, CV_8UC3));
  if (options_.upscale != 1.0f) {
    scaled_image_.reset(new cv::Mat(actual_height * options_.upscale,
                                    actual_width * options_.upscale,
                                    CV_8UC3));
  }

  // Gap between windows.
  g_summed_window_width += frame_width_ * options_.upscale + 50;

  // FPS control.
  last_update_time_ = boost::posix_time::microsec_clock::local_time();

  fractional_level_ = options_.hierarchy_level;
  CHECK_GE(fractional_level_, 0.0f);
  CHECK_LE(fractional_level_, 1.0f);
  main_window_->SetLevel(fractional_level_);

  // Force refresh now.
  QApplication::processEvents();
  return true;
}

void SegmentationDisplayUnit::ProcessFrame(FrameSetPtr input,
                                           list<FrameSetPtr>* output) {
  // Fps control, early return if updated too frequently.
  ptime curr_time = boost::posix_time::microsec_clock::local_time();
  float msecs_passed = boost::posix_time::time_period(
                           last_update_time_, curr_time).length().total_milliseconds();
  output->push_back(input);


  const PointerFrame<SegmentationDesc>& seg_frame =
    input->at(seg_stream_idx_)->As<PointerFrame<SegmentationDesc>>();

  const SegmentationDesc& desc = seg_frame.Ref();

  // Update hierarchy if present.
  if (desc.hierarchy_size() > 0) {
    seg_hier_.reset(new SegmentationDesc(desc));
  }

  if (msecs_passed < kMinFrameTime) {
    // Ok if we wait a bit?
    if (msecs_passed + kMaxWaitTime < kMinFrameTime) {
        boost::thread::sleep(boost::get_system_time() +
                             boost::posix_time::milliseconds(
                               kMinFrameTime - msecs_passed));
    } else {
      return;
    }
  }
 
  // Fetch video frame.
  cv::Mat frame_view;
  if (vid_stream_idx_ >= 0) {
    const VideoFrame* frame =
        dynamic_cast<const VideoFrame*>(input->at(vid_stream_idx_).get());
    CHECK_NOTNULL(frame);
    frame->MatView(&frame_view);
  }

  // Parse slider (percentage)
  // Only update if slider was changed.
  const float curr_level = main_window_->GetLevel();
  if (curr_level != fractional_level_) {
    fractional_level_ = curr_level;

    // Fractional specification for hierarchy-level.
    absolute_level_ =
      (int)(fractional_level_ * (seg_hier_->hierarchy_size() - 1.f));
  }

  // Render segmentation.
  RenderRegionsRandomColor(absolute_level_,
                           options_.highlight_edges,
                           options_.draw_shape_descriptors,
                           desc,
                           &seg_hier_->hierarchy(),
                           render_frame_.get());

  if (options_.concat_with_source) {
    CHECK_GE(vid_stream_idx_, 0)
        << "Request concatenation with source but no video stream present.";
    cv::Mat top_half = output_frame_->rowRange(0, frame_height_);
    render_frame_->copyTo(top_half);
    cv::Mat bottom_half = output_frame_->rowRange(frame_height_, 2 * frame_height_);
    frame_view.copyTo(bottom_half);
  } else {
    if (vid_stream_idx_ >= 0) {
      // Blend with input.
      cv::addWeighted(frame_view,
                      1.0 - options_.blend_alpha,
                      *render_frame_,
                      options_.blend_alpha,
                      0,              // Offset.
                      *output_frame_);
    } else {
      *output_frame_ = *render_frame_;
    }
  }

  // Draw.
  std::unique_ptr<QImage> image;
  if (options_.upscale != 1.0f) {
    // Resize
    cv::resize(*output_frame_, *scaled_image_, scaled_image_->size());
    image.reset(new QImage((const uint8_t*)scaled_image_->data,
                           scaled_image_->cols,
                           scaled_image_->rows,
                           scaled_image_->step[0],
                           QImage::Format_RGB888));
  } else {
    image.reset(new QImage((const uint8_t*)output_frame_->data,
                           output_frame_->cols,
                           output_frame_->rows,
                           output_frame_->step[0],
                           QImage::Format_RGB888));
  }

  // Show Qt image
  main_window_->DrawImage(image->rgbSwapped());

  // Force refresh
  QApplication::processEvents();
  // fps control
  last_update_time_ = boost::posix_time::microsec_clock::local_time();
}


bool SegmentationDisplayUnit::PostProcess(std::list<FrameSetPtr>* append) {
  return false;
}

int SegmentationDisplayUnit::GetHierarchyLevel() {
  float level = -1;
  if (seg_hier_) {
    // Parse slider (percentage)
    level = main_window_->GetLevel();
    // Fractional specification for hierarchy-level
    level = (int)(level * (seg_hier_->hierarchy_size() - 1.f));
  }
  return level;
}


std::pair<int, int> SegmentationDisplayUnit::GetMouseLoc(){
  return
    std::make_pair<int, int>(main_window_->GetMouseLoc().first / options_.upscale,
                             main_window_->GetMouseLoc().second / options_.upscale);
}

bool SegmentationDisplayUnit::SpaceKeyPressed() {
  return main_window_->SpaceKeyPressed();
}

} // namespace videoframework.

