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

#include "video_framework/flow_reader.h"

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "base/base_impl.h"

namespace video_framework {

DenseFlowFrame::DenseFlowFrame(int width, int height, bool backward_flow, int64_t pts)
    : DataFrame(&typeid(DenseFlowFrame), 2 * width * height * sizeof(float), pts),
      width_(width),
      height_(height),
      backward_flow_(backward_flow) {
  // Data is already zeroed by DataFrame.
}

void DenseFlowFrame::MatView(cv::Mat* flow_x, cv::Mat* flow_y) {
  CHECK(flow_x != NULL && flow_y != NULL) << "Flow matrices not set.";
  std::vector<cv::Mat> flows{*flow_x, *flow_y};
  cv::split(MatViewInterleaved(), flows);
  // Re-allocation might have occured.
  *flow_x = flows[0];
  *flow_y = flows[1];
}

cv::Mat DenseFlowFrame::MatViewInterleaved() {
  return cv::Mat(height_, width_, CV_32FC2, mutable_data(), width_ * 2 * sizeof(float));
}

const cv::Mat DenseFlowFrame::MatViewInterleaved() const {
  return cv::Mat(height_, width_, CV_32FC2, (void*)data(), width_ * 2 * sizeof(float));
}

bool DenseFlowReader::OpenAndReadHeader() {
  // Open file.
  VLOG(1) << "Reading flow from file " << filename_;
  ifs_.open(filename_.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!ifs_) {
    LOG(ERROR) << "DenseFlowReader::OpenAndReadHeader: "
               << "Can not open binary flow file.\n";
    return false;
  }

  ifs_.read((char*)&width_, sizeof(width_));
  ifs_.read((char*)&height_, sizeof(height_));
  ifs_.read((char*)&flow_type_, sizeof(int));

  return true;
}

bool DenseFlowReader::MoreFramesAvailable() {
  return ifs_.peek() != std::char_traits<char>::eof();
}

void DenseFlowReader::GetNextFlowFrame(uint8_t* buffer) {
  ifs_.read((char*)buffer, RequiredBufferSize());
}

bool DenseFlowReaderUnit::OpenStreams(StreamSet* set) {
  vid_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);
  if (vid_stream_idx_ < 0) {
    LOG(ERROR) << "Can not find video stream.\n";
    return false;
  }

  const VideoStream& vid_stream = set->at(vid_stream_idx_)->As<VideoStream>();

  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();

  reader_.OpenAndReadHeader();

  if (reader_.width() != frame_width_ || reader_.height() != frame_height_) {
    LOG(ERROR) << "Flow file has different dimension than input video.\n";
    return false;
  }

  // Add stream.
  if (reader_.FlowType() == FLOW_FORWARD ||
      reader_.FlowType() == FLOW_BOTH) {
    DataStream* flow_stream = new DataStream(options_.forward_flow_stream_name);
    set->push_back(shared_ptr<DataStream>(flow_stream));
  }

  // Add stream.
  if (reader_.FlowType() == FLOW_BACKWARD ||
      reader_.FlowType() == FLOW_BOTH) {
    DataStream* flow_stream = new DataStream(options_.backward_flow_stream_name);
    set->push_back(shared_ptr<DataStream>(flow_stream));
  }

  frame_number_ = 0;
  return true;
}

void DenseFlowReaderUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  const VideoFrame* frame = input->at(vid_stream_idx_)->AsPtr<VideoFrame>();
  const int64_t pts = frame->pts();

  if (!reader_.MoreFramesAvailable()) {
    LOG(WARNING) << "No more frames available, outputting empty flow frames.";
    input->push_back(shared_ptr<DataFrame>(
          new DenseFlowFrame(0, 0, reader_.FlowType() == FLOW_BACKWARD, pts)));
    if (reader_.FlowType() == FLOW_BOTH) {
      // Forward flow was output above.
      input->push_back(shared_ptr<DataFrame>(new DenseFlowFrame(0, 0, true, pts)));
    }
  } else {
    if (frame_number_) {
      if (reader_.FlowType() == FLOW_FORWARD ||
          reader_.FlowType() == FLOW_BOTH) {
        DenseFlowFrame* flow_frame = new DenseFlowFrame(frame_width_,
                                                        frame_height_,
                                                        false,
                                                        pts);
        reader_.GetNextFlowFrame(flow_frame->mutable_data());
        input->push_back(shared_ptr<DataFrame>(flow_frame));
      }

      if (reader_.FlowType() == FLOW_BACKWARD ||
          reader_.FlowType() == FLOW_BOTH) {
        DenseFlowFrame* flow_frame = new DenseFlowFrame(frame_width_,
                                                        frame_height_,
                                                        true,
                                                        pts);
        reader_.GetNextFlowFrame(flow_frame->mutable_data());
        input->push_back(shared_ptr<DataFrame>(flow_frame));
      }
    } else {
      input->push_back(shared_ptr<DataFrame>(
          new DenseFlowFrame(0, 0, reader_.FlowType() == FLOW_BACKWARD)));
      if (reader_.FlowType() == FLOW_BOTH) {
        // Forward flow was output above.
        input->push_back(shared_ptr<DataFrame>(new DenseFlowFrame(0, 0, true, pts)));
      }
    }
  }

  output->push_back(input);
  ++frame_number_;
}

bool DenseFlowReaderUnit::PostProcess(list<FrameSetPtr>* output) {
  return false;
}

DenseFlowUnit::~DenseFlowUnit() {

}

bool DenseFlowUnit::OpenStreams(StreamSet* set) {
  video_stream_idx_ = FindStreamIdx(options_.input_stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find video stream!\n";
    return false;
  }

  // Prepare flow lib.
  flow_engine_.reset(new cv::Ptr<cv::DenseOpticalFlow>());
  *flow_engine_ = cv::createOptFlow_DualTVL1();
  (*flow_engine_)->set("warps", options_.num_warps);
  (*flow_engine_)->set("iterations", options_.flow_iterations);

  frame_number_ = 0;

  // Get video stream.
  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  int frame_width = vid_stream.frame_width();
  int frame_height = vid_stream.frame_height();

  prev_img_.reset(new cv::Mat(frame_height, frame_width, CV_8U));

  if (vid_stream.pixel_format() != PIXEL_FORMAT_LUMINANCE) {
    LOG(ERROR) << "Expecting luminance input.\n";
    return false;
  }

  width_step_ = frame_width * 3;
  if (width_step_ % 4) {
    width_step_ = width_step_ + (4 - width_step_ % 4);
    CHECK_EQ(0, width_step_ % 4);
  }

  if (options_.flow_type == FLOW_FORWARD || options_.flow_type == FLOW_BOTH) {
    CHECK(!options_.forward_flow_stream_name.empty()) << "Forward flow stream is empty.";
    DataStream* flow_stream = new DataStream(options_.forward_flow_stream_name);
    set->push_back(shared_ptr<DataStream>(flow_stream));
  }

  if (options_.flow_type == FLOW_BACKWARD || options_.flow_type == FLOW_BOTH) {
    CHECK(!options_.backward_flow_stream_name.empty())
      << "Backward flow stream is empty.";
    DataStream* flow_stream = new DataStream(options_.backward_flow_stream_name);
    set->push_back(shared_ptr<DataStream>(flow_stream));
  }

  if (!options_.video_out_stream_name.empty()) {
    VideoStream* video_out_stream =
        new VideoStream(frame_width,
                        frame_height,
                        width_step_,
                        vid_stream.fps(),
                        PIXEL_FORMAT_BGR24,
                        options_.video_out_stream_name);
    set->push_back(shared_ptr<VideoStream>(video_out_stream));
  }

  // Open file.
  if (!options_.flow_output_file.empty()) {
    ofs_.open(options_.flow_output_file.c_str(),
              std::ios_base::out | std::ios_base::binary);

    // Write header.
    int flow_type = options_.flow_type;
    ofs_.write((char*)&frame_width, sizeof(frame_width));
    ofs_.write((char*)&frame_height, sizeof(frame_height));
    ofs_.write((char*)&flow_type, sizeof(flow_type));
  }

  return true;
}

DenseFlowUnit::DenseFlowUnit(const DenseFlowOptions& options) : options_(options) {
}

void DenseFlowUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  // Get raw video frame data.
  const VideoFrame* frame = input->at(video_stream_idx_)->AsPtr<VideoFrame>();

  int width = frame->width();
  int height = frame->height();
  int64_t pts = frame->pts();

  cv::Mat image;
  frame->MatView(&image);

  if (frame_number_ > 0) {
    cv::Mat render_flow;

    shared_ptr<DenseFlowFrame> forward_flow_frame;
    shared_ptr<DenseFlowFrame> backward_flow_frame;

    if (options_.flow_type == FLOW_FORWARD || options_.flow_type == FLOW_BOTH) {
      forward_flow_frame.reset(new DenseFlowFrame(width, height, false, pts));
      cv::Mat flow_field = forward_flow_frame->MatViewInterleaved();
      (*flow_engine_)->calc(*prev_img_, image, flow_field);

      if (!options_.forward_flow_stream_name.empty()) {
        input->push_back(forward_flow_frame);
      }

      if (!options_.flow_output_file.empty()) {
        // Write to file.
        ofs_.write(flow_field.ptr<char>(0), flow_field.step[0] * height);
      }

      render_flow = flow_field;
    }

    if (options_.flow_type == FLOW_BACKWARD || options_.flow_type == FLOW_BOTH) {
      backward_flow_frame.reset(new DenseFlowFrame(width, height, true, pts));
      cv::Mat flow_field = backward_flow_frame->MatViewInterleaved();
      (*flow_engine_)->calc(image, *prev_img_, flow_field);

      if (!options_.backward_flow_stream_name.empty()) {
        input->push_back(backward_flow_frame);
      }

      if (!options_.flow_output_file.empty()) {
        ofs_.write(flow_field.ptr<char>(0), flow_field.step[0] * height);
      }
      render_flow = flow_field;
    }

    if (!options_.video_out_stream_name.empty()) {
      // Convert to HSV color image.
      // TODO: color wheel for flow output.
      VideoFrame* vid_frame = new VideoFrame(width, height, 3, width_step_,
                                             frame->pts());

      cv::Mat rgb_image;
      vid_frame->MatView(&rgb_image);

      cv::Mat hsv_image(height, width, CV_8UC3);

      for (int i = 0; i < height; ++i) {
        const float* flow_ptr = render_flow.ptr<float>(i);

        uint8_t* hsv_ptr = hsv_image.ptr<uint8_t>(i);
        for (int j = 0; j < width; ++j, hsv_ptr +=3, flow_ptr += 2) {
          hsv_ptr[0] = (uint8_t) ((atan2(flow_ptr[1], flow_ptr[0]) / M_PI + 1) * 90);
          hsv_ptr[1] = hsv_ptr[2] = (uint8_t) std::min<float>(
              hypot(flow_ptr[1], flow_ptr[0]) * 20, 255.0);
        }
      }

      cv::cvtColor(hsv_image, rgb_image, CV_HSV2BGR);
      input->push_back(shared_ptr<VideoFrame>(vid_frame));
    }
  } else {
    // No previous frame exists.

    // If forward stream set.
    if (options_.flow_type == FLOW_FORWARD || options_.flow_type == FLOW_BOTH) {
      input->push_back(shared_ptr<DenseFlowFrame>(
          new DenseFlowFrame(width, height, false, pts)));
    }

    // If backward stream set.
    if (options_.flow_type == FLOW_BACKWARD || options_.flow_type == FLOW_BOTH) {
      input->push_back(shared_ptr<DenseFlowFrame>(
          new DenseFlowFrame(width, height, false, pts)));
      LOG(INFO) << "Adding empty backward stream";
    }

    // If video out stream set.
    if (!options_.video_out_stream_name.empty()) {
      shared_ptr<VideoFrame> vid_frame;
      vid_frame.reset(new VideoFrame(width, height, 3, width_step_, pts));
      input->push_back(vid_frame);
      LOG(INFO) << "Adding video stream";
    }
  }

  image.copyTo(*prev_img_);
  output->push_back(input);
  ++frame_number_;

  if (frame_number_ % 10 == 0) {
    LOG(INFO) << "Processed Frame #" << frame_number_ << "\n";
  }
}

bool DenseFlowUnit::PostProcess(list<FrameSetPtr>* append) {
  // Close file.
  if (!options_.flow_output_file.empty()) {
    ofs_.close();
  }
  return false;
}

}  // namespace video_framework.
