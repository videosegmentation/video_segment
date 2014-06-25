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

#ifndef VIDEO_SEGMENT_VIDEO_FRAMEWORK_FLOW_READER_H__
#define VIDEO_SEGMENT_VIDEO_FRAMEWORK_FLOW_READER_H__

#include "base/base.h"
#include "video_framework/video_unit.h"

namespace cv {
  class Mat;
  class DenseOpticalFlow;
  template<class T> class Ptr;
}

namespace video_framework {

// Stores dense flow as interleaved (x,y) float pair over specified domain.
// Either expresses flow w.r.t. previous (backward_flow = true) or
// next (backward_flow = false) frame.
class DenseFlowFrame : public DataFrame {
public:
  DenseFlowFrame(int width, int height, bool backward_flow, int64_t pts = 0);

  int width() const { return width_; }
  int width_step() const { return width_ * sizeof(float); }
  int height() const { return height_; }
  bool is_backward_flow() const { return backward_flow_; }

  // Returns flow as 2 channel (x,y) float image.
  const cv::Mat MatViewInterleaved() const;
  cv::Mat MatViewInterleaved();

  // Splits flow into x and y matrices of type CV_32F. Use not as efficient as
  // MatViewInterleaved().
  void MatView(cv::Mat* flow_x, cv::Mat* flow_y);


private:
  DenseFlowFrame() : DenseFlowFrame(0, 0, false, 0) {
  }

protected:
  int width_ = 0;
  int height_= 0;
  bool backward_flow_ = true;
};

enum DenseFlowType { FLOW_FORWARD = 0, FLOW_BACKWARD = 1, FLOW_BOTH = 2 };

class DenseFlowReader {
public:
  DenseFlowReader(const std::string& filename)
      : filename_(filename), width_(0), height_(0), flow_type_(FLOW_FORWARD) {}

  bool OpenAndReadHeader();

  int RequiredBufferSize() const { return sizeof(float) * width_ * height_ * 2; }
  // If flow type is FLOW_BOTH, two calls are necessary, first call returns forward,
  // second one backward flow.
  void GetNextFlowFrame(uint8_t* buffer);
  bool MoreFramesAvailable();

  int width() const { return width_; }
  int height() const { return height_; }

  int FlowType() const { return flow_type_; }
private:
  const std::string filename_;

  int width_;
  int height_;
  DenseFlowType flow_type_;

  std::ifstream ifs_;
};

// If both streams are present, outputs first forward then backward flow stream.
struct DenseFlowReaderOptions {
  std::string video_stream_name = "VideoStream";
  std::string backward_flow_stream_name = "BackwardFlowStream";
  std::string forward_flow_stream_name = "ForwardFlowStream";
};

class DenseFlowReaderUnit : public VideoUnit {
public:
  // Based on flow file type either one or both flow streams are added.
  DenseFlowReaderUnit(
      DenseFlowReaderOptions options,
      const std::string& file) : options_(options), flow_file_(file), reader_(file) {
  }

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

protected:
  DenseFlowReaderOptions options_;
  std::string flow_file_;

  int vid_stream_idx_;
  DenseFlowReader reader_;

  int frame_width_;
  int frame_height_;
  DenseFlowType flow_type_;
  int frame_number_;
};

// Specify to computed flow type via flow_type. Flow is written to file (if specified)
// or output as streams (if set).
// Optional video_out_stream_name renders backward (or forward if type=FLOW_FORWARD)
// flow as image, where angle is denoted by hue while saturation and value are
// determined from flow magnitude.

struct DenseFlowOptions {
  DenseFlowType flow_type = FLOW_BACKWARD;
  int flow_iterations = 10;
  int num_warps = 2;
  std::string input_stream_name = "LuminanceStream";
  std::string backward_flow_stream_name = "BackwardFlowStream";
  std::string forward_flow_stream_name = "ForwardFlowStream";
  std::string video_out_stream_name;
  std::string flow_output_file;
};

class DenseFlowUnit : public VideoUnit {
public:
  DenseFlowUnit(const DenseFlowOptions& options);
  ~DenseFlowUnit();

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

private:
  DenseFlowOptions options_;
  int video_stream_idx_;

  int frame_number_;
  int width_step_;
  std::ofstream ofs_;

  std::unique_ptr<cv::Mat> prev_img_;
  std::unique_ptr<cv::Ptr<cv::DenseOpticalFlow>> flow_engine_;
};

}  //namespace video_framework.

#endif  // VIDEO_SEGMENT_VIDEO_FRAMEWORK_FLOW_READER_H__
