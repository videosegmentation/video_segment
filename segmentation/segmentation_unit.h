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

#ifndef SEGMENTATION_UNIT_H__
#define SEGMENTATION_UNIT_H__

#include "base/base.h"
#include "segment_util/segmentation_io.h"
#include "segmentation/dense_seg_graph_interface.h"
#include "segmentation/dense_segmentation.h"
#include "segmentation/region_segmentation.h"
#include "segmentation/segmentation.h"
#include "video_framework/video_unit.h"

namespace cv {
  class Mat;
}

namespace segmentation {

using namespace video_framework;

class DenseSegmentation;
class RegionSegmentation;
class SegmentationWriter;

struct DenseSegmentationUnitOptions {
  std::string video_stream_name = "VideoStream";

  std::string flow_stream_name = "BackwardFlowStream";

  // Name of output segmentation stream.
  std::string segment_stream_name = "SegmentationStream";
};

// Derive for redefining pixel descriptors and distances.
// Create you own options 
class DenseSegmentationUnit : public VideoUnit  {
 public:
  // Creates new unit with optional dense_seg_options;
  DenseSegmentationUnit(const DenseSegmentationUnitOptions& options,
                        const DenseSegmentationOptions* dense_seg_options);

  virtual ~DenseSegmentationUnit() = default;

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

 protected:
  // Re-define the following functions:

  // Override to load feature specific streams. Called during OpenStreams.
  virtual bool OpenFeatureStreams(StreamSet* set);

  // Method to extract features from a FrameSet. Features are passed to
  // DenseSegmentation::Process.
  virtual void ExtractFrameSetFeatures(FrameSetPtr input, std::vector<cv::Mat>* features);

  // Returns DenseSegmentation initialized with appropiate options. By default
  // uses DenseSegmentation base class. Called during OpenStreams after
  // OpenFeatureStreams.
  virtual std::unique_ptr<DenseSegmentation> CreateDenseSegmentation();

 protected:
  // Accessors for read-only content.
  int video_stream_idx() const { return video_stream_idx_; }

  // Returns -1 if flow is not present.
  int flow_stream_idx() const { return flow_stream_idx_; }

  int frame_width() const { return frame_width_; }
  int frame_height() const { return frame_height_; }

  const DenseSegmentationUnitOptions& options() const { return options_; }

 private:
  void OutputSegmentation(std::vector<std::unique_ptr<SegmentationDesc>>* results,
                          std::list<FrameSetPtr>* output);

  int video_stream_idx_ = -1;
  int flow_stream_idx_ = -1;

  DenseSegmentationUnitOptions options_;
  DenseSegmentationOptions dense_seg_options_;

  int frame_width_ = 0;
  int frame_height_ = 0;

  int input_frames_ = 0;
  // Used to signal process to the user.
  int output_frames_ = 0;
  int processed_chunks_ = 0;

  std::unique_ptr<DenseSegmentation> dense_seg_;

  // Buffer until segmentation outputs results.
  std::list<FrameSetPtr> frame_set_buffer_;
};

struct RegionSegmentationUnitOptions {
  std::string video_stream_name = "VideoStream";

  std::string flow_stream_name = "BackwardFlowStream";

  // Name of input/output segmentation stream.
  std::string segment_stream_name = "SegmentationStream";

  bool free_video_frames = false;
  bool free_flow_frames = true;
};

class RegionSegmentationUnit : public VideoUnit {
 public:
  // Creates new region segmentation unit with optional region options.
  RegionSegmentationUnit(const RegionSegmentationUnitOptions& options,
                         const RegionSegmentationOptions* region_options);
  ~RegionSegmentationUnit() = default;

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);
 protected:
  // Overload the functions below to supply your own region features and distances.
  // Override to load feature specific streams, called at the end of OpenStreams.

  // Called at the end of OpenStreams, but before CreateRegionSegmentation is called.
  // Overload to customize further feature extraction.
  virtual bool OpenFeatureStreams(StreamSet* set);

  // Returns RegionSegmentation initialized with appropiate options. By default
  // uses RegionSegmentation base class. Called during OpenStreams after
  // OpenFeatureStreams.
  virtual std::unique_ptr<RegionSegmentation> CreateRegionSegmentation();

  // Method to extract features from a FrameSet. Features are passed to
  // RegionSegmentation::Process.
  virtual void ExtractFrameSetFeatures(FrameSetPtr input,
                                       std::vector<cv::Mat>* features);

 protected:
  // Accessors for read-only content.
  int video_stream_idx() const { return video_stream_idx_; }

  // Returns -1 if flow is not present.
  int flow_stream_idx() const { return flow_stream_idx_; }

  int frame_width() const { return frame_width_; }
  int frame_height() const { return frame_height_; }

  const RegionSegmentationUnitOptions& options() const { return options_; }

 private:
  void OutputSegmentation(
    std::vector<std::unique_ptr<SegmentationDesc>>* results,
    std::list<FrameSetPtr>* output);

 private:
  int video_stream_idx_ = -1;
  int flow_stream_idx_ = -1;
  int seg_stream_idx_ = -1;

  RegionSegmentationUnitOptions options_;
  RegionSegmentationOptions region_options_;

  std::unique_ptr<RegionSegmentation> region_seg_;

  int frame_width_ = 0;
  int frame_height_ = 0;
  int num_input_frames_ = 0;

  // Buffer FrameSet's for one chunk set.
  std::list<FrameSetPtr> frame_set_buffer_;
};

struct SegmentationWriterUnitOptions {
  // Optional. No frames are actually read. However downscale information is extracted
  // from the video stream.
  std::string video_stream_name = "VideoStream";   // Optional.
  std::string segment_stream_name = "SegmentationStream";

  // Output filename result is written to.
  std::string filename;

  // If set, segmentation result is prepared for annotator (storing reduced format).
  bool strip_to_essentials = false;

  // Chunks in which results are written out.
  int chunk_size = 20;

  // If segmentation was computed on a downscaled frame, upscales segmentation
  // to original resolution. Requires vectorization.
  bool upscale_segmentation_if_necessary = true;

  // If set, only vectorization is saved.
  bool remove_rasterization = false;
};

// Streaming writer, frames are written in chunks of chunk_size.
// If strip_essentials is set, we don't write protobuffer for each frame but
// specialized binary format for annotiation flash UI while stripping some non-important
// informatation (descriptors, etc.)
class SegmentationWriterUnit : public VideoUnit {
 public:
  SegmentationWriterUnit(const SegmentationWriterUnitOptions& options);
  ~SegmentationWriterUnit() = default;

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

 private:
  SegmentationWriterUnitOptions options_;

  int video_stream_idx_ = -1;
  int seg_stream_idx_;
  int frame_number_ = 0;

  int original_width_ = 0;
  int original_height_ = 0;

  std::unique_ptr<SegmentationWriter> writer_;
};

struct SegmentationReaderUnitOptions {
  // Input filename segmentation is read from.
  std::string filename;

  // Output segmentation stream.
  std::string segment_stream_name = "SegmentationStream";
};

class SegmentationReaderUnit : public VideoUnit {
public:
  SegmentationReaderUnit(const SegmentationReaderUnitOptions& options);
  ~SegmentationReaderUnit() = default;

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append);

protected:
  virtual bool SeekImpl(int64_t dts);
  void ReadNextFrame(FrameSetPtr input);

private:
  SegmentationReaderUnitOptions options_;

  int seg_stream_index_;
  int frame_width_;
  int frame_height_;

  std::unique_ptr<SegmentationReader> reader_;
};

struct SegmentationRenderUnitOptions {
  // Optional when blend_alpha  == 1.
  std::string video_stream_name = "VideoStream";
  std::string segment_stream_name = "SegmentationStream";
  std::string out_stream_name = "RenderedRegionStream";

  // Amount previous image should be blended with input video.
  float blend_alpha = 0.5;

  // Hierarchy level to render. Specify value >= 1 for absolute level,
  // 0 for over-segmentation or value in (0, 1) as fractional level w.r.t.
  // number of hierarchy levels in first chunk.
  float hierarchy_level = 0;
  bool highlight_edges = true;
  bool draw_shape_descriptors = false;
  bool concat_with_source = false;
};

class SegmentationRenderUnit : public VideoUnit {
public:
  SegmentationRenderUnit(const SegmentationRenderUnitOptions& options);
  ~SegmentationRenderUnit() = default;

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);

private:
  SegmentationRenderUnitOptions options_;

  int vid_stream_idx_;
  int seg_stream_idx_;

  int frame_width_ = 0;
  int frame_height_ = 0;
  int frame_width_step_ = 0;

  int prev_chunk_id_ = -1;
  int frame_number_ = 0;

  std::unique_ptr<cv::Mat> frame_buffer_;

  // Holds the segmentation for the current chunk.
  std::unique_ptr<SegmentationDesc> seg_hier_;
};

}  // namespace segmentation.

#endif // SEGMENTATION_UNIT_H__
