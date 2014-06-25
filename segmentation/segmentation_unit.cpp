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


#include "segmentation/segmentation_unit.h"

#include <gflags/gflags.h>
#include <google/protobuf/repeated_field.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "base/base_impl.h"
#include "video_framework/flow_reader.h"
#include "segment_util/segmentation_render.h"
#include "segmentation/dense_segmentation.h"
#include "segmentation/region_segmentation.h"

DEFINE_bool(strip_to_essentials, false, "If set to true, SegmentationWriter "
                                        "strips segmentation result to essential "
                                        "information for annotation.");
namespace segmentation {

DenseSegmentationUnit::DenseSegmentationUnit(
    const DenseSegmentationUnitOptions& options,
    const DenseSegmentationOptions* dense_seg_options)
    : options_(options) {
  if (dense_seg_options) {
    dense_seg_options_ = *dense_seg_options;
  }
  // else default options.
}

bool DenseSegmentationUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  video_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find video stream!\n";
    return false;
  }

  // Get video stream info.
  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();

  if (vid_stream.pixel_format() != PIXEL_FORMAT_BGR24) {
    LOG(ERROR) << "Expecting video format to be BGR24.\n";
    return false;
  }

  // Flow stream present?
  if (!options_.flow_stream_name.empty()) {
    flow_stream_idx_ = FindStreamIdx(options_.flow_stream_name, set);
    if (flow_stream_idx_ < 0) {
      LOG(ERROR) << "Flow stream specified but not present";
      return false;
    }
  } else {
    flow_stream_idx_ = -1;
  }

  // Add segmentation stream.
  set->push_back(std::shared_ptr<DataStream>(
        new SegmentationStream(frame_width_,
                               frame_height_,
                               options_.segment_stream_name)));

  if (!OpenFeatureStreams(set)) {
    LOG(ERROR) << "Could not open feature streams.";
    return false;
  }

  // Initialize actual segmentation object.
  dense_seg_ = CreateDenseSegmentation();
  SetRateBufferSize(dense_seg_->ChunkSize() * 3);

  return true;
}

bool DenseSegmentationUnit::OpenFeatureStreams(StreamSet*) {
  return true;
}

std::unique_ptr<DenseSegmentation> DenseSegmentationUnit::CreateDenseSegmentation() {
  return std::unique_ptr<DenseSegmentation>(
      new DenseSegmentation(dense_seg_options_,
                            frame_width_,
                            frame_height_));
}

void DenseSegmentationUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  VLOG(1) << "Processing frame #" << input_frames_;

  vector<cv::Mat> features;
  ExtractFrameSetFeatures(input, &features);

  cv::Mat flow;
  if (input_frames_ > 0 && flow_stream_idx_ >= 0) {
    const DenseFlowFrame& flow_frame =
      input->at(flow_stream_idx_)->As<DenseFlowFrame>();
    flow = flow_frame.MatViewInterleaved();
    DCHECK(!flow.empty());
  }

  frame_set_buffer_.push_back(input);
  ++input_frames_;

  vector<std::unique_ptr<SegmentationDesc>> results;
  if (dense_seg_->ProcessFrame(false,
                               &features,
                               flow_stream_idx_ >= 0 ? &flow : nullptr,
                               &results) > 0) {
    OutputSegmentation(&results, output);
  }
}

void DenseSegmentationUnit::ExtractFrameSetFeatures(
    FrameSetPtr input, std::vector<cv::Mat>* features) {
  // Appearance only.
  CHECK_NOTNULL(features);
  const VideoFrame& video_frame = input->at(video_stream_idx_)->As<VideoFrame>();
  cv::Mat mat_view;
  video_frame.MatView(&mat_view);
  features->push_back(mat_view);
}

bool DenseSegmentationUnit::PostProcess(list<FrameSetPtr>* append) {
  vector<std::unique_ptr<SegmentationDesc>> results;
  if (dense_seg_->ProcessFrame(true, nullptr, nullptr, &results) > 0) {
    OutputSegmentation(&results, append);
  }

  return false;
}

void DenseSegmentationUnit::OutputSegmentation(
    std::vector<std::unique_ptr<SegmentationDesc>>* results,
    std::list<FrameSetPtr>* output) {
  for (int k = 0; k < results->size(); ++k) {
    FrameSetPtr frame_set = frame_set_buffer_.front();
    frame_set_buffer_.pop_front();
    int64_t pts = frame_set->at(video_stream_idx_)->pts();
    frame_set->push_back(std::shared_ptr<Frame>(
          new PointerFrame<SegmentationDesc>(std::move((*results)[k]), pts)));
    output->push_back(frame_set);
    ++output_frames_;
  }
  
  // Report current streaming size. This is used to indicate progress to the user.
  LOG(INFO) << "__STREAMING_SIZE__: " << output_frames_ << "\n";
}

RegionSegmentationUnit::RegionSegmentationUnit(
    const RegionSegmentationUnitOptions& options,
    const RegionSegmentationOptions* region_options)
    : options_(options) {
  if (region_options) {
    region_options_ = *region_options;
  }
  // else use default.
  SetRateBufferSize(300);
}

bool RegionSegmentationUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  video_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

  if (video_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find video stream!\n";
    return false;
  }

  // Get video stream info.
  const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();

  frame_width_ = vid_stream.frame_width();
  frame_height_ = vid_stream.frame_height();

  if (vid_stream.pixel_format() != PIXEL_FORMAT_BGR24) {
    LOG(ERROR) << "Expecting video format to be BGR24.\n";
    return false;
  }

  // Flow stream present?
  if (!options_.flow_stream_name.empty()) {
    flow_stream_idx_ = FindStreamIdx(options_.flow_stream_name, set);
    if (flow_stream_idx_ < 0) {
      LOG(ERROR) << "Flow stream specified but not present";
      return false;
    }
  } else {
    flow_stream_idx_ = -1;
  }

  // Get segmentation stream.
  seg_stream_idx_ = FindStreamIdx(options_.segment_stream_name, set);

  if (seg_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find Segmentation stream!\n";
    return false;
  }

  if(!OpenFeatureStreams(set)) {
    LOG(ERROR) << "Error opening feature streams!";
    return false;
  }

  region_seg_ = CreateRegionSegmentation();

  return true;
}

void RegionSegmentationUnit::ProcessFrame(FrameSetPtr input, list<FrameSetPtr>* output) {
  // Retrieve Segmentation.
  PointerFrame<SegmentationDesc>* seg_frame = input->at(seg_stream_idx_)->AsMutablePtr<
      PointerFrame<SegmentationDesc>>();

  const SegmentationDesc* desc = seg_frame->Ptr();

  std::vector<cv::Mat> features;
  ExtractFrameSetFeatures(input, &features);
  frame_set_buffer_.push_back(input);

  vector<std::unique_ptr<SegmentationDesc>> results;
  region_seg_->ProcessFrame(false,
                            desc,
                            &features,
                            &results);

  // Remove segmentation. Will be replaced by hierarchical segmentation. Keep pts alive.
  seg_frame->release();

  // Free flow or video frames?
  if (options_.free_video_frames) {
    input->at(video_stream_idx_).reset();
  }

  if (flow_stream_idx_ >= 0 && options_.free_flow_frames) {
    input->at(flow_stream_idx_).reset();
  }

  if (!results.empty()) {
    OutputSegmentation(&results, output);
  }

  ++num_input_frames_;
}

bool RegionSegmentationUnit::PostProcess(list<FrameSetPtr>* append) {
  vector<std::unique_ptr<SegmentationDesc>> results;
  if (region_seg_->ProcessFrame(true, nullptr, nullptr, &results) > 0) {
    OutputSegmentation(&results, append);
  }
  return false;
}

void RegionSegmentationUnit::OutputSegmentation(
    std::vector<std::unique_ptr<SegmentationDesc>>* results,
    std::list<FrameSetPtr>* output) {
  for (int k = 0; k < results->size(); ++k) {
    FrameSetPtr frame_set = frame_set_buffer_.front();
    frame_set_buffer_.pop_front();
    const int64_t pts = frame_set->at(seg_stream_idx_)->pts();

    // Reset with hierarchical segmentation.
    frame_set->at(seg_stream_idx_).reset(
        new PointerFrame<SegmentationDesc>(std::move((*results)[k]), pts));
    output->push_back(frame_set);
  }
}

bool RegionSegmentationUnit::OpenFeatureStreams(StreamSet* set) {
  return true;
}

std::unique_ptr<RegionSegmentation> RegionSegmentationUnit::CreateRegionSegmentation() {
  region_options_.use_flow = flow_stream_idx_ >= 0;
  return std::unique_ptr<RegionSegmentation>(new RegionSegmentation(region_options_,
                                                                    frame_width_,
                                                                    frame_height_));
}

void RegionSegmentationUnit::ExtractFrameSetFeatures(
    FrameSetPtr input, std::vector<cv::Mat>* features) {
  CHECK_NOTNULL(features);

  // By default we only use appearance and flow features.
  const VideoFrame& frame = input->at(video_stream_idx_)->As<VideoFrame>();
  cv::Mat image_view;
  frame.MatView(&image_view);

  features->push_back(image_view);

  if (flow_stream_idx_ >=0) {
    if (num_input_frames_ > 0) {
      const DenseFlowFrame& flow_frame =
        input->at(flow_stream_idx_)->As<DenseFlowFrame>();
      cv::Mat flow = flow_frame.MatViewInterleaved();
      features->push_back(flow);
    } else {
      features->push_back(cv::Mat());
    }
  }
}

SegmentationWriterUnit::SegmentationWriterUnit(
    const SegmentationWriterUnitOptions& options)
  : options_(options) {
    writer_.reset(new SegmentationWriter(options_.filename));
    if (FLAGS_strip_to_essentials) {
      options_.strip_to_essentials = true;
    }
}

bool SegmentationWriterUnit::OpenStreams(StreamSet* set) {

  if (!options_.video_stream_name.empty()) {
    video_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

    if (video_stream_idx_ < 0) {
      LOG(ERROR) << "Could not find Video stream!\n";
      return false;
    }
   
    // Get video stream info.
    const VideoStream& vid_stream = set->at(video_stream_idx_)->As<VideoStream>();
    original_width_ = vid_stream.original_width();
    original_height_ = vid_stream.original_height();
  }

  // Get segmentation stream.
  seg_stream_idx_ = FindStreamIdx(options_.segment_stream_name, set);

  if (seg_stream_idx_ < 0) {
    LOG(ERROR) << "Could not find Segmentation stream!\n";
    return false;
  }

  frame_number_ = 0;
  std::vector<int> header_entries;
  header_entries.push_back(1);   // use vectorization.
  header_entries.push_back(0);   // no shape moments.
  return writer_->OpenFile(header_entries);
}

void SegmentationWriterUnit::ProcessFrame(FrameSetPtr input,
                                          list<FrameSetPtr>* output) {
  // Retrieve Segmentation.
  const PointerFrame<SegmentationDesc>& seg_frame =
    input->at(seg_stream_idx_)->As<PointerFrame<SegmentationDesc>>();

  SegmentationDesc desc = seg_frame.Ref();  // Create local copy.
  if (original_width_ != desc.frame_width() || original_height_ != desc.frame_height()) {
    if (!desc.has_vector_mesh()) {
      LOG(WARNING) << "Downscale requested but vector mesh is not present. ";
    } else {
      ScaleVectorization(original_width_, original_height_, &desc);
      if (options_.remove_rasterization) {
        RemoveRasterization(&desc);
      } else {
        ReplaceRasterizationFromVectorization(&desc);
      }
    }
  } else {
    if (desc.has_vector_mesh() && options_.remove_rasterization) {
      RemoveRasterization(&desc);
    }
  }

  if (options_.strip_to_essentials) {
    // We use a different encoding scheme here. Binary, without using protobuffers.
    // The segmentation is stripped to its essentials. The use-case is visualization
    // in flash UI.
    string stripped_data;
    StripToEssentials(desc, true, false, &stripped_data);
    writer_->AddSegmentationDataToChunk(stripped_data, seg_frame.pts());
  } else {
    writer_->AddSegmentationToChunk(desc, seg_frame.pts());
  }

  output->push_back(input);
  ++frame_number_;
}

bool SegmentationWriterUnit::PostProcess(list<FrameSetPtr>* append) {
  writer_->WriteTermHeaderAndClose();
  return false;
}

SegmentationReaderUnit::SegmentationReaderUnit(
    const SegmentationReaderUnitOptions& options) : options_(options) {
  reader_.reset(new SegmentationReader(options_.filename));
}

bool SegmentationReaderUnit::OpenStreams(StreamSet* set) {
  bool res = reader_->OpenFileAndReadHeaders();
  reader_->SegmentationResolution(&frame_width_, &frame_height_);

  // Add segmentation stream.
  SegmentationStream* seg_stream = new SegmentationStream(frame_width_,
                                                          frame_height_,
                                                          options_.segment_stream_name);
  set->push_back(shared_ptr<DataStream>(seg_stream));
  seg_stream_index_ = set->size() - 1;
  return res;
}

void SegmentationReaderUnit::ProcessFrame(FrameSetPtr input,
                                          list<FrameSetPtr>* output) {
  ReadNextFrame(input);
  output->push_back(input);
}

bool SegmentationReaderUnit::PostProcess(list<FrameSetPtr>* append) {
  // If reader used as source, add frames until empty.
  if (reader_->RemainingFrames()) {
    CHECK_EQ(seg_stream_index_, 0)
      << "Reader encountered remaining frames but not used as source.";
    FrameSetPtr input(new FrameSet);
    ReadNextFrame(input);
    append->push_back(input);
    return true;
  }
  return false;
}

void SegmentationReaderUnit::ReadNextFrame(FrameSetPtr input) {
  // Read frame.
  std::unique_ptr<SegmentationDesc> segmentation(new SegmentationDesc());
  if (!reader_->ReadNextFrame(segmentation.get())) {
    LOG(ERROR) << "Could not read from segmentation.";
    return;
  }

  input->push_back(std::shared_ptr<PointerFrame<SegmentationDesc>>(
        new PointerFrame<SegmentationDesc>(std::move(segmentation))));
}

bool SegmentationReaderUnit::SeekImpl(int64_t pts) {
  vector<int64_t>::const_iterator pos =
      std::lower_bound(reader_->TimeStamps().begin(),
                       reader_->TimeStamps().end(),
                       pts);
  if (pos == reader_->TimeStamps().end() || *pos != pts)
    return false;

  reader_->SeekToFrame(pos - reader_->TimeStamps().begin());
  return true;
}

SegmentationRenderUnit::SegmentationRenderUnit(
    const SegmentationRenderUnitOptions& options) : options_(options) {
}

bool SegmentationRenderUnit::OpenStreams(StreamSet* set) {
  // Find video stream idx.
  float fps = 0;
  int frame_count = 0;
  if (options_.video_stream_name.empty()) {
    if (options_.blend_alpha != 1.f) {
      options_.blend_alpha = 1.f;
      LOG(WARNING) << "No video stream request. Fixing blend alpha to 1.";
    }

    fps = 25;         // Standard values.
    frame_count = 0;  // Unknown.
    vid_stream_idx_ = -1;
  } else {
    vid_stream_idx_ = FindStreamIdx(options_.video_stream_name, set);

    if (vid_stream_idx_ < 0) {
      LOG(ERROR) << "Could not find Video stream!\n";
      return false;
    }

    // Get video stream info.
    const VideoStream& vid_stream = set->at(vid_stream_idx_)->As<VideoStream>();
    frame_width_ = vid_stream.frame_width();
    frame_height_ = vid_stream.frame_height();
    frame_width_step_ = vid_stream.width_step();
    fps = vid_stream.fps();
  }

  // Get segmentation stream.
  seg_stream_idx_ = FindStreamIdx(options_.segment_stream_name, set);

  if (seg_stream_idx_ < 0) {
    LOG(ERROR) << "SegmentationRenderUnit::OpenStreams: "
               << "Could not find Segmentation stream!\n";
    return false;
  }

  if (frame_width_ == 0) {
    // Read dimensions from segmentation stream.
    const SegmentationStream& seg_stream =
        set->at(seg_stream_idx_)->As<SegmentationStream>();
    frame_width_ = seg_stream.frame_width();
    frame_height_ = seg_stream.frame_height();
    frame_width_step_ = frame_width_ * 3;
    if (frame_width_step_ % 4) {
      frame_width_step_ += 4 - frame_width_step_ % 4;
    }
  }

  const int actual_height = frame_height_ * (options_.concat_with_source ? 2 : 1);

  // Add region output stream.
  VideoStream* rendered_stream = new VideoStream(frame_width_,
                                                 actual_height,
                                                 frame_width_step_,
                                                 fps,
                                                 PIXEL_FORMAT_BGR24,
                                                 options_.out_stream_name);

  set->push_back(shared_ptr<DataStream>(rendered_stream));

  // Allocate render buffer.
  frame_buffer_.reset(new cv::Mat(frame_height_, frame_width_, CV_8UC3));
  return true;
}

void SegmentationRenderUnit::ProcessFrame(FrameSetPtr input,
                                         list<FrameSetPtr>* output) {
  // Dummy pts, if no video frame present.
  float pts = frame_number_ * 100;

  cv::Mat frame_view;
  if (vid_stream_idx_ >= 0) {
    const VideoFrame& frame = input->at(vid_stream_idx_)->As<VideoFrame>();
    frame.MatView(&frame_view);
    pts = frame.pts();
  }

  // Retrieve Segmentation.
  const PointerFrame<SegmentationDesc>& seg_frame =
    input->at(seg_stream_idx_)->As<PointerFrame<SegmentationDesc>>();

  const SegmentationDesc& desc = seg_frame.Ref();

  // If it is the first frame, save it into seg_hier_ and determine overall
  // hierarchy render level.
  if (seg_hier_ == nullptr) {
    seg_hier_.reset(new SegmentationDesc(desc));

    // Fractional level.
    if (options_.hierarchy_level != floor(options_.hierarchy_level)) {
      options_.hierarchy_level =
        (int)(options_.hierarchy_level * seg_hier_->hierarchy_size());
    }

    options_.hierarchy_level = std::min<int>(options_.hierarchy_level,
                                             seg_hier_->hierarchy_size() - 1);
  }

  // Update hierarchy when one present.
  if (desc.hierarchy_size() > 0) {
    *seg_hier_ = desc;
  }

  // Local pointer to support oversegmentations.
  const Hierarchy* hierarchy = nullptr;
  if (seg_hier_->hierarchy_size() > 0) {
    hierarchy = &seg_hier_->hierarchy();
  }

  // Render to temporary frame.
  RenderRegionsRandomColor((int)options_.hierarchy_level,
                           options_.highlight_edges,
                           options_.draw_shape_descriptors,
                           desc,
                           hierarchy,
                           frame_buffer_.get());

  // Allocate new output frame.
  VideoFrame* render_frame =
      new VideoFrame(frame_width_,
                     frame_height_ * (options_.concat_with_source ? 2 : 1),
                     3,
                     frame_width_step_,
                     pts);

  cv::Mat render_view;
  render_frame->MatView(&render_view);

  if (options_.concat_with_source) {
    CHECK_GE(vid_stream_idx_, 0)
      << "Request concatenation with source but no video stream present.";


    cv::Mat top(render_view, cv::Range(0, frame_height_), cv::Range(0, frame_width_));
    frame_buffer_->copyTo(top);

    cv::Mat bottom(render_view,
                   cv::Range(frame_height_, 2 * frame_height_),
                   cv::Range(0, frame_width_));
    frame_view.copyTo(bottom);
  } else {
    if (vid_stream_idx_ >= 0) {
      // Blend with input.
      cv::addWeighted(frame_view, 1.0f - options_.blend_alpha,
                      *frame_buffer_, options_.blend_alpha,
                      0,
                      render_view);
    } else {
      frame_buffer_->copyTo(render_view);
    }
  }

  if (desc.chunk_id() != prev_chunk_id_) {
    prev_chunk_id_ = desc.chunk_id();
    string output_text = base::StringPrintf("Change to chunk id %d", desc.chunk_id());

    cv::putText(render_view, output_text, cv::Point(5, frame_height_ - 30),
                cv::FONT_HERSHEY_PLAIN, 0.8,
                cv::Scalar(255, 255, 255));
  }

  std::string output_text = base::StringPrintf("Frame #%04d", frame_number_);

  cv::putText(render_view, output_text, cv::Point(5, frame_height_ - 10),
              cv::FONT_HERSHEY_PLAIN, 0.8,
              cv::Scalar(255, 255, 255));

  input->push_back(shared_ptr<DataFrame>(render_frame));
  output->push_back(input);
  ++frame_number_;
}

}  // namespace segmentation.
