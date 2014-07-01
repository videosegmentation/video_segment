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

#ifndef VIDEO_UNIT_H__
#define VIDEO_UNIT_H__

#include "base/base.h"

#include <boost/circular_buffer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/mutex.hpp>
#include <glog/logging.h>

namespace cv {
  class Mat;
}

namespace video_framework {

enum VideoPixelFormat { PIXEL_FORMAT_RGB24,
                        PIXEL_FORMAT_BGR24,
                        PIXEL_FORMAT_ARGB32,
                        PIXEL_FORMAT_ABGR32,
                        PIXEL_FORMAT_RGBA32,
                        PIXEL_FORMAT_BGRA32,
                        PIXEL_FORMAT_YUV422,
                        PIXEL_FORMAT_LUMINANCE };

// Returns number of channels for each PixelFormat.
int PixelFormatToNumChannels(VideoPixelFormat);

// Base class for different Frame's. Can not be instantiated directly.
// Supports checked casting from Frame to derived Frame's via As[Ptr|Ref]<Type>.
class Frame : public base::TypedType {
 protected:
  Frame(const std::type_info* type, int64_t pts = 0) : base::TypedType(type), pts_(pts) {
  }

 public:
  virtual ~Frame() {
  }

  int64_t pts() const { return pts_; }
  void set_pts(int64_t pts) { pts_ = pts; }

  // Casting method for derived frames inherited from TypedType.
  // Usage: Frame* frame = new DerivedFrame(...);
  // // Function will fails with CHECK for anything else than a DerivedFrame
  // DerivedFrame& derived = frame->AsRef<DerivedFrame>();
  // OR:
  // const DerivedFrame* derived_ptr = frame->AsPtr<DerivedFrame>();
  // template <class T> const T* AsPtr();
  // template <class T> T* AsMutablePtr();
  // template <class T> T& AsRef();
  // template <class T> const T& As();

 private:
  int64_t pts_;
};

// Basic Frame container that holds any binary data of a specific size captured at a
// specific time (described by pts_, timebase is microseconds, i.e. pts_ * 1e-6
// specifies time in seconds).
// A derived class can opt to save all its data into the binary data_ member
// but this is not required.
// Data is always initialized to be zero.
class DataFrame : public Frame {
public:
  DataFrame(int size = 0, int64_t pts = 0)
      : Frame(&typeid(DataFrame), pts), data_(size, 0) {
  };

  virtual ~DataFrame() {
  }

  const uint8_t* data() const { return &data_[0]; }
  uint8_t* mutable_data() { return &data_[0]; }

  int size() const { return data_.size(); }

  DataFrame Copy() const {
    return *this;
  }

protected:
  // Prohibit accidental copies.
  DataFrame(const DataFrame&) = default;
  DataFrame& operator=(const DataFrame&) = default;
  DataFrame(const std::type_info* type, int size, int64_t pts)
    : Frame(type, pts), data_(size, 0) {
  }

protected:
  // Vector holds underlying data.
  std::vector<uint8_t> data_;
};

// Stores type T.
template <class T>
class ValueFrame : public Frame {
public:
  // Uses DataFrame's storage for T.
  ValueFrame<T>(const T& t, int64_t pts = 0)
      : Frame(&typeid(ValueFrame<T>), pts), value_(t) {
  }

  const T& Value() const {
    return value_;
  }

  void SetValue(const T& t) {
    value_ = t;
  }

 private:
  T value_;
};

template <class T>
class PointerFrame : public Frame {
 public:
  PointerFrame(std::unique_ptr<T> ptr, int64_t pts = 0)
    : Frame(&typeid(PointerFrame<T>), pts), ptr_(std::move(ptr)) {
  }

  const T* Ptr() const {
    return ptr_.get();
  }

  T* MutablePtr() {
    return ptr_.get();
  }

  const T& Ref() const {
    return *ptr_;
  }

  std::unique_ptr<T> release() {
    return std::move(ptr_);
  }

 private:
  std::unique_ptr<T> ptr_;
};

// Fundamental class for video frames.
class VideoFrame : public DataFrame {
public:
  VideoFrame(int width,
             int height,
             int channels,
             int width_step = 0,    // If zero, will be set to width * channels.
             int64_t pts = 0);

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  int width_step() const { return width_step_; }

  // Returns MatView onto the data
  void MatView(cv::Mat* view) const;

private:
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;      // bytes per pixel.
  int width_step_ = 0;
};

// TODO(dcastro): LOWPRI: Implement an audio frame.

// Basic Stream class.
class DataStream : public base::TypedType {
public:
  DataStream(const std::string& stream_name)
    : base::TypedType(&typeid(*this)), stream_name_(stream_name) {
  }

  virtual std::string stream_name() { return stream_name_; }
  virtual ~DataStream() {};

  // Casting method for derived streams derived from TypedType.
  // Usage: DataStream* stream = new DerivedStream(...);
  // // Function will fails with CHECK for anything else than a DerivedStream
  // const DerivedStream& derived = stream->As<DerivedStream>();
  // template <class T> const T& As() const;

  // Non-copyable.
  DataStream(const DataStream&) = delete;
  DataStream& operator=(const DataStream&) = delete;

 protected:
  // For use by derived classes.
  DataStream(const std::type_info* type, const std::string& stream_name)
      : base::TypedType(type), stream_name_(stream_name) {
  }

 protected:
  std::string stream_name_;
};

// Derived class for VideoStreams.
class VideoStream : public DataStream {
public:
  VideoStream(int width,
              int height,
              int width_step,
              float fps = 0,                          // Set to zero if not known.
              VideoPixelFormat pixel_format = PIXEL_FORMAT_BGR24,
              const std::string& stream_name = "VideoStream")
      : DataStream(&typeid(VideoStream), stream_name),
        frame_width_(width),
        frame_height_(height),
        width_step_(width_step),
        fps_(fps),
        pixel_format_(pixel_format),
        original_width_(width),
        original_height_(height) {
  }

  int frame_width() const { return frame_width_; }
  int frame_height() const { return frame_height_; }
  int width_step() const { return width_step_; }
  float fps() const { return fps_; }
  VideoPixelFormat pixel_format() const { return pixel_format_; }

  int original_width() const { return original_width_; }
  int original_height() const { return original_height_; }

  void set_original_width(int width) { original_width_ = width; }
  void set_original_height(int height) { original_height_ = height; }

private:
  int frame_width_;
  int frame_height_;
  int width_step_;
  float fps_;               // Not necessary exact.
  VideoPixelFormat pixel_format_;

  int original_width_;
  int original_height_;
};

// Describes a Segmentation stream. 
class SegmentationStream : public DataStream {
public:
  SegmentationStream(int frame_width,
                     int frame_height,
                     const std::string& stream_name = "SegmentationStream")
      : DataStream(&typeid(SegmentationStream), stream_name),
        frame_width_(frame_width),
        frame_height_(frame_height) {
  }

  int frame_width() const { return frame_width_; }
  int frame_height() const { return frame_height_; }

private:
  int frame_width_;
  int frame_height_;
};

typedef std::vector<std::shared_ptr<Frame>> FrameSet;
typedef std::vector<std::shared_ptr<DataStream>> StreamSet;
typedef std::shared_ptr<FrameSet> FrameSetPtr;

class VideoPool;
class VideoPipelineSource;

// Specifies how fast unit is processing frames.
// Only roots and extensions such as pipeline sources are enforcing RatePolicies,
// regular units must not alter their behavior.

// If max_rate_ != 0, root unit is limited to max_rate.
// If dynamic_rate is true, rate update is performed after startup_frames and
// re-evaluated after specifed update_interval (in seconds).
// dynamic_rate_scale specifies multiplier to the minimum rate when rate gets updated.
// Recommened use: Root unit (video reader, capture unit) < 1
//                 Pipeline units > 1
// In addition, root units query the queue size across all units. If the maximum queue
// size is above the queue_throttle_threshold, scale is halfed for every
// num_throttle_frames above the threshold. However throttled rate is limited to be
// above min_throttle_rate to avoid stalling the execution.
struct RatePolicy {
  RatePolicy() : max_rate(0),
                 dynamic_rate(false),
                 dynamic_rate_scale(1.0f),
                 startup_frames(0),
                 update_interval(0),
                 queue_throttle_threshold(8),
                 num_throttle_frames(4),
                 min_throttle_rate(0.2f) {
  }

  RatePolicy(float max_rate_,
             bool dynamic_rate_,
             float dynamic_rate_scale_,
             int startup_frames_,
             float update_interval_) : RatePolicy() {
    max_rate = max_rate_;
    dynamic_rate = dynamic_rate_;
    dynamic_rate_scale = dynamic_rate_scale_;
    startup_frames = startup_frames_;
    update_interval = update_interval_;
  }

  float max_rate;
  bool dynamic_rate;
  float dynamic_rate_scale;
  int startup_frames;
  float update_interval;
  int queue_throttle_threshold;
  int num_throttle_frames;
  float min_throttle_rate;
};

// Base class for any VideoUnit.
class VideoUnit {
public:
  // Common interface for all VideoUnit's. Derive and re-implement as desired.
  // Every unit should at least implement OpenStreams and ProcessFrame or the
  // corresponding *FromSender decorator functions.
  // VideoUnit::Run() is designed to make calls to the decorator functions.

  // OpensStreams gets passed a StreamSet that holds all Streams created by parents of
  // the current unit. Use to determine index of streams you are interested in or
  // to add additional streams.
  virtual bool OpenStreams(StreamSet* set) { return true; }

  // ProcessFrame gets the current FrameSetPtr in input. Use to
  // do any desired processing and push_back to output when done.
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output) {
    output->push_back(input);
  }

  // Any PostProcessing goes here. Will be called by parent node iteratively as long as
  // true is returned (in which case append must not be empty).
  // Return false to signal end of processing.
  virtual bool PostProcess(std::list<FrameSetPtr>* append) {
    return false;
  }

  // Decorator functions for explicit caller representation. For units with multiple input
  // support, e.g. VideoPool.
  // Simply forward to corresponding function by default (treat all senders equally).
  virtual bool OpenStreamsFromSender(StreamSet* set, const VideoUnit* sender) {
    return OpenStreams(set);
  }

  virtual void ProcessFrameFromSender(FrameSetPtr input,
                                      std::list<FrameSetPtr>* output,
                                      const VideoUnit* sender) {
    return ProcessFrame(input, output);
  }

  virtual bool PostProcessFromSender(std::list<FrameSetPtr>* append,
                                     const VideoUnit* sender) {
    return PostProcess(append);
  }

  // Rate management. Rate management is performed in downstream direction,
  // i.e. propagated from the roots to all its children.
  // By default, units do not respond to LimitRate() calls, overload LimtRateImpl() in
  // derived units to define behavior.
  void LimitRate(float fps);

public:
  void AddChild(VideoUnit* child);
  void AttachTo(VideoUnit* parent);
  void RemoveChild(VideoUnit* child);
  void RemoveFrom(VideoUnit* parent);
  bool HasChild(VideoUnit* child);

  VideoUnit* ParentUnit() const { return parent_; }

  // Return root for this unit.
  VideoUnit* RootUnit();

  // Initiate processing in batch (effectively PrepareProcessing + Run).
  bool PrepareAndRun();

  // Process all frames at a time.
  virtual bool Run();

  // Run with rate policy. Note, this can only be invoked on the root unit.
  virtual bool RunRateLimited(const RatePolicy& rate_policy);

  // Frame based processing.
  bool PrepareProcessing();

  // Process one frame at a time. Returns true, if more frames are present,
  // otherwise false.
  bool NextFrame();

  // Seeking during Frame-based processing.
  // Returns if position was changed due to the call of Seek.
  bool Seek();
  bool Seek(int64_t pts);

  VideoUnit ();
  virtual ~VideoUnit();

protected:
  // Returns whether the seek resulted in a change of position.
  // Children are only called with SeekImpl if return value is true.
  virtual bool SeekImpl(int64_t pts) { return true; }

  // Returns index for stream_name in set, -1 if not found.
  int FindStreamIdx(const std::string& stream_name, const StreamSet* set);

  // The two next preceding rate calls use a circular buffer of size buffer_size
  // to determine current rate and unit rate. Standard is a buffer of 64 frames,
  // adjust to fit your application.
  void SetRateBufferSize(int buffer_size);

  // Overload to perform rate limitation. Most units that consume and produce a single
  // FrameSet will likely not implement this method. FrameSet producer's such as video
  // file readers, and camera readers also do not have to implement this method, as the
  // framework already implements this functionality for root units. Extensions to the
  // framework such as pipeline sources should implement this method.
  virtual void LimitRateImpl(float fps) { }
public:
  // Average period spend in ProcessFrame in seconds.
  float UnitPeriod() const;

  // Average fps of ProcessFrame.
  float UnitRate() const;

  // Returns minimum rate of this unit and all of its children.
  float MinTreeRate() const;

  // Queue size if applicable (only pipeline sinks implement this).
  virtual int GetQueueSize() const { return 0; }

  // Returns maximum of queue sizes across the tree.
  int MaxTreeQueueSize() const;

  void PrintTree() const;

protected:
  // Generate's a new frame and outputs to output. Return's false if no more frames
  // are available. In that case output might still be non-empty.
  bool NextFrameImpl(const VideoUnit* sender, std::list<FrameSetPtr>* output);

  // Called by PrepareProcessing to setup graph, calls PrepareProcessingFromSender for
  // this unit.
  virtual bool OpenStreamsImpl(StreamSet*, const VideoUnit* sender);

  // Called from parent unit for each output frame. Calls ProcessFrameFromSender for this
  // unit.
  virtual void ProcessFrameImpl(const FrameSetPtr frame_set_ptr,
                                const VideoUnit* sender);

  // Called for root unit and for each unit once no more frames are available.
  // Repeatedly calls PostProcessFromSender until it returns false (no more frames
  // are output by unit).
  virtual void PostProcessImpl(const VideoUnit* sender);

  // Compiles a list of all children under the current unit. When called from the root
  // unit returns the whole graph as list.
  virtual void GetDownStreamUnits(std::list<const VideoUnit*>* down_stream_units) const;

  // Over-ride to prevent PostProcessImpl from calling itself recursively for all
  // children.
  virtual bool PostProcessingPassToChildren() { return true; }

  // Recursive tree printing function
  virtual std::string PrintTreeImpl(int indent) const;

private:
  std::vector<VideoUnit*> children_;
  VideoUnit* parent_;

  int stream_sz_;
  bool initialized_;

  boost::circular_buffer<float> unit_period_buffer_;
  mutable boost::mutex buffer_mutex_;

  RatePolicy rate_policy_;
  boost::posix_time::ptime rate_policy_updated_;

  friend class VideoPool;
  friend class VideoPipelineSource;
};

}  // namespace video_framework.

#endif // VIDEO_UNIT_H__
