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

#include "video_unit.h"

#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>
#include <numeric>
#include <opencv2/core/core.hpp>

#include "base/base_impl.h"

namespace video_framework {

int PixelFormatToNumChannels(VideoPixelFormat pix_fmt) {
  switch (pix_fmt) {
    case PIXEL_FORMAT_RGB24:
      return 3;
    case PIXEL_FORMAT_BGR24:
      return 3;
    case PIXEL_FORMAT_RGBA32:
    case PIXEL_FORMAT_ARGB32:
    case PIXEL_FORMAT_BGRA32:
    case PIXEL_FORMAT_ABGR32:
      return 4;
    case PIXEL_FORMAT_YUV422:
      return 2;
    case PIXEL_FORMAT_LUMINANCE:
      return 1;
    default:
      LOG(ERROR) << "PixelFormatToNumChannels: unknown pixel format";
      return 0;
  }
}

VideoFrame::VideoFrame(int width,
                       int height,
                       int channels,
                       int width_step,
                       int64_t pts)
    : DataFrame(&typeid(VideoFrame),
                (width_step == 0 ? width * channels : width_step) * height,
                pts),
      width_(width),
      height_(height),
      channels_(channels),
      width_step_(width_step == 0 ? width * channels : width_step) {
}

void VideoFrame::MatView(cv::Mat* view) const {
  CHECK(view) << "View not set.";
  *view = cv::Mat(height_, width_, CV_8UC(channels_), (void*)data(), width_step_);
}

VideoUnit::VideoUnit() : parent_(nullptr), initialized_(false)  {
  unit_period_buffer_.set_capacity(64);
}

VideoUnit::~VideoUnit() {

}

void VideoUnit::LimitRate(float fps) {
  // Call per-unit implementation function.
  LimitRateImpl(fps);

  // Recursively call children, DFS.
  std::for_each(children_.begin(), children_.end(),
                std::bind2nd(
                    std::mem_fun1_t<void, VideoUnit, float>(&VideoUnit::LimitRate),
                    fps));
}

void VideoUnit::AddChild(VideoUnit* v) {
  // Ensure we dont add v to this twice.
  RemoveChild(v);
  children_.push_back(v);
  v->parent_ = this;
}

void VideoUnit::AttachTo(VideoUnit* v) {
  // Ensure we dont add this to v twice.
  v->RemoveChild(this);
  v->children_.push_back(this);
  parent_ = v;
}

void VideoUnit::RemoveChild(VideoUnit* v) {
  vector<VideoUnit*>::iterator i = std::find(children_.begin(), children_.end(), v);
  if (i != children_.end()) {
    children_.erase(i);
    (*i)->parent_ = 0;
  }
}

void VideoUnit::RemoveFrom(VideoUnit* v) {
  v->RemoveChild(this);
  parent_ = 0;
}

bool VideoUnit::HasChild(VideoUnit* v) {
  return std::find(children_.begin(), children_.end(), v) != children_.end();
}

VideoUnit* VideoUnit::RootUnit() {
  VideoUnit* current = this;
  while (current->parent_ != 0) {
    current = current->parent_;
  }

  return current;
}

bool VideoUnit::PrepareAndRun() {
  if (!PrepareProcessing()) {
    return false;
  }
  initialized_ = true;

  Run();
  return true;
}

bool VideoUnit::Run() {
  if (!initialized_) {
    LOG(ERROR) << "Unit is not initialized, call PrepareProcessing first.";
    return false;
  }

  // Initial frame_set is empty.
  PostProcessImpl(nullptr);		// Root, call without sender.
  return true;
}

bool VideoUnit::RunRateLimited(const RatePolicy& rate_policy) {
  CHECK(this == RootUnit()) << "Only root unit can enforce rate policy.";
  rate_policy_ = rate_policy;
  rate_policy_updated_ = boost::posix_time::microsec_clock::local_time();
  PostProcessImpl(nullptr);		// Root, call without sender.
  return true;
}

bool VideoUnit::PrepareProcessing() {
  CHECK(this == RootUnit()) << "Only root unit can initiate setup.";

  // Populate Streams.
  StreamSet stream_set;
  if (!OpenStreamsImpl(&stream_set, nullptr)) {  // Root, call without sender.
    return false;
  }

  initialized_ = true;
  return true;
}

bool VideoUnit::NextFrameImpl(const VideoUnit* sender, list<FrameSetPtr>* output) {
  // Get current time.
  boost::posix_time::ptime time_before_call =
        boost::posix_time::microsec_clock::local_time();

  const bool end_of_stream = !PostProcessFromSender(output, nullptr);

  // Measure difference.
  boost::posix_time::ptime time_after_call =
      boost::posix_time::microsec_clock::local_time();

  if (!output->empty()) {
    const float time_per_frame_passed =
        boost::posix_time::time_period(time_before_call,
                                       time_after_call).length().total_microseconds() *
                                       1.e-6f  / output->size();

    // Store in secs.
    {
      boost::mutex::scoped_lock lock(buffer_mutex_);
      for (int i = 0; i < output->size(); ++i) {
        unit_period_buffer_.push_back(time_per_frame_passed);
      }
    }

    // Check that size of each frame set corresponds with size of streamset.
    for (const auto& frame_set_ptr : *output) {
      CHECK_EQ(frame_set_ptr->size(), stream_sz_)
          << "Unit of type: " << typeid(*this).name()
          << "\nNumber of streams set in PostProcessImpl not consistent "
          << "with returned FrameSet. " << stream_sz_ << " streams for this unit, "
          << "but FrameSet consists of " << frame_set_ptr->size();

    }
  }
  return end_of_stream;
}

bool VideoUnit::NextFrame() {
  if (!initialized_) {
    LOG(ERROR) << "Unit is not initialized, call PrepareProcessing first.";
    return false;
  }

  list<FrameSetPtr> append;
  const bool end_of_stream = NextFrameImpl(nullptr,   // no sender.
                                           &append);

  if (!append.empty()) {
    // Output generated, pass to children.
    for (const auto& frame_set_ptr : append) {
      // Pass immediately to children.
      for (auto& child_ptr : children_) {
        child_ptr->ProcessFrameImpl(frame_set_ptr, this);
      }
    }
  }

  if (end_of_stream) {
    // End of stream, signal to children.
    for (auto& child_ptr : children_) {
      child_ptr->PostProcessImpl(this);
    }
    return false;
  } else {
    // More frames.
    return true;
  }
}

bool VideoUnit::Seek() {
  return Seek(0);
}

bool VideoUnit::Seek(int64_t seek_time) {
  bool changed_pos = this->SeekImpl(seek_time);
  if(changed_pos) {
    for (auto& child : children_) {
      child->Seek(seek_time);
    }
  }
  return changed_pos;
}

int VideoUnit::FindStreamIdx(const string& stream_name, const StreamSet* set) {
  for (StreamSet::const_iterator i = set->begin(); i != set->end(); ++i) {
    if ((*i)->stream_name() == stream_name) {
      return i - set->begin();
    }
  }

  return -1;
}

void VideoUnit::SetRateBufferSize(int buffer_size) {
  boost::mutex::scoped_lock lock(buffer_mutex_);
  unit_period_buffer_.set_capacity(buffer_size);
}

float VideoUnit::UnitPeriod() const {
  boost::mutex::scoped_lock lock(buffer_mutex_);
  float total_secs =
      std::accumulate(unit_period_buffer_.begin(),  unit_period_buffer_.end(), 0.f);
  if (total_secs) {
    return total_secs / unit_period_buffer_.size();
  } else {
    return 0;
  }
}

float VideoUnit::UnitRate() const {
  const float unit_period = UnitPeriod();
  if (unit_period > 0) {
    return 1.0f / unit_period;
  } else {
    return 1e4f;
  }
}

float VideoUnit::MinTreeRate() const {
  float min_rate = UnitRate();
  for (const auto& child : children_) {
    min_rate = std::min(min_rate, child->MinTreeRate());
  }
  return min_rate;
}

int VideoUnit::MaxTreeQueueSize() const {
  int max_queue_size = GetQueueSize();
  for(const auto& child : children_) {
    max_queue_size = std::max(max_queue_size, child->MaxTreeQueueSize());
  }

  return max_queue_size;
}

bool VideoUnit::OpenStreamsImpl(StreamSet* set, const VideoUnit* sender) {
  // Remember current size of stream set (in case unit adds streams).
  const int prev_stream_sz = set->size();

  // Call units implementation function.
  if (!OpenStreamsFromSender(set, sender)) {
    return false;
  }

  // Remember stream size to check consistency during ProcessFrame.
  stream_sz_ = set->size();

  // Check for duplicate names (will break FindStreamIdx).
  for (int i = prev_stream_sz; i < stream_sz_; ++i) {
    const string curr_stream_name = (*set)[i]->stream_name();
    if (FindStreamIdx(curr_stream_name, set) < i) {
      LOG(ERROR) << "Duplicate stream found: " << curr_stream_name;
      return false;
    }
  }

  // Recursively call children.
  for (auto& child_ptr : children_) {
    if (!child_ptr->OpenStreamsImpl(set, this)) {
      return false;
    }
  }

  return true;
}

void VideoUnit::ProcessFrameImpl(FrameSetPtr frame_set_ptr, const VideoUnit* sender) {
  list<FrameSetPtr> output;

  // Get current time to measure unit period of this unit.
  boost::posix_time::ptime time_before_call =
      boost::posix_time::microsec_clock::local_time();

  // Call derived implementation function for this unit.
  ProcessFrameFromSender(frame_set_ptr, &output, sender);

  // Consistency that size of each frame set corresponds with size of StreamSet.
  for (const auto& frame_set_ptr : output) {
    CHECK_EQ(frame_set_ptr->size(), stream_sz_)
        << "Unit of type: " << typeid(*this).name()
        << "\nNumber of streams set in OpenStreams not consistent "
        << "with returned FrameSet. " << stream_sz_ << " streams for this unit, "
        << "but FrameSet consists of " << frame_set_ptr->size();
  }

  // Measure difference.
  boost::posix_time::ptime time_after_call =
      boost::posix_time::microsec_clock::local_time();

  float micros_passed =
      boost::posix_time::time_period(time_before_call,
                                     time_after_call).length().total_microseconds();

  // Store in buffer in seconds.
  {
    boost::mutex::scoped_lock lock(buffer_mutex_);
    unit_period_buffer_.push_back(micros_passed * 1.e-6f);
  }

  // Pass each output frame to children.
  for (const auto& output_ptr : output) {
    for (auto& child_ptr : children_) {
      child_ptr->ProcessFrameImpl(output_ptr, this);
    }
  }
}

void VideoUnit::PostProcessImpl(const VideoUnit* sender) {
  // Call until PostProcessFromSender returns false.
  while (true) {
    list<FrameSetPtr> append;
    // Process one frame.
    bool end_of_stream = NextFrameImpl(sender, &append);

    // Wait with passing this on, if rate_limited. Only root enforces rate limitation.
    if (rate_policy_.max_rate > 0) {
      CHECK(this == RootUnit()) << "Expected root unit.";
      const float target_frame_duration = 1.0f / rate_policy_.max_rate;
      // If not on target fps, wait a bit.
      const float last_processing_period = unit_period_buffer_.back();
      int wait_time = (int) ((target_frame_duration - last_processing_period) * 1e6);
      // wait_time in microseconds
      if (wait_time > 100) {   // Dont enforce slight fluctuations.
        boost::thread::sleep(boost::get_system_time() +
                             boost::posix_time::microseconds(wait_time));
      }
    }

    // Perform rate update if specified.
    if (rate_policy_.dynamic_rate) {
      CHECK(this == RootUnit()) << "Rate policy can only be adapted by root unit";
      if (rate_policy_.startup_frames >= unit_period_buffer_.capacity()) {
        LOG(ERROR) << "Specified startup frames in RatePolicy larger than rate buffer "
                   << "size of " << unit_period_buffer_.capacity()
                   << " . Rate policy is NOT enforced.";
      }

      if (rate_policy_.startup_frames < unit_period_buffer_.size()) {
        // Enforce policy.
        boost::posix_time::ptime current_time =
          boost::posix_time::microsec_clock::local_time();
        const float time_since_last_update =
            boost::posix_time::time_period(
                rate_policy_updated_, current_time).length().total_microseconds();

        if (time_since_last_update * 1.e-6f > rate_policy_.update_interval) {
          // Get minimum tree rate.
          const float min_rate = MinTreeRate();
          float rate_scale = 1.0f;

	        // Determine size of buffering queue's across pipeline threads.
          const int max_queue_size = MaxTreeQueueSize();

	         // If queues are getting too big throttle
          // (avoid memory consumption from exploding) 
          if (max_queue_size > rate_policy_.queue_throttle_threshold) {
            // For every N frames over threshold, half the rate scale. 
            rate_scale *=
                pow(0.5, (float)(max_queue_size - rate_policy_.queue_throttle_threshold) /
                rate_policy_.num_throttle_frames);
            // Limit throttling to avoid stall.
            rate_scale = std::max(rate_scale, rate_policy_.min_throttle_rate);
          }

	        // Update rate.
          rate_policy_.max_rate = min_rate * rate_scale * rate_policy_.dynamic_rate_scale;

          // Enforce across all children.
          LimitRate(min_rate);
          rate_policy_updated_ = current_time;
        }
      }
    }

    // If output was generated, pass to children.
    for (const auto& frame_set_ptr : append) {
      for (auto& child_ptr : children_) {
        child_ptr->ProcessFrameImpl(frame_set_ptr, this);
      }
    }

    if (end_of_stream) {
      // This unit is done.
      break;
    }

    if (append.empty()) {
      // Nothing was output by the unit but end of stream has not been reached.
      // Ensure we don't hammer downstream units with requests.
      boost::thread::sleep(boost::get_system_time() +
                           boost::posix_time::microseconds(500));

    }
  }  // end while.

  // End of stream, signal to children.
  if (PostProcessingPassToChildren()) {
    for (auto& child_ptr : children_) {
      child_ptr->PostProcessImpl(this);
    }
  }
}

void VideoUnit::PrintTree() const {
  LOG(INFO) << "Tree layout:\n" << PrintTreeImpl(0);
}

std::string VideoUnit::PrintTreeImpl(int indent) const {
  std::string result =
    base::StringPrintf("%*s" "%s\n", indent, " ",
                       base::demangle(typeid(*this).name()).c_str());
  for (auto& child_ptr : children_) {
    result += child_ptr->PrintTreeImpl(indent + 2);
  }
}


void VideoUnit::GetDownStreamUnits(list<const VideoUnit*>* down_stream_units) const {
  for (auto& child_ptr : children_) {
    child_ptr->GetDownStreamUnits(down_stream_units);
    down_stream_units->push_back(child_ptr);
  }
}

}  // namespace video_framework.
