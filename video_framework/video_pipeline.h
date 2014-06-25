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


#ifndef VIDEO_SEGMENT_VIDEO_FRAMEWORK_VIDEO_PIPELINE_H__
#define VIDEO_SEGMENT_VIDEO_FRAMEWORK_VIDEO_PIPELINE_H__

#include "base/base.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include "video_framework/video_unit.h"
#include "tbb/concurrent_queue.h"

namespace boost {
  class thread;
  class thread_group;
}

// Common usage for video pipeline:
// 1. Determine partition of your graph that you want to run independently
// 2. Connect disjoint parts by adding a PipelineSink as the last element and a
//    connected PipelineSource (don't use AttachTo) as first element of the next part.
// 3. Call PreprareProcessing on the root
// 4. Place root and each PipelineSource in a separate thread with Run() being
//    the main thread function.
namespace video_framework {

// Places Frame-Sets into a producer-consumer queue.
class VideoPipelineSource;

class VideoPipelineSink : public VideoUnit {
public:
  VideoPipelineSink();

  virtual bool OpenStreams(StreamSet* set) { return true; }
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);
  virtual bool PostProcess(std::list<FrameSetPtr>* append) {
    source_exhausted_ = true;
    // Don't terminate as long as we have items in our queue.
    return GetQueueSize() != 0;
  }

  int GetQueueSize() const { return frameset_queue_.unsafe_size(); }

protected:
  // Attached source runs in a different thread and calls PostProcess when
  // IsExhausted == false.
  virtual bool PostProcessingPassToChildren() { return false; }

private:
  // Returns first element from frameset_queue. Called by attached pipeline source.
  bool TryFetchingFrameSet(FrameSetPtr* ptr);
  bool IsExhausted() { return source_exhausted_; }

private:
  bool source_exhausted_;
  tbb::concurrent_queue<FrameSetPtr> frameset_queue_;

  int frame_number_;

  friend class VideoPipelineSource;
};

// Specifies how the source responds to LimitRate calls.
struct SourceRatePolicy {
  SourceRatePolicy(bool respond_to_limit_rate_, float rate_scale_) :
      respond_to_limit_rate(respond_to_limit_rate_), rate_scale(rate_scale_) {
  }

  SourceRatePolicy() = default;

  bool respond_to_limit_rate = false;
  float rate_scale = 1.0f;

  // If set > 0 rate is scaled down if attached sinks queue size is too large.
  int sink_max_queue_size = 0;
};

// Reads Frame-Sets from producer-consumer queue specified by sink
// and passes it to children at a pre-specified rate.
class VideoPipelineSource : public VideoUnit {
public:
  // By standard as fast as possible consumer source. If idle_unit is set with a suitable
  // root unit, pipeline source repeatedly calls ProcessFrame with an empty FrameSet
  // on the idle_unit.
  // For example, this can be used to monitor the state of the system.
  VideoPipelineSource(VideoPipelineSink* sink,
                      VideoUnit* idle_unit = nullptr,
                      const SourceRatePolicy& policy = SourceRatePolicy(),
                      float max_fps = 0);    // As fast as possible.

  virtual bool OpenStreams(StreamSet* set);

  // No input, nothing to do here.
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output) { }

  // Ditto.
  virtual bool PostProcess(std::list<FrameSetPtr>* append) { return false; }

  // Specialized run. Keeps querying queue until attached sink is exhausted.
  virtual bool Run();

  void SetIdleUnit(VideoUnit* idle_unit) { idle_unit_ = idle_unit; }

  // Sets the sink to be monitored during rate limiting calls. This is usally
  // different from sink that this source is attached to, but the sink
  // this segment of the pipeline is writing to.
  void SetMonitorSink(VideoPipelineSink* sink) { monitor_sink_ = sink; }

protected:
  // Implements rate limitation call from root unit.
  void LimitRateImpl(float fps);

private:
  // Called during wait periods.
  void OnIdle();

protected:
  VideoPipelineSink* sink_ = nullptr;
  VideoUnit* idle_unit_ = nullptr;
  VideoPipelineSink* monitor_sink_ = nullptr;
  SourceRatePolicy source_rate_policy_;
  float max_fps_ = 0;

  int frame_num_ = 0;
  boost::posix_time::ptime prev_process_time_;
};

// Helper class to run a pipeline.
// Note, visualization should be run in main thread.
class VideoPipelineInvoker {
 public:
  VideoPipelineInvoker();
  ~VideoPipelineInvoker();
  VideoPipelineInvoker(const VideoPipelineInvoker&) = delete;
  VideoPipelineInvoker& operator=(const VideoPipelineInvoker&) = delete;

  void RunRoot(VideoUnit* root);
  void RunRootRateLimited(const RatePolicy& policy, VideoUnit* root);
  void RunPipelineSource(VideoPipelineSource* source);

  void WaitUntilPipelineFinished();

private:
  std::unique_ptr<boost::thread_group> threads_;
  std::vector<boost::thread*> thread_ptrs_;
};

struct VideoPipelineStatsOptions {
  int frame_width = 320;
  int frame_height = 240;
  int max_queue_height = 20;
  std::string video_stream_name = "VideoPipelineStats";
};

// Visualizes frame queue sizes of passed pipeline sinks.
class VideoPipelineStats : public VideoUnit {
public:
  // Pass size of desired bar plot rendering.
  VideoPipelineStats(const VideoPipelineStatsOptions& options,
                     std::vector<std::pair<const VideoPipelineSink*, std::string> >& sinks)
      : options_(options), sinks_(sinks) {
  }

  virtual bool OpenStreams(StreamSet* set);
  virtual void ProcessFrame(FrameSetPtr input, std::list<FrameSetPtr>* output);

protected:
  VideoPipelineStatsOptions options_;
  std::vector<std::pair<const VideoPipelineSink*, std::string> > sinks_;
  int frame_width_step_;
  boost::posix_time::ptime start_time_;

  std::string video_stream_name_;
};

}  // namespace video_framework.

#endif  // VIDEO_SEGMENT_VIDEO_FRAMEWORK_VIDEO_PIPELINE_H__
