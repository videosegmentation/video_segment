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

#ifndef VIDEO_SEGMENT_IMAGEFILTER_IMAGE_UTIL_H__
#define VIDEO_SEGMENT_IMAGEFILTER_IMAGE_UTIL_H__

#include "base/base.h"
#include <opencv2/core/core.hpp>

namespace imagefilter {

// Helper functions for pointer indexing.
template <class T>
T* PtrOffset(T* t, int offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uchar*>(t) + offset);
}

template <class T>
const T* PtrOffset(const T* t, int offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const uchar*>(t) + offset);
}

// Dimension checks
inline bool HasSameDimensions(const cv::Mat& img_1, const cv::Mat& img_2) {
  return (img_1.cols == img_2.cols && img_1.rows == img_2.rows);
}

inline cv::Point trunc(const cv::Point2f& pt) {
  return cv::Point((int)pt.x, (int)pt.y);
}

inline cv::Point2f rotate(const cv::Point2f& pt, float angle) {
  const float c = cos(angle);
  const float s = sin(angle);
  return cv::Point2f(c * pt.x - s * pt.y, s * pt.x + c * pt.y);
}

inline cv::Point2f normalize(const cv::Point2f& pt) {
  return pt * (1.0f / norm(pt));
}

inline cv::Point3f homogPt(const cv::Point2f& pt) {
  return cv::Point3f(pt.x, pt.y, 1.0f);
}

inline cv::Point2f homogPt(const cv::Point3f& pt) {
  if (pt.z != 0) {
    return cv::Point2f(pt.x / pt.z, pt.y / pt.z);
  } else {
    return cv::Point2f(0, 0);
  }
}

template <class T>
inline T clamp(const T& value, const T& a, const T& b) {
  return std::max<T>(a, std::min<T>(value, b));
}

}  // namespace imagefilter.

#endif  // VIDEO_SEGMENT_IMAGEFILTER_IMAGE_UTIL_H__
