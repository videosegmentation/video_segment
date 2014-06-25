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


#include "imagefilter/image_filter.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <tbb/tbb.h>

#include "base/base_impl.h"
#include "imagefilter/image_util.h"

namespace imagefilter {

namespace {

// Parallel filter over image rows.
class ParallelBilateralGray {
public:
ParallelBilateralGray(const cv::Mat& image,
                      cv::Mat* output,
                      const float sigma_space,
                      const float sigma_color,
                      const std::vector<float>& expLUT,
                      const std::vector<int>& space_ofs,
                      const std::vector<float>& space_weights,
                      const int space_sz,
                      const float scale)
  : image_(image),
    output_(output),
    sigma_space_(sigma_space),
    sigma_color_(sigma_color),
    expLUT_(expLUT),
    space_ofs_(space_ofs),
    space_weights_(space_weights),
    space_sz_(space_sz),
    scale_(scale) {
}

void operator()(const tbb::blocked_range<int>& r) const {
  for (int i = r.begin(); i != r.end(); ++i) {
    const float* src_ptr = image_.ptr<float>(i);
    float* dst_ptr = output_->ptr<float>(i);

    const int img_width = image_.cols;
    for (int j = 0; j < img_width; ++j, ++src_ptr, ++dst_ptr) {
      float my_val = src_ptr[0];
      float weight_sum = 0;
      float val_sum = 0;

      for (int k = 0; k < space_sz_; ++k) {
        const float* local_ptr = PtrOffset(src_ptr, space_ofs_[k]);
        float val_diff = my_val - local_ptr[0];
        val_diff *= val_diff;
        const float weight = space_weights_[k] * expLUT_[int(val_diff * scale_)];
        weight_sum += weight;
        val_sum += local_ptr[0] * weight;
      }
      if (weight_sum > 0) {
        *dst_ptr = val_sum / weight_sum;
      } else {
        *dst_ptr = 0;
      }
    }
  }
}

private:
const cv::Mat& image_;
cv::Mat* output_;
const float sigma_space_;
const float sigma_color_;
const std::vector<float>& expLUT_;
const std::vector<int>& space_ofs_;
const std::vector<float>& space_weights_;
const int space_sz_;
const float scale_;
};

// Parallel filter over image rows.
class ParallelBilateralColor {
public:
ParallelBilateralColor(const cv::Mat& image,
                       cv::Mat* output,
                       const float sigma_space,
                       const float sigma_color,
                       const std::vector<float>& expLUT,
                       const std::vector<int>& space_ofs,
                       const std::vector<float>& space_weights,
                       const int space_sz,
                       const float scale,
                       const int num_bins)
  : image_(image),
    output_(output),
    sigma_space_(sigma_space),
    sigma_color_(sigma_color),
    expLUT_(expLUT),
    space_ofs_(space_ofs),
    space_weights_(space_weights),
    space_sz_(space_sz),
    scale_(scale),
    num_bins_(num_bins) {
}

void operator()(const tbb::blocked_range<int>& r) const {
  CHECK_EQ(3, image_.channels());
  for (int i = r.begin(); i != r.end(); ++i) {
    const float* src_ptr = image_.ptr<float>(i);
    float* dst_ptr = output_->ptr<float>(i);
    const int img_width = image_.cols;
    for (int j = 0; j < img_width; ++j, src_ptr += 3, dst_ptr += 3) {
      float my_b = src_ptr[0];
      float my_g = src_ptr[1];
      float my_r = src_ptr[2];
      float weight_sum = 0;
      float sum_r = 0;
      float sum_g = 0;
      float sum_b = 0;
      for (int k = 0; k < space_sz_; ++k) {
        const float* local_ptr = PtrOffset(src_ptr, space_ofs_[k]);
        const float diff_b = my_b - local_ptr[0];
        const float diff_g = my_g - local_ptr[1];
        const float diff_r = my_r - local_ptr[2];
        const int idx = (int)((diff_b * diff_b + diff_g * diff_g + diff_r * diff_r) * scale_);
        const float weight = space_weights_[k] * expLUT_[idx];
        weight_sum += weight;
        sum_b += local_ptr[0] * weight;
        sum_g += local_ptr[1] * weight;
        sum_r += local_ptr[2] * weight;
      }
      if (weight_sum > 0) {
        weight_sum = 1.0 / weight_sum;
        dst_ptr[0] = sum_b * weight_sum;
        dst_ptr[1] = sum_g * weight_sum;
        dst_ptr[2] = sum_r * weight_sum;
      } else {
        dst_ptr[0] = dst_ptr[1] = dst_ptr[2] = 0.0f;
      }
    }
  }
}

private:
const cv::Mat& image_;
cv::Mat* output_;
const float sigma_space_;
const float sigma_color_;
const std::vector<float>& expLUT_;
const std::vector<int>& space_ofs_;
const std::vector<float>& space_weights_;
const int space_sz_;
const float scale_;
const int num_bins_;
};

}  // namespace.

void BilateralFilter(const cv::Mat& image,
                     float sigma_space,
                     float sigma_color,
                     cv::Mat* output) {
  CHECK_NOTNULL(output);
  CHECK(HasSameDimensions(image, *output))
      << "Input and output image dimensions differ.";
  CHECK_EQ(image.depth(), CV_32F) << "Input image must be 32f.";
  CHECK(image.channels() == 3 || image.channels() == 1)
      << "Input image must have one or three channels.";

  // Setup temp image.
  // Cover 86.6% of the data.
  const int radius = sigma_space * 1.5f;
  const int diam = 2 * radius + 1;
  const int img_width = image.cols;
  const int img_height = image.rows;
  const int cn = image.channels();

  cv::Mat tmp_image_border(img_height + 2 * radius,
                           img_width + 2 * radius,
                           CV_32FC(cn));
  cv::copyMakeBorder(image, tmp_image_border,
                     radius, radius, radius, radius, cv::BORDER_REPLICATE);
  cv::Mat tmp_image(tmp_image_border, cv::Rect(radius, radius, img_width, img_height));

  // Calculate space offsets and weights.
  std::vector<int> space_ofs(diam * diam);
  std::vector<float> space_weights(diam * diam);
  int space_sz = 0;
  const float space_coeff = -0.5f / (sigma_space * sigma_space);

  for (int i = -radius; i <= radius; ++i) {
    for (int j = -radius; j <= radius; ++j) {
      int r2 = i*i + j*j;
      if (r2 > radius*radius)
        continue;

      space_ofs[space_sz] = i * tmp_image.step[0] + j * sizeof(float) * cn;
      space_weights[space_sz++] = exp(space_coeff * (float)r2);
    }
  }

  // Compute color exp-weight lookup table.
  double min_val, max_val;
  cv::Mat flattened_image = image.reshape(1);
  cv::minMaxLoc(flattened_image, &min_val, &max_val);

  // Compute slighlty larger diff range.
  const float diff_range =
      std::max<float>(1e-3f, (max_val - min_val) * (max_val - min_val) * cn * 1.02f);

  // 4K bins times channels.
  const int num_bins = (1 << 12) * cn;
  const float scale = (float) num_bins / diff_range;

  std::vector<float> expLUT(num_bins);
  const float color_coeff = -0.5 / (sigma_color * sigma_color);
  bool zero_reached = false;
  for (int i = 0; i < num_bins; ++i) {
    if (!zero_reached) {
      expLUT[i] = exp( (float)i / scale * color_coeff);
      zero_reached = (expLUT[i] < 1e-10);
    } else {
      expLUT[i] = 0;
    }
  }

  // Bilateral filtering.
  if (image.channels() == 1) {
     tbb::parallel_for(tbb::blocked_range<int>(0, img_height, img_height / 8),
                       ParallelBilateralGray(tmp_image,
                                             output,
                                             sigma_space,
                                             sigma_color,
                                             expLUT,
                                             space_ofs,
                                             space_weights,
                                             space_sz,
                                             scale),
                                             tbb::simple_partitioner());
  } else {    // Color case.
    tbb::parallel_for(tbb::blocked_range<int>(0, img_height, img_height / 8),
                      ParallelBilateralColor(tmp_image,
                                             output,
                                             sigma_space,
                                             sigma_color,
                                             expLUT,
                                             space_ofs,
                                             space_weights,
                                             space_sz,
                                             scale,
                                             num_bins),
                                             tbb::simple_partitioner());
  }
}

}  // namespace imagefilter.
