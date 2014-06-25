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

#include "base/base_impl.h"

#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_render.h"
#include "segment_util/segmentation_util.h"

DEFINE_bool(text_format, false, "Outputs one protobuffer per frame as text.");
DEFINE_bool(binary_format, false, "Outputs one protobuffer per frame as binary.");
DEFINE_double(bitmap_ids, -1, "If set to >= 0, outputs one image per frame as "
                             "3 channel PNG. Pixel represents region ID as single 24bit "
                             "int (little endian!). "
                             "If in [0, 1] fractional level w.r.t. hiearchy is used.");
DEFINE_double(bitmap_color, -1,
             "If set to >= 0, outputs one image per frame as "
             "3 channel PNG. Pixel represents per-region random color as 3 "
             "channel uint. Region boundary highlighted in black. "
             "If in [0, 1] fractional level w.r.t. hiearchy is used.");
DEFINE_string(strip, "", "If set to <destination>, saves proto buffer in binary format, "
                          "stripping it to essentials.");
DEFINE_bool(use_rasterization, false, "If set always outputs rasterization in stripped "
                                      "format even if vectorization is present.");
DEFINE_string(input, "", "The input segmentation protobuffer (.pb). REQUIRED");
DEFINE_bool(logging, false, "If set output various logging information.");
DEFINE_string(output_dir, ".", "Output directory for bitmaps and protobufs.");

using namespace segmentation;

int main(int argc, char** argv) {
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_logging) {
    FLAGS_logtostderr = 1;
  }

  if (FLAGS_input.empty()) {
    std::cerr << "Input file not specified. Specify via -input.\n";
    return 1;
  }

  // Determine conversion mode.
  enum ConvMode { CONV_TEXT, CONV_BINARY, CONV_BITMAP_ID, CONV_BITMAP_COLOR, STRIP };
  ConvMode mode = CONV_TEXT;
  float hier_level = 0;
  string dest_filename;

  if (FLAGS_text_format) {
    mode = CONV_TEXT;
    std::cout << "Converting to text.";
  } else if (FLAGS_binary_format) {
    mode = CONV_BINARY;
    std::cout << "Converting to binary format.";
  } else if (FLAGS_bitmap_ids >= 0) {
    mode = CONV_BITMAP_ID;
    hier_level = FLAGS_bitmap_ids;
    std::cout << "Converting to id bitmaps for hierarchy level: " << hier_level << "\n";
  } else if (FLAGS_bitmap_color >= 0) {
    mode = CONV_BITMAP_COLOR;
    hier_level = FLAGS_bitmap_color;
    std::cout << "Converting to color bitmaps for hierarchy level: "
              << hier_level << "\n";
  } else if (!FLAGS_strip.empty()) {
    mode = STRIP;
    dest_filename = FLAGS_strip;
  } else {
    std::cout << "Unknown mode specified.\n";
    return 1;
  }
  std::string filename = FLAGS_input;

  // Read segmentation file.
  // Don't need rasterization when we are stripping file.
  const bool valid_rasterization = !FLAGS_strip.empty();
  SegmentationReader segment_reader(filename, valid_rasterization);
  segment_reader.OpenFileAndReadHeaders();
  std::vector<int> segment_headers = segment_reader.GetHeaderFlags();

  std::cout << "Segmentation file " << filename << " contains "
            << segment_reader.NumFrames() << " frames.\n";

  SegmentationWriter* writer = nullptr;
  bool use_vectorization = false;
  if (mode == STRIP) {
    writer = new SegmentationWriter(dest_filename);
    std::vector<int> header_entries;
    if (!FLAGS_use_rasterization && segment_headers.size() > 0) {
      header_entries.push_back(use_vectorization = segment_headers[0]);
    } else {
      header_entries.push_back(0);
    }

    header_entries.push_back(0);   // No shape moments.

    if (!writer->OpenFile(header_entries)) {
      std::cout << "Could not open destination file.\n";
      delete writer;
      return 1;
    }

    LOG(INFO) << "Stripping files with " << (use_vectorization ? "vectorization"
                                                               : "rasterization"); 
  }

  Hierarchy hierarchy;
  const int chunk_size = 100;    // By default use chunks of 100 frames.
  int absolute_level = -1;
  // Use absolute level if supplied.
  if (hier_level == 0 || hier_level >= 1) {
    absolute_level = hier_level;
  }

  for (int f = 0; f < segment_reader.NumFrames(); ++f) {
    segment_reader.SeekToFrame(f);

    // Read from file.
    SegmentationDesc segmentation;
    segment_reader.ReadNextFrame(&segmentation);

    if (segmentation.hierarchy_size() > 0) {
      hierarchy.Clear();
      hierarchy.MergeFrom(segmentation.hierarchy());
      // Convert fractional to constant absolute level.
      if (absolute_level < 0) {
        absolute_level = hier_level * (float)hierarchy.size();
        LOG(INFO) << "Selecting level " << absolute_level << " of " << hierarchy.size()
                  << std::endl;
      }
    }

    std::string curr_file = FLAGS_output_dir + "/";
    if (mode == CONV_TEXT) {
      curr_file += base::StringPrintf("frame%05d.pbtxt", f);
    } else if (mode == CONV_BINARY) {
      curr_file += base::StringPrintf("frame%05d.pb", f);
    } else {
      curr_file += base::StringPrintf("frame%05d.png", f);
    }

    if (f % 5 == 0) {
      std::cout << "Writing frame " << f << " of "
                << segment_reader.NumFrames() << "\n";
    }

    int frame_width = segmentation.frame_width();
    int frame_height = segmentation.frame_height();

    if (mode == CONV_BINARY) {
      std::ofstream ofs(curr_file, std::ios_base::out | std::ios_base::binary);
      segmentation.SerializeToOstream(&ofs);
    } else if (mode == CONV_TEXT) {
      std::ofstream ofs(curr_file, std::ios_base::out);
      ofs << segmentation.DebugString();
    } else if (mode == CONV_BITMAP_ID) {
      cv::Mat id_image(frame_height, frame_width, CV_32S);
      SegmentationDescToIdImage(absolute_level,
                                segmentation,
                                &hierarchy,
                                &id_image);

      // Get 4 channel view via evil casting.
      cv::Mat id_image_view(frame_height, frame_width, CV_8UC4, id_image.ptr<uint8_t>(0));

      // Discard most significant 8bit (little endian).
      vector<cv::Mat> id_channels;
      cv::split(id_image_view, id_channels);
      cv::Mat frame_buffer(frame_height, frame_width, CV_8UC3);
      cv::merge(&id_channels[0], 3, frame_buffer);
      cv::imwrite(curr_file, frame_buffer);
    } else if (mode == CONV_BITMAP_COLOR) {
      cv::Mat frame_buffer(frame_height, frame_width, CV_8UC3);
      RenderRegionsRandomColor(absolute_level,
                               true,
                               false,
                               segmentation,
                               &hierarchy,
                               &frame_buffer);
      cv::imwrite(curr_file, frame_buffer);
    } else if (mode == STRIP) {
      string stripped_data;
      StripToEssentials(segmentation,
                        use_vectorization,
                        false,   // no shape moments.
                        &stripped_data);
      writer->AddSegmentationDataToChunk(stripped_data, f);
      if (f > 0 && f % chunk_size == 0) {
        writer->WriteChunk();
      }
    }
  }

  if (mode == STRIP) {
    writer->WriteTermHeaderAndClose();
    delete writer;
  }

  segment_reader.CloseFile();
  return 0;
}
