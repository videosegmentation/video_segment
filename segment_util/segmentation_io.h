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

// Segmentation Reader and Writer to be used with segmentation.pb files.

// The binary, streaming file format we use has the following form:
// Each file contains various Headers followed by protobuffers containing the actual data.
// Format:
// Initial chunk, used for flags regarding the encoding.
// HEADER {
//   type                                         : char[4] = "HEAD"
//   num_entries (M)                              : int32
//   M flags                                      : int32[M]
// }
// Current header flags are:
// flags[0] : use_vectorization
// flags[1] : shape_moments_present.
// Use of header flags is application dependent.
//
// Multiple chunks of the form:
// CHUNK_HEADER {
//   type                                         : char[4] = "CHNK"
//   Header ID                                    : int32
//   Number of frames in chunk (N)                : int32
//   N FileOffsets for SEG_FRAMES                 : int64[N]
//   N TimeStamps (pts)                           : int64[N]
//   FileOffset of next chunk header              : int64
// }
//
// followed by N frames
// SEG_FRAME {
//   type                                         : char[4] = "SEGD"
//   Size of protobuffer in bytes (sz)            : int32
//   Protobuffer serialized to binary format      : char[sz]
// }
//
// with terminating header at the end
// TERM_HEADER {
//   type                                         : char[4] = "TERM"
//   number of chunks                             : int32
// }

#ifndef SEGMENTATION_IO_H
#define SEGMENTATION_IO_H

#include "base/base.h"

#ifdef __linux
#include <stdint.h>
#endif

#ifdef _WIN32
typedef __int64 int64_t;
typedef __int32 int32_t;
#endif

namespace segmentation {

class SegmentationDesc;

// Usage (user supplied variables in caps).
// SegmentationWriter writer(FILENAME);
// writer.OpenFile();
//
// for (int i = 0; i < NUM_FRAMES; ++i) {
//   writer.AddSegmentationToChunk(SEGMENTATION_FRAME[i], PTS[i]);
//   // When reasonable boundary is reached, write buffered chunk to file, e.g.
//   if ((i + 1) % 10 == 0) {
//     writer.WriteChunk();
//   }
// }
//
// writer.WriteTermHeaderAndClose();
class SegmentationWriter {
public:
  SegmentationWriter(const std::string& filename) : filename_(filename) {
  }

  // Returns false if file could not be opened. Pass list of optional header entries
  // to be written in file_header.
  bool OpenFile(const std::vector<int>& header_entries = std::vector<int>());

  // Buffers segmentation in chunk_buffer_.
  void AddSegmentationToChunk(const SegmentationDesc& desc, int64_t pts = 0);

  // Same as above if data was already serialized (or in stripped binary format).
  void AddSegmentationDataToChunk(const std::string& data, int64_t pts = 0);

  // Call to write whole chunk to file.
  void WriteChunk();

  // Finish file.
  void WriteTermHeaderAndClose();

  // Reuse writer for another file.
  void FlushAndReopen(const std::string& filename);
private:
  std::string filename_;
  std::ofstream ofs_;

  int num_chunks_ = 0;
  std::vector<std::string> chunk_buffer_;

  int64_t curr_offset_ = 0;
  std::vector<int64_t> file_offsets_;
  std::vector<int64_t> time_stamps_;

  std::vector<int> header_entries_;

  int total_frames_ = 0;
};

// Usage (user supplied variables in caps):
// SegmentationReader reader(FILE_NAME);
// reader.OpenFileAndReadHeaders();
// while (reader.RemainingFrames()) {
//   int frame_sz = reader.ReadNextFrameSize();
//   std::vector<uint8_t> buffer(frame_sz);
//   reader.ReadNextFrame(&buffer[0]);
//   
//   // Get segmentation protobuffer.
//   SegmentationDesc segmentation;
//   segmentation.ParseFromArray(&buffer[0], buffer.size());
//
//   // Process segmentation...
// }

class SegmentationReader {
public:
  // Creates new reader from filename. By default, if only vectorization
  // is saved to file, creates valid rasterization from vectorization.
  SegmentationReader(const std::string& filename,
                     bool valid_rasterization = true) : filename_(filename) {
  }

  ~SegmentationReader() {
    CloseFile();
  }

  bool OpenFileAndReadHeaders();

  // Reads and parses first frame, returns resolution, seeks back to current playhead.
  void SegmentationResolution(int* width, int* height);

  // Reads next frame from file. Returns true on success.
  bool ReadNextFrame(SegmentationDesc* desc);
  
  // Reads binary blob from file. Do not use directly to read SegmentationDesc, but
  // use ReadNextFrame instead.
  bool ReadNextFrameBinary(std::string* data);

  const std::vector<int32_t>& GetHeaderFlags() const { return header_flags_; }

  const std::vector<int64_t>& TimeStamps() const { return time_stamps_; }
  void SeekToFrame(int frame);

  int NumFrames() const { return file_offsets_.size(); }
  int RemainingFrames() const { return NumFrames() - curr_frame_; }

  void CloseFile() { ifs_.close(); }
private:
  std::vector<int64_t> file_offsets_;
  std::vector<int64_t> time_stamps_;
  std::vector<int32_t> header_flags_;

  int frame_sz_= 0;
  int curr_frame_ = 0;
  bool valid_rasterization_ = true;

  std::string filename_;
  std::ifstream ifs_;
};

// Utility functions.

// Converts a SegmentationDesc to a stripped down binary representation for compact
// representation and fast loading. For example, stripped format is used by Flash
// annotator.
// If save_vectorization is set to true, vectorization  is saved instead rasterization
// (indicated via flag).
// If save_shape_moments is set to true, corresponding flag needs to be set in header!
void StripToEssentials(const SegmentationDesc& desc,
                       bool save_vectorization,
                       bool save_shape_moments,
                       std::string* binary_rep);

}  // namespace segmentation.

#endif
