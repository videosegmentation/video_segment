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

#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_util.h"
#include "segment_util/segmentation.pb.h"

namespace segmentation {

namespace {
  template <class T> const char* ToConstCharPtr(const T* t) {
    return reinterpret_cast<const char*>(t);
  }
  template <class T> char* ToCharPtr(T* t) {
    return reinterpret_cast<char*>(t);
  }
}

bool SegmentationWriter::OpenFile(const vector<int>& header_entries) {
  header_entries_ = header_entries;

  // Open file to write
  VLOG(1) << "Writing segmentation to file " << filename_;
  ofs_.open(filename_.c_str(),
            std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

  if (!ofs_) {
    LOG(ERROR) << "Could not open " << filename_ << " to write!\n";
    return false;
  }

  num_chunks_ = 0;
  curr_offset_ = 0;

  // Write header information.
  ofs_.write("HEAD", 4);
  const int32_t num_entries = header_entries_.size();
  ofs_.write(ToConstCharPtr(&num_entries), sizeof(num_entries));
  for (int i = 0; i < num_entries; ++i) {
    ofs_.write(ToConstCharPtr(&header_entries_[i]), sizeof(header_entries_[i]));
  }
  curr_offset_ = 4 + 4 + num_entries * 4;

  return true;
}

void SegmentationWriter::AddSegmentationToChunk(const SegmentationDesc& desc,
                                                int64_t pts) {
  string data;
  desc.SerializeToString(&data);
  AddSegmentationDataToChunk(data, pts);
}

void SegmentationWriter::AddSegmentationDataToChunk(const string& data,
                                                    int64_t pts) {
  // Buffer for later.
  file_offsets_.push_back(curr_offset_);
  chunk_buffer_.push_back(data);

  curr_offset_ += data.size() + 4 + sizeof(int32_t);    // SEG_FRAME total size.
  time_stamps_.push_back(pts);
}

void SegmentationWriter::WriteChunk() {
  const int32_t num_frames = file_offsets_.size();
  const int32_t chunk_id = num_chunks_++;

  CHECK_EQ(file_offsets_.size(), chunk_buffer_.size());
  CHECK_EQ(file_offsets_.size(), time_stamps_.size());

  ofs_.write("CHNK", 4);
  ofs_.write(ToConstCharPtr(&chunk_id), sizeof(chunk_id));
  ofs_.write(ToConstCharPtr(&num_frames), sizeof(num_frames));

  int64_t size_of_header =           // HEADER total size.
    4 +
    2 * sizeof(int32_t) +
    num_frames * 2 * sizeof(int64_t) +
    sizeof(int64_t);

  // Advance offsets by size of header.
  curr_offset_ += size_of_header;
  for (size_t i = 0; i < file_offsets_.size(); ++i) {
    file_offsets_[i] += size_of_header;
  }

  // Write offsets and pts.
  for (size_t i = 0; i < file_offsets_.size(); ++i) {
    ofs_.write(ToConstCharPtr(&file_offsets_[i]), sizeof(file_offsets_[i]));
  }

  CHECK_EQ(file_offsets_.size(), time_stamps_.size());
  for (size_t i = 0; i < time_stamps_.size(); ++i) {
    ofs_.write(ToConstCharPtr(&time_stamps_[i]), sizeof(time_stamps_[i]));
  }

  // Write offset of next header.
  ofs_.write(ToConstCharPtr(&curr_offset_), sizeof(curr_offset_));

  // Write frames.
  CHECK_EQ(file_offsets_.size(), chunk_buffer_.size());
  for (size_t i = 0; i < chunk_buffer_.size(); ++i) {
    ofs_.write("SEGD", 4);
    int32_t frame_size = chunk_buffer_[i].length();
    ofs_.write(ToConstCharPtr(&frame_size), sizeof(frame_size));
    ofs_.write(ToConstCharPtr(&chunk_buffer_[i][0]), frame_size);
  }

  total_frames_ += chunk_buffer_.size();

  // Clear chunk information.
  chunk_buffer_.clear();
  file_offsets_.clear();
  time_stamps_.clear();
}

void SegmentationWriter::WriteTermHeaderAndClose() {
   if (!chunk_buffer_.empty()) {
     WriteChunk();
   }

   ofs_.write("TERM", 4);
   ofs_.write(ToConstCharPtr(&num_chunks_), sizeof(num_chunks_));
   ofs_.close();

   LOG(INFO) << "Wrote a total of " << total_frames_ << " frames.";
}

void SegmentationWriter::FlushAndReopen(const string& filename) {
  if (!chunk_buffer_.empty()) {
    WriteChunk();
  }

  WriteTermHeaderAndClose();
  filename_ = filename;
  curr_offset_ = 0;
  num_chunks_ = 0;
  OpenFile();
}

bool SegmentationReader::OpenFileAndReadHeaders() {
  // Open file.
  VLOG(1) << "Reading segmentation from file " << filename_;
  ifs_.open(filename_.c_str(), std::ios_base::in | std::ios_base::binary);

  if (!ifs_) {
    LOG(ERROR) << "Could not open segmentation file " << filename_ << "\n";
    return false;
  }

  // Read file offsets until TERM header.
  char header_type[5] = {0, 0, 0, 0, 0};
  int prev_header_id = -1;
  while (true) {
    ifs_.read(header_type, 4);

    // End of file, return.
    if (strcmp(header_type, "TERM") == 0) {
      break;
    }

    if (strcmp(header_type, "HEAD") == 0) {
      // Read header, save entries and continue to next chunk.
      int32_t num_entries;
      ifs_.read(ToCharPtr(&num_entries), sizeof(num_entries));
      header_flags_.resize(num_entries);
      for (int i = 0; i < num_entries; ++i) {
        ifs_.read(ToCharPtr(&header_flags_[i]), sizeof(header_flags_[i]));
      }
      continue;
    }

    // We only process chunk headers while skipping over seg frames.
    if (strcmp(header_type, "CHNK") != 0) {
      LOG(ERROR) << "Parsing error, expected chunk header at current offset."
                 << " Found: " << header_type;
      return false;
    }

    int32_t header_id;
    ifs_.read(ToCharPtr(&header_id), sizeof(header_id));
    CHECK_EQ(prev_header_id + 1, header_id)
        << prev_header_id << " " << header_id;

    prev_header_id = header_id;

    int32_t num_frames_in_chunk;
    ifs_.read(ToCharPtr(&num_frames_in_chunk), sizeof(num_frames_in_chunk));

    // Read offsets.
    for (int f = 0; f < num_frames_in_chunk; ++f) {
      int64_t offset;
      ifs_.read(ToCharPtr(&offset), sizeof(offset));
      file_offsets_.push_back(offset);
    }

    // Read timestamps.
    for (int f = 0; f < num_frames_in_chunk; ++f) {
      int64_t timestamp;
      ifs_.read(ToCharPtr(&timestamp), sizeof(timestamp));
      time_stamps_.push_back(timestamp);
    }

    int64_t next_header_pos;
    ifs_.read(ToCharPtr(&next_header_pos), sizeof(next_header_pos));
    ifs_.seekg(next_header_pos);
  }

  return true;
}

void SegmentationReader::SegmentationResolution(int* width, int* height) {
  CHECK_NOTNULL(width);
  CHECK_NOTNULL(height);

  const int curr_playhead = curr_frame_;
  SeekToFrame(0);

  // Read via binary as we don't need rasterization here.
  string data;
  ReadNextFrameBinary(&data);

  SegmentationDesc segmentation;
  segmentation.ParseFromString(data);

  *width = segmentation.frame_width();
  *height = segmentation.frame_height();

  if (curr_playhead < NumFrames()) {
    SeekToFrame(curr_playhead);
  }
}

void SegmentationReader::SeekToFrame(int frame) {
  CHECK_LT(frame, file_offsets_.size()) << "Requested frame out of bound.";
  curr_frame_ = frame;
}

bool SegmentationReader::ReadNextFrameBinary(string* data) {
  // Seek to next frame (to skip chunk headers).
  ifs_.seekg(file_offsets_[curr_frame_]);
  char header_type[5] = {0, 0, 0, 0, 0};
  ifs_.read(header_type, 4);
  if (strcmp(header_type, "SEGD") != 0) {
    LOG(ERROR) << "Expecting segmentation header. Error parsing file.";
    return false;
  }

  ifs_.read(ToCharPtr(&frame_sz_), sizeof(frame_sz_));

  data->resize(frame_sz_);
  ifs_.read(&data->at(0), frame_sz_);
  ++curr_frame_;

  return true;
}

bool SegmentationReader::ReadNextFrame(SegmentationDesc* desc) {
  CHECK_NOTNULL(desc);
  string data;
  if (!ReadNextFrameBinary(&data)) {
    LOG(ERROR) << "Could not read from file.";
    return false;
  }
  if (!desc->ParseFromString(data)) {
    LOG(ERROR) << "Could not parse segmentation proto.";
    return false;
  }
  if (valid_rasterization_ && desc->rasterization_removed()) {
    CHECK(desc->has_vector_mesh());
    ReplaceRasterizationFromVectorization(desc);
  }
  return true;
}

namespace {

template <class T> void write(std::ostringstream& stream, const T& t) {
  stream.write((const char*)&t, sizeof(t));
}

}

void StripToEssentials(const SegmentationDesc& desc,
                       bool save_vectorization,
                       bool save_shape_moments,
                       string* binary_rep) {
  std::ostringstream seg_data;

  int frame_width = desc.frame_width();
  int frame_height = desc.frame_height();
  
  write(seg_data, frame_width);
  write(seg_data, frame_height);

  if (save_vectorization) {
    CHECK(desc.has_vector_mesh());
    int num_points = desc.vector_mesh().coord_size();
    CHECK_LT(num_points, std::numeric_limits<short>::max())
      << "Overlflow, too many points for vectorization format!!!";

    LOG(INFO) << "Number of points: " << num_points;

    write(seg_data, num_points);
    for (float coord : desc.vector_mesh().coord()) {
      // Save coord as short.
      short short_coord = (short)(coord);
      write(seg_data, short_coord);
    }
  }

  // Number of regions.
  int num_regions = desc.region_size();
  write(seg_data, num_regions);

  // Save regions.
  for(int i = 0; i < num_regions; ++i) {
    const SegmentationDesc::Region2D& r = desc.region(i);

    // Id.
    int id = r.id();
    write(seg_data, id);

    if (save_vectorization) {
      int num_polygons = r.vectorization().polygon_size();
      write(seg_data, num_polygons);

      for (const auto& polygon : r.vectorization().polygon()) {
        short num_coords = polygon.coord_idx_size();
        uint8_t is_hole = polygon.hole();
        write(seg_data, num_coords);
        write(seg_data, is_hole);

        for (int coord : polygon.coord_idx()) {
          short short_coord = coord;
          write(seg_data, short_coord);
        }
      }
    } else {
      // Scanlines.
      int num_scan_inter = r.raster().scan_inter_size();
      write(seg_data, num_scan_inter);

      for (int j = 0; j < num_scan_inter; ++j) {
        int16_t y = r.raster().scan_inter(j).y();
        int16_t left = r.raster().scan_inter(j).left_x();
        int16_t right = r.raster().scan_inter(j).right_x();

        write(seg_data, y);
        write(seg_data, left);
        write(seg_data, right);
      }
    }

    // ShapeMoments.
    if (save_shape_moments) {
      int shape_sz = r.shape_moments().size();
      write(seg_data, shape_sz);
      int mean_x = r.shape_moments().mean_x();
      write(seg_data, mean_x);
      int mean_y = r.shape_moments().mean_y();
      write(seg_data, mean_y);
      int moment_xx = r.shape_moments().moment_xx();
      write(seg_data, moment_xx);
      int moment_xy = r.shape_moments().moment_xy();
      write(seg_data, moment_xy);
      int moment_yy = r.shape_moments().moment_yy();
      write(seg_data, moment_yy);
    }
  }

  // Save hierarchies.
  int hierarchy_size = desc.hierarchy_size();
  write(seg_data, hierarchy_size);

  for (int i = 0; i < hierarchy_size; ++i) {
    int num_regions = desc.hierarchy(i).region_size();
    write(seg_data, num_regions);

    for (int j = 0; j < num_regions; ++j) {
      const SegmentationDesc::CompoundRegion& r = desc.hierarchy(i).region(j);

      // Id.
      int id = r.id();
      write(seg_data, id);

      // Size.
      int size = r.size();
      write(seg_data, size);

      // Parent id.
      int parent_id = r.parent_id();
      write(seg_data, parent_id);

      // Children.
      int num_children = r.child_id_size();
      write(seg_data, num_children);

      for (int c = 0; c < num_children; ++c) {
        int child_id = r.child_id(c);
        write(seg_data, child_id);
      }

      // Start and end frame.
      int start_frame = r.start_frame();
      write(seg_data, start_frame);
      int end_frame = r.end_frame();
      write(seg_data, end_frame);
    }
  }

  seg_data.flush();
  *binary_rep = string(seg_data.str());
}

}  // namespace segmentation.
