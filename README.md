The Video Segmentation Project
=============

Main repository for the Video Segmentation Project.
Online implementation with annotation system available at
www.videosegmentation.com

To build you need the following build dependencies:
- [Boost](http://www.boost.org/)
- [FFMPEG](https://www.ffmpeg.org/)
- [Google protobuffer](https://code.google.com/p/protobuf/)
- [Google logging](https://code.google.com/p/google-glog/)
- [Google gflags](https://code.google.com/p/gflags/)
- Intel TBB (to be removed)
- [OpenCV](http://opencv.org/)

Configuration is done via cmake, *outside* the main source tree:

Assuming source is checked out under ~/video_segment

```shell
mkdir -p bin/seg_tree_sample
cd bin/seg_tree_sample
cmake ~/video_segment/seg_tree_sample
make -j4
```

This would build the binary seg_tree_sample in bin/seg_tree_sample/seg_tree_sample.

List of current executables:
- seg_tree_sample: Main segmentation algorithm
- segment_converter: Render segmentations to images, region images or text
- segment_renderer: Render segmentation to video
- segment_viewer: Interactive viewer
- video_example: Example to run video_framework single threaded or as pipeline


---

Algorithm with many improvements is loosely based on

Matthias Grundmann and Vivek Kwatra and Mei Han and Irfan Essa

Efficient Hierarchical Graph Based Video Segmentation

IEEE CVPR, 2010

http://www.cc.gatech.edu/cpl/projects/videosegmentation/

---

List of contributors over time:
- Matthias Grundmann
- Vivek Kwatra
- Mei Han
- Daniel Castro
- Chris McClanahan
- Irfan Essa
