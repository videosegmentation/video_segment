The Video Segmentation Project
=============

Main repository for the Video Segmentation Project.
Online implementation with annotation system available at
www.videosegmentation.com

To build you need the following build dependencies:
- [Boost](http://www.boost.org/)
- [FFMPEG](https://www.ffmpeg.org/) - Works with 2.2.3*.
- [Google protobuffer](https://code.google.com/p/protobuf/)
- [Google logging](https://code.google.com/p/google-glog/)
- [Google gflags](https://code.google.com/p/gflags/)
- [OpenCV](http://opencv.org/) - Works with 2.4.7*.
- [Jsoncpp](https://github.com/open-source-parsers/jsoncpp) (only needed by segment_util).

*We haven't tested it on earlier versions of ffmpeg / opencv yet. Feel free to
try and let us know.

We have put together preliminary commands to build and run seg_tree_sample on
a fresh installation of Ubuntu 14.04 LTS. [You may see those here](https://docs.google.com/document/d/1idKVuSn-8Muhx4bIk5peXzaaYmDgK8bDgw4mgMn8gUY/edit?usp=sharing).

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
- seg_tree_sample: 

Runs main segmentation algorithm (over segmentation and hierarchical segmentation).
Example (will create INPUT.mp4.pb segmentation result)
```shell
./seg_tree_sample --input_file=INPUT.mp4 --logging --write-to-file
```

- segment_converter:

Render segmentations to images (randomized color), region images (id stored as 24 bit int)
or protobuffers (extract one protobuffer per frame in either binary or text format).
Example (will render images at 10% of hierarchy to OUTPUT_DIR/frameXXXXX.png):
```shell
./segment_converter --input=INPUT.mp4.pb --bitmap_color=0.1 --output_dir=OUTPUT_DIR
```

- segment_renderer:

Renders segmentation to video and images with support to parse project JSON files.
Example (to render annotated result to images):
```shell
./segment_renderer --input_file=DOWNLOAD_FROM_WEBSITE.pb --output_image_dir=SOME_DIR --json_file=DOWNLOAD_FROM_WEBSITE.json --logging
```
Example (to render segmentation result at 10% of hierarchy to video):
```shell
./segment_renderer --input_file=INPUT.pb --output_video_file=OUTPUT_VIDEO --render_level=0.1
```

- segment_viewer: Interactive viewer for segmentation pb files
```shell
./segment_viewer --input=INPUT.pb
```

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
