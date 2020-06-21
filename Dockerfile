FROM ubuntu:trusty-20191217
WORKDIR /usr/src

ARG DEBIAN_FRONTEND=noninteractive

# update apt
RUN apt-get update -y && apt-get upgrade -y

# dependencies
RUN apt-get install -y \
	    build-essential \
	    yasm \
	    cmake \
	    libboost-dev \
	    libboost-filesystem-dev \
	    libboost-system-dev \
	    libboost-regex-dev \
	    libboost-thread-dev \
	    libprotoc-dev \
	    libprotobuf-dev \
	    protobuf-compiler \
	    libavcodec-dev \
	    libavformat-dev \
	    libswscale-dev \
	    libgtk2.0-dev \
	    libgflags-dev \
	    libjsoncpp-dev \
	    libboost-program-options-dev \
	    libgoogle-glog-dev

# other tools
RUN apt-get install -y \
	    git-core \
	    unzip \
	    wget

# download stuff
RUN \
	wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip && \
	unzip opencv-2.4.9.zip && \
	wget http://www.ffmpeg.org/releases/ffmpeg-2.2.3.tar.gz && \
	tar -zxvf ffmpeg-2.2.3.tar.gz

# build opencv
RUN \
	cd opencv-2.4.9 && \
	mkdir release && \
	cd release && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
	make install -j4

#build ffmpeg
RUN \
	cd ffmpeg-2.2.3 && \
	./configure --enable-gpl && \
	make install -j4

# download video_segment
RUN git clone https://github.com/videosegmentation/video_segment

# build video_segment
RUN \
	mkdir -p seg_tree_sample && \
	cd seg_tree_sample && \
	cmake /usr/src/video_segment/seg_tree_sample && \
	make -j4

RUN \
	mkdir -p segment_converter && \
	cd segment_converter && \
	cmake /usr/src/video_segment/segment_converter && \
	make -j4

RUN \
	mkdir -p segment_renderer && \
	cd segment_renderer && \
	cmake /usr/src/video_segment/segment_renderer && \
	make -j4

RUN \
	mkdir -p segment_viewer && \
	cd segment_viewer && \
	cmake /usr/src/video_segment/segment_viewer && \
	make -j4

CMD [ "bash" ]
