cmake_minimum_required(VERSION 2.6)

project(segment_util)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

set(SOURCES segmentation_io.cpp
            segmentation_util.cpp
            segmentation_render.cpp
            segmentation_boundary.cpp
            )

headers_from_sources_cpp(HEADERS "${SOURCES}")
set(SOURCES "${SOURCES}" "${HEADERS}")

# Custom build steps.
find_package(Protobuf REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HDR segmentation.proto)

add_library(segment_util ${SOURCES} ${PROTO_SRC} ${PROTO_HDR})

apply_dependencies(segment_util)
