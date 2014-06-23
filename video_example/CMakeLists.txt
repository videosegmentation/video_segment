cmake_minimum_required(VERSION 2.6)

project(video_example)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

add_executable(video_example video_example.cpp)

apply_dependencies(video_example)
