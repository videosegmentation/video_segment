cmake_minimum_required(VERSION 2.6)

project(segment_converter)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

add_executable(segment_converter converter.cpp)

apply_dependencies(segment_converter)
