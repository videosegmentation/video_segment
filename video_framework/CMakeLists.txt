cmake_minimum_required(VERSION 2.6)

project(video_framework)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

set(SOURCES video_unit.cpp
            video_display_unit.cpp
            video_reader_unit.cpp
            video_writer_unit.cpp
            conversion_units.cpp
            flow_reader.cpp
            video_capture_unit.cpp
            video_pipeline.cpp
         )

headers_from_sources_cpp(HEADERS "${SOURCES}")
set(SOURCES "${SOURCES}" "${HEADERS}" )

add_library(video_framework ${SOURCES})

apply_dependencies(video_framework)
