cmake_minimum_required(VERSION 2.6)

project(segmentation)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

set(SOURCES boundary.cpp
            histograms.cpp
            dense_segmentation.cpp
            dense_segmentation_graph.cpp
            dense_seg_graph_interface.cpp
            pixel_distance.cpp
            region_segmentation_graph.cpp
            region_segmentation.cpp
            region_descriptor.cpp
            segmentation.cpp
            segmentation_graph.cpp
            segmentation_common.cpp
            segmentation_unit.cpp
            )

headers_from_sources_cpp(HEADERS "${SOURCES}")
set(SOURCES "${SOURCES}" "${HEADERS}")

add_library(segmentation ${SOURCES})

apply_dependencies(segmentation)
