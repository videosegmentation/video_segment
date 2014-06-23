cmake_minimum_required(VERSION 2.6)

project(base)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include("${CMAKE_MODULE_PATH}/common.cmake")
include("${CMAKE_SOURCE_DIR}/depend.cmake")

set(SOURCES base.cpp
            base_impl.cpp)
headers_from_sources_cpp(HEADERS "${SOURCES}")

set(SOURCES "${SOURCES}" "${HEADERS}")

add_library(base ${SOURCES})

apply_dependencies(base)
