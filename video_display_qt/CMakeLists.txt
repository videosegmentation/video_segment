cmake_minimum_required(VERSION 2.6)

project(video_display_qt)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

find_package(Qt4 COMPONENTS QtGui QtCore REQUIRED)

QT4_WRAP_CPP(MOC_SOURCES main_window.h)

set(SOURCES video_display_qt_unit.cpp
	          main_window.cpp
            )

headers_from_sources_cpp(HEADERS "${SOURCES}")
set(SOURCES "${SOURCES}"
            "${HEADERS}"
             ${MOC_SOURCES}
)

add_library(video_display_qt ${SOURCES})

apply_dependencies(video_display_qt)
