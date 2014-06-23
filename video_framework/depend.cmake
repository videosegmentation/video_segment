find_package(Boost COMPONENTS filesystem system program_options thread regex REQUIRED)
find_package(FFMPEG REQUIRED)
find_package(GLog REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(ZLIB REQUIRED)

set(DEPENDENT_PACKAGES base imagefilter)

set(DEPENDENT_INCLUDES ${OpenCV_INCLUDE_DIRS}
                       ${Boost_INCLUDE_DIR}
                       ${FFMPEG_INCLUDE_DIR}
                       ${GLOG_INCLUDE_DIR}
                       )

set(DEPENDENT_LIBRARIES ${OpenCV_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${FFMPEG_LIBRARIES}
                        ${GLOG_LIBRARIES}
                        ${ZLIB_LIBRARIES})

set(CREATED_PACKAGES video_framework)
