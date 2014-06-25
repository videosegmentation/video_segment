find_package(GFlags REQUIRED)
find_package(JsonCpp REQUIRED)

set(DEPENDENT_INCLUDES ${GFLAGS_INCLUDE_DIRS}
                       ${JSONCPP_INCLUDE_DIR}
                       )

set(DEPENDENT_LIBRARIES ${GFLAGS_LIBRARIES}
                        ${JSONCPP_LIBRARY}
                        )

set(DEPENDENT_PACKAGES base
                       video_framework
                       segment_util
                       segmentation)
