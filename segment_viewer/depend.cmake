find_package(OpenCV2 REQUIRED)
find_package(GFlags REQUIRED)

set(DEPENDENT_INCLUDES ${OpenCV_INCLUDE_DIRS}
                       ${GFLAGS_INCLUDE_DIRS}
                       )

set(DEPENDENT_LIBRARIES ${OpenCV_LIBRARIES}
                        ${GFLAGS_LIBRARIES}
                        )
set(DEPENDENT_LINK_DIRECTORIES ${OpenCV_LINK_DIRECTORIES}
                               )

set(DEPENDENT_PACKAGES base
                       segment_util)
