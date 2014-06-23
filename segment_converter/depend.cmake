
find_package(GFlags REQUIRED)

set(DEPENDENT_INCLUDES ${GFLAGS_INCLUDE_DIRS}
                       )

set(DEPENDENT_LIBRARIES ${GFLAGS_LIBRARIES}
                        )

set(DEPENDENT_LINK_DIRECTORIES
                              )

set(DEPENDENT_PACKAGES base
                       segment_util)
