find_package(Qt4 COMPONENTS QtGui QtCore REQUIRED)
include(${QT_USE_FILE})

find_package(Boost COMPONENTS system thread REQUIRED)

set(DEPENDENT_PACKAGES base
                       video_framework
                       segment_util
                       )

set(DEPENDENT_INCLUDES ${Boost_INCLUDE_DIR}
                       )

set(DEPENDENT_LIBRARIES ${QT_LIBRARIES}
                        ${Boost_LIBRARIES}
                        )

set(DEPENDENT_LINK_DIRECTORIES ${Boost_LIBRARY_DIRS}
                               )

set(CREATED_PACKAGES video_display_qt)
