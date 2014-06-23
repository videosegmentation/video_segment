find_package(Boost REQUIRED)
find_package(GLog REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(TBB REQUIRED)

set(DEPENDENT_PACKAGES base)

set(DEPENDENT_INCLUDES ${OpenCV_INCLUDE_DIRS}
                       ${Boost_INCLUDE_DIR}
                       ${GLOG_INCLUDE_DIR}
                       ${TBB_INCLUDE_DIRS}
                       )

set(DEPENDENT_LIBRARIES ${OpenCV_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${TBB_LIBRARIES}
                        ${GLOG_LIBRARIES}
                        )

set(CREATED_PACKAGES imagefilter)
