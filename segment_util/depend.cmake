find_package(Boost REQUIRED)
find_package(GLog REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(Protobuf REQUIRED)

set(DEPENDENT_PACKAGES base)

set(DEPENDENT_INCLUDES ${PROTOBUF_INCLUDE_DIRS}
                       ${OpenCV_INCLUDE_DIRS}
		       ${Boost_INCLUDE_DIR}
                       ${GLOG_INCLUDE_DIR}
                       )

set(DEPENDENT_LIBRARIES ${PROTOBUF_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${GLOG_LIBRARIES}
                        )

set(CREATED_PACKAGES segment_util)
