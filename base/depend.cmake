find_package(Boost REQUIRED)
find_package(GLog REQUIRED)

set(DEPENDENT_PACKAGES)
set(DEPENDENT_INCLUDES ${Boost_INCLUDE_DIR}
                       ${GLOG_INCLUDE_DIR}
                       )

set(DEPENDENT_LIBRARIES ${Boost_LIBRARIES}
                        ${GLOG_LIBRARIES}
                        )
set(DEPENDENT_LINK_DIRECTORIES ${Boost_LIBRARY_DIRS}
                               )
set(CREATED_PACKAGES base)
