find_package(GFlags REQUIRED)

set(DEPENDENT_INCLUDES ${GFLAGS_INCLUDE_DIRS}
                       )

set(DEPENDENT_LIBRARIES ${GFLAGS_LIBRARIES}
                        )

set(DEPENDENT_PACKAGES base
                       video_framework
                       segmentation
                       )
if(NO_X_SUPPORT)
  # Nothing to do here.
else(NO_X_SUPPORT)
  if(WITH_QT)
    set(DEPENDENT_PACKAGES ${DEPENDENT_PACKAGES}
                           video_display_qt
                           )
  endif(WITH_QT)
endif(NO_X_SUPPORT)
