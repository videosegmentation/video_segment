# - Try to find OpenCV library installation
# See http://sourceforge.net/projects/opencvlibrary/
#
# The following variables are optionally searched for defaults
#  OpenCV_ROOT_DIR:            Base directory of OpenCv tree to use.
#  OpenCV_FIND_REQUIRED_COMPONENTS : FIND_PACKAGE(OpenCV COMPONENTS ..)
#    compatible interface. typically  CV CXCORE CVAUX HIGHGUI CVCAM .. etc.
#
# The following are set after configuration is done:
#  OpenCV_FOUND
#  OpenCV_INCLUDE_DIR
#  OpenCV_LIBRARIES
#  OpenCV_LINK_DIRECTORIES
#
# 2004/05 Jan Woetzel, Friso, Daniel Grest
# 2006/01 complete rewrite by Jan Woetzel
# 1006/09 2nd rewrite introducing ROOT_DIR and PATH_SUFFIXES
#   to handle multiple installed versions gracefully by Jan Woetzel
#
# tested with:
# -OpenCV 0.97 (beta5a):  MSVS 7.1, gcc 3.3, gcc 4.1
# -OpenCV 0.99 (1.0rc1):  MSVS 7.1
#
# www.mip.informatik.uni-kiel.de/~jw
#
# ---------------------
#
# $Id: FindOpenCV.cmake 8 2009-01-04 21:13:48Z adeguet1 $
# ERC-CISST version:
#
# - Removed deprecated code starting with cap. OPENCV
# - Removed debugging code and messages
# - Removed path and options specifics to previous authors setups
#
# This file should be removed when CMake will provide an equivalent

# Adapted for new opencv 2.2 modules.

# required cv components with header and library if COMPONENTS unspecified
IF(NOT OpenCV_FIND_REQUIRED_COMPONENTS)
  # default
  SET(OpenCV_FIND_REQUIRED_COMPONENTS core imgproc video highgui calib3d features2d)
#  IF(WIN32)
#    LIST(APPEND OpenCV_FIND_REQUIRED_COMPONENTS  CVCAM ) # WIN32 only actually
#  ENDIF(WIN32)
ENDIF(NOT OpenCV_FIND_REQUIRED_COMPONENTS)


# typical root dirs of installations, exactly one of them is used
SET(OpenCV_POSSIBLE_ROOT_DIRS
  "${OpenCV_ROOT_DIR}"
  "${OpenCV_ROOT_DIR}/build"
  "$ENV{OpenCV_ROOT_DIR}"
  "$ENV{OpenCV_ROOT_DIR}/build"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Intel(R) Open Source Computer Vision Library_is1;Inno Setup: App Path]"
  "$ENV{ProgramFiles}/OpenCV"
  "/usr/local"
  "/usr"
	"/opt/local"
  )

SET(COMPONENTS calib3d core contrib features2d flann gpu highgui imgproc legacy ml objdetect video)

#
# select exactly ONE OpenCV base directory/tree
# to avoid mixing different version headers and libs
#
FIND_PATH(OpenCV_ROOT_DIR
  NAMES
  include/opencv2/core/core_c.h
  PATHS ${OpenCV_POSSIBLE_ROOT_DIRS})

# header include dir suffixes appended to OpenCV_ROOT_DIR
SET(OpenCV_INCDIR_SUFFIXES
  opencv2
  include/opencv2
  include
  )

# library linkdir suffixes appended to OpenCV_ROOT_DIR
SET(OpenCV_LIBDIR_SUFFIXES
  lib
  OpenCV/lib
  )

FOREACH(COMP ${COMPONENTS})
  # Developers = people that annoy you ...
  SET (COMP_HEADER opencv2/${COMP}/${COMP}.hpp)
  IF (${COMP} STREQUAL "video")
    SET(COMP_HEADER "opencv2/video/tracking.hpp")
  ENDIF (${COMP} STREQUAL "video")

  FIND_PATH(OPENCV_${COMP}_INC_DIR
            NAMES ${COMP_HEADER}
            PATHS ${OpenCV_ROOT_DIR}
            PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})
  MARK_AS_ADVANCED(${OPENCV_${COMP}_INC_DIR})

  FIND_LIBRARY(OPENCV_${COMP}_LIB
               NAMES opencv_${COMP} opencv_${COMP}231 opencv_${COMP}22 opencv_${COMP}23
               PATHS ${OpenCV_ROOT_DIR}
               PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})
  MARK_AS_ADVANCED(${OPENCV_${COMP}_LIB})
ENDFOREACH(COMP ${COMPONENTS})

# Logic selecting required libs and headers
SET(OpenCV_FOUND ON)
FOREACH(NAME ${OpenCV_FIND_REQUIRED_COMPONENTS})
  # only good if header and library both found
  IF(OPENCV_${NAME}_INC_DIR AND OPENCV_${NAME}_LIB)
    LIST(APPEND OpenCV_INCLUDE_DIRS "${OPENCV_${NAME}_INC_DIR}")
    LIST(APPEND OpenCV_LIBRARIES    "${OPENCV_${NAME}_LIB}")
  ELSE(OPENCV_${NAME}_INC_DIR AND OPENCV_${NAME}_LIB)
    message("Could not find ${OPENCV_${NAME}_INC_DIR} or ${OPENCV_${NAME}_LIB}")
    SET(OpenCV_FOUND OFF)
  ENDIF(OPENCV_${NAME}_INC_DIR AND OPENCV_${NAME}_LIB)
ENDFOREACH(NAME)

# display help message
IF(NOT OpenCV_FOUND)
  # make FIND_PACKAGE friendly
  IF(OpenCV_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR
            "OpenCV required but some headers or libs not found. Please specify it's location with OpenCV_ROOT_DIR env. variable.")
  ELSE(OpenCV_FIND_REQUIRED)
    MESSAGE(STATUS
            "ERROR: OpenCV was not found.")
  ENDIF(OpenCV_FIND_REQUIRED)
ELSE(NOT OpenCV_FOUND)
  MESSAGE(STATUS
          "OpenCV found at ${OpenCV_ROOT_DIR}")
ENDIF(NOT OpenCV_FOUND)


#
# $Log: FindOpenCV.cmake,v $
# Revision 1.2  2008/11/03 16:40:34  vagvoba
# cmake: Proper cvcam and highgui directories added to FindOpenCV.cmake
#
# Revision 1.1  2007/07/12 00:22:25  anton
# cmake utilities: Added FindOpenCV.cmake based on example found all over the
# web.  Will have to be removed when CMake provides one.
#
#

