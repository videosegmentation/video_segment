# The MIT License
#
# Copyright (c) 2011 daniperez
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
find_package(PkgConfig)

pkg_check_modules(_GLOG QUIET libglog)

find_path(GLOG_INCLUDE_DIR glog/logging.h
          HINTS ${_GLOG_INCLUDEDIR} ${_GLOG_INCLUDE_DIRS} )

find_library(GLOG_LIBRARY NAMES glog libglog
             HINTS ${_GLOG_LIBDIR} ${_GLOG_LIBRARY_DIRS} )

set(GLOG_LIBRARIES ${GLOG_LIBRARY} )
set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Glog  DEFAULT_MSG
                                  GLOG_LIBRARY GLOG_INCLUDE_DIR)

mark_as_advanced(GLOG_INCLUDE_DIR GLOG_LIBRARY)
