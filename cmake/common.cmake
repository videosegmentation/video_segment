# Copyright (c) 2010-2014, The Video Segmentation Project
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the The Video Segmentation Project nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ---

# BUILD system for /video_segment source tree.

cmake_policy(SET CMP0015 OLD)
cmake_policy(SET CMP0011 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Common flags for all projects.
if (UNIX)
  set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")
  set(CMAKE_BUILD_TYPE Release)
  # set(CMAKE_BUILD_TYPE Debug)

  # Parallelization.
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif (NOT APPLE)

  # C++11 support.
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else (APPLE)
    # Linux adaptations.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif (APPLE)
endif (UNIX)

if (APPLE)
  set(CMAKE_HOST_SYSTEM_PROCESSOR "x86_64")
endif (APPLE)

# Profiling
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lprofiler -L/opt/local/lib")

# Checks for each cpp file if header exists and adds it to HEADERS
function(headers_from_sources_cpp HEADERS SOURCES)
set(HEADERS)
  foreach(SOURCE_FILE ${SOURCES})
    string(REPLACE ".cpp" ".h" HEADER_FILE ${SOURCE_FILE})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_FILE}")
           list(APPEND HEADERS ${HEADER_FILE})
    endif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_FILE}")
  endforeach(SOURCE_FILE)

set(HEADERS "${HEADERS}" PARENT_SCOPE)

endfunction(headers_from_sources_cpp)

# Reads the depend.cmake file in the current source folder and processes
# dependent libraries and packages in the source tree.
function(apply_dependencies TARGET)

  # Use property APPLY_DEPENDENCIES_CALLED to use add_subdirectory
  # only on the main CMakeLists.txt
  # This is set by the first call of apply_dependencies.
  get_property(CALL_PROPERTY_DEFINED
               GLOBAL
               PROPERTY APPLY_DEPENDENCIES_CALLED
               SET)

  if (${CALL_PROPERTY_DEFINED})
    # Only add include and dependency
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/depend.cmake")
     set(DEPENDENT_PACKAGES)
     include("${CMAKE_CURRENT_SOURCE_DIR}/depend.cmake")

     if(DEPENDENT_PACKAGES)
        foreach(PACKAGE "${DEPENDENT_PACKAGES}")
         # Add package directory root to include path.
         include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

         # Add package binary path to include path.
         # Temporary files, like protobuffer or qt's moc are placed here.
         include_directories("${CMAKE_BINARY_DIR}")
         add_dependencies(${TARGET} ${PACKAGE})
        endforeach(PACKAGE)
      endif(DEPENDENT_PACKAGES)

      include_directories(${DEPENDENT_INCLUDES})
   endif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/depend.cmake")
  else (${CALL_PROPERTY_DEFINED})
    # First call of apply_dependencies.
    set_property(GLOBAL PROPERTY APPLY_DEPENDENCIES_CALLED TRUE)
		set_property(GLOBAL PROPERTY ROOT_PROJECT_NAME ${TARGET})

    # Get list of dependencies.
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/depend.cmake")
      set(DEPENDENT_PACKAGES)
      include("${CMAKE_CURRENT_SOURCE_DIR}/depend.cmake")

      # Each package is unique but can have duplicate dependent libraries.
      set(MY_DEPENDENT_LIBRARIES "${DEPENDENT_LIBRARIES}")
      set(MY_DEPENDENT_LINK_DIRECTORIES "${DEPENDENT_LINK_DIRECTORIES}")
      set(MY_DEPENDENT_INCLUDES "${DEPENDENT_INCLUDES}" )

      if(DEPENDENT_PACKAGES)
        # Recursively get all dependent packages. Depth first.
        set(PACKAGES_TO_PROCESS "${DEPENDENT_PACKAGES}")
        recursive_dependency_retrieve(PACKAGES_TO_PROCESS "${PACKAGES_TO_PROCESS}")
        # Remove duplicates, keep first one. Does not alter the depth first property.
        list(REMOVE_DUPLICATES PACKAGES_TO_PROCESS)
        message("Adding the following dependent packages: ${PACKAGES_TO_PROCESS}")

        # The dependent packages binary dirs and created libraries.
        # We need to reverse the order before adding them.
        set(PACKAGE_LINK_DIRS)
        set(PACKAGE_LINK_LIBS)

        # Add main source path.
        include_directories(..)
        # For temporary generated files like moc or protoc.
        include_directories("${CMAKE_BINARY_DIR}")
        # Root directory for protos generated in main project.
        include_directories("${CMAKE_BINARY_DIR}/..")

        foreach(PACKAGE ${PACKAGES_TO_PROCESS})
          message("Adding dependency  ../${PACKAGE} ${PACKAGE}")
          add_subdirectory(../${PACKAGE} ${PACKAGE})
          add_dependencies(${TARGET} ${PACKAGE})

          # Add linker information
          set(DEPENDENT_PACKAGES)
          set(DEPENDENT_INCLUDES)
          set(DEPENDENT_LIBRARIES)
          set(DEPENDENT_LINK_DIRECTORIES)
          set(CREATED_PACKAGES)

          include("${CMAKE_CURRENT_SOURCE_DIR}/../${PACKAGE}/depend.cmake")

          if (CREATED_PACKAGES)
            list(APPEND PACKAGE_LINK_DIRS ${PACKAGE})
            list(APPEND PACKAGE_LINK_LIBS ${CREATED_PACKAGES})
          endif(CREATED_PACKAGES)

          if (DEPENDENT_LIBRARIES)
            list(APPEND MY_DEPENDENT_LIBRARIES "${DEPENDENT_LIBRARIES}")
            list(APPEND MY_DEPENDENT_LINK_DIRECTORIES "${DEPENDENT_LINK_DIRECTORIES}")
            list(APPEND MY_DEPENDENT_INCLUDES "${DEPENDENT_INCLUDES}")
          endif(DEPENDENT_LIBRARIES)
        endforeach(PACKAGE)

        # Add package libraries and binary directories.
        if (PACKAGE_LINK_DIRS)
          # Reverse to adhere linking order.
          list(REVERSE PACKAGE_LINK_DIRS)
          list(REVERSE PACKAGE_LINK_LIBS)

          foreach(LINK_DIR ${PACKAGE_LINK_DIRS})
            link_directories(${LINK_DIR})
          endforeach(LINK_DIR)

          foreach(LINK_LIB ${PACKAGE_LINK_LIBS})
            target_link_libraries(${TARGET} ${LINK_LIB})
          endforeach(LINK_LIB)

        endif(PACKAGE_LINK_DIRS)
      endif(DEPENDENT_PACKAGES)

      if (MY_DEPENDENT_LINK_DIRECTORIES)
        list(REMOVE_DUPLICATES MY_DEPENDENT_LINK_DIRECTORIES)
        link_directories(${MY_DEPENDENT_LINK_DIRECTORIES})
      endif(MY_DEPENDENT_LINK_DIRECTORIES)


      if (MY_DEPENDENT_LIBRARIES)
        # Remove duplicates while handling debug; and optimized;
        SET(ESCAPED_ITEMS)
        while(0 LESS 1)
          # Collapse debug;lib into one item __debug__lib
          list(FIND MY_DEPENDENT_LIBRARIES debug DEBUG_POS)
          if (NOT ${DEBUG_POS} LESS 0)
            list(REMOVE_AT MY_DEPENDENT_LIBRARIES ${DEBUG_POS})
            list(GET MY_DEPENDENT_LIBRARIES ${DEBUG_POS} DEBUG_LIB)
            list(REMOVE_AT MY_DEPENDENT_LIBRARIES ${DEBUG_POS})
            list(LENGTH MY_DEPENDENT_LIBRARIES LIST_LEN)
            if(${DEBUG_POS} LESS LIST_LEN)
              list(INSERT MY_DEPENDENT_LIBRARIES ${DEBUG_POS} "__debug__${DEBUG_LIB}")
            else(${DEBUG_POS} LESS LIST_LEN)
              list(APPEND MY_DEPENDENT_LIBRARIES "__debug__${DEBUG_LIB}")
            endif(${DEBUG_POS} LESS LIST_LEN)
            list(APPEND ESCAPED_ITEMS "__debug__${DEBUG_LIB}")
          endif (NOT ${DEBUG_POS} LESS 0)

          list(FIND MY_DEPENDENT_LIBRARIES optimized RELEASE_POS)
          if (NOT ${RELEASE_POS} LESS 0)
            list(REMOVE_AT MY_DEPENDENT_LIBRARIES ${RELEASE_POS})
            list(GET MY_DEPENDENT_LIBRARIES ${RELEASE_POS} RELEASE_LIB)
            list(REMOVE_AT MY_DEPENDENT_LIBRARIES ${RELEASE_POS})
            if(${RELEASE_POS} LESS LIST_LEN)
              list(INSERT MY_DEPENDENT_LIBRARIES ${RELEASE_POS}
                   "__optimized__${RELEASE_LIB}")
            else(${RELEASE_POS} LESS LIST_LEN)
              list(APPEND MY_DEPENDENT_LIBRARIES
                   "__optimized__${RELEASE_LIB}")
            endif(${RELEASE_POS} LESS LIST_LEN)
            list(APPEND ESCAPED_ITEMS "__optimized__${RELEASE_LIB}")
          endif (NOT ${RELEASE_POS} LESS 0)

          if(${DEBUG_POS} EQUAL -1 AND ${RELEASE_POS} EQUAL -1)
            break()
          endif(${DEBUG_POS} EQUAL -1 AND ${RELEASE_POS} EQUAL -1)
        endwhile(0 LESS 1)

        list(REMOVE_DUPLICATES MY_DEPENDENT_LIBRARIES)

        #Unescape.
        foreach(ITEM ${ESCAPED_ITEMS})
          list(FIND MY_DEPENDENT_LIBRARIES ${ITEM} ITEM_POS)
          if (NOT ${ITEM_POS} LESS 0)
            list(GET MY_DEPENDENT_LIBRARIES ${ITEM_POS} ITEM_LIB)
            list(REMOVE_AT MY_DEPENDENT_LIBRARIES ${ITEM_POS})
            string(REPLACE __optimized__ "" RELEASE_STRIPPED_LIB ${ITEM_LIB})
            string(COMPARE EQUAL ${RELEASE_STRIPPED_LIB} ${ITEM_LIB} STRIP_RESULT)
            if (NOT STRIP_RESULT)
              list(LENGTH MY_DEPENDENT_LIBRARIES LIST_LEN)
              if(${ITEM_POS} LESS LIST_LEN)
                list(INSERT MY_DEPENDENT_LIBRARIES ${ITEM_POS}
                     "optimized;${RELEASE_STRIPPED_LIB}")
              else(${ITEM_POS} LESS LIST_LEN)
                list(APPEND MY_DEPENDENT_LIBRARIES "optimized;${RELEASE_STRIPPED_LIB}")
              endif(${ITEM_POS} LESS LIST_LEN)
            endif (NOT STRIP_RESULT)

            string(REPLACE __debug__ "" DEBUG_STRIPPED_LIB ${ITEM_LIB})
            string(COMPARE EQUAL ${DEBUG_STRIPPED_LIB} ${ITEM_LIB} STRIP_RESULT)
            if (NOT STRIP_RESULT)
              list(LENGTH MY_DEPENDENT_LIBRARIES LIST_LEN)
              if(${ITEM_POS} LESS LIST_LEN)
                list(INSERT MY_DEPENDENT_LIBRARIES ${ITEM_POS} "debug;${DEBUG_STRIPPED_LIB}")
              else(${ITEM_POS} LESS LIST_LEN)
                list(APPEND MY_DEPENDENT_LIBRARIES "debug;${DEBUG_STRIPPED_LIB}")
              endif(${ITEM_POS} LESS LIST_LEN)
            endif (NOT STRIP_RESULT)
          endif (NOT ${ITEM_POS} LESS 0)
        endforeach(ITEM ${ESCAPED_ITEMS})

        target_link_libraries(${TARGET} ${MY_DEPENDENT_LIBRARIES})
      endif(MY_DEPENDENT_LIBRARIES)

      if (MY_DEPENDENT_INCLUDES)
        list(REMOVE_DUPLICATES MY_DEPENDENT_INCLUDES)
        include_directories(${MY_DEPENDENT_INCLUDES})
      endif(MY_DEPENDENT_INCLUDES)
    endif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/depend.cmake")
  endif(${CALL_PROPERTY_DEFINED})
endfunction(apply_dependencies)

# Finds recursively dependencies of packages in the source tree.
# Implementation function called by apply_dependencies.
function(recursive_dependency_retrieve ADD_LIST PACKAGES)
  set(LOCAL_LIST ${PACKAGES})
  foreach(item ${PACKAGES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${item}/depend.cmake")
      SET(DEPENDENT_PACKAGES)
      include ( "${CMAKE_CURRENT_SOURCE_DIR}/../${item}/depend.cmake")
      if(DEPENDENT_PACKAGES)
        SET(MY_PACKAGES "${DEPENDENT_PACKAGES}")
        SET(MY_LIST)
        recursive_dependency_retrieve(MY_LIST "${MY_PACKAGES}")
        # Build order is important. This is like depth first search.
        list(INSERT LOCAL_LIST 0 "${MY_LIST}")
      endif(DEPENDENT_PACKAGES)
    endif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${item}/depend.cmake")
  endforeach(item)
  set(${ADD_LIST} "${LOCAL_LIST}" PARENT_SCOPE)
endfunction(recursive_dependency_retrieve)
