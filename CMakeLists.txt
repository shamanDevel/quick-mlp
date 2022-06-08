# CMakeLists.txt -- Build system for the cuda-kernel-loader (qmlp) modules
#
# Copyright (c) 2022 Sebastian Weiss <sebastian13.weiss@tum.de>
#
# All rights reserved. Use of this source code is governed by a
# MIT-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.4)

# The `cmake_minimum_required(VERSION 3.4...3.18)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()
cmake_policy(SET CMP0127 NEW)

# Extract project version from source
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/qmlp/common.h"
     qmlp_version_defines REGEX "#define QUICKMLP_VERSION_(MAJOR|MINOR|PATCH) ")

foreach(ver ${qmlp_version_defines})
  if(ver MATCHES [[#define QUICKMLP_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$]])
    set(QUICKMLP_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endif()
endforeach()

if(QUICKMLP_VERSION_PATCH MATCHES [[\.([a-zA-Z0-9]+)$]])
  set(qmlp_VERSION_TYPE "${CMAKE_MATCH_1}")
endif()
string(REGEX MATCH "^[0-9]+" QUICKMLP_VERSION_PATCH "${QUICKMLP_VERSION_PATCH}")

project(
  qmlp
  LANGUAGES CXX
  VERSION "${QUICKMLP_VERSION_MAJOR}.${QUICKMLP_VERSION_MINOR}.${QUICKMLP_VERSION_PATCH}")

# Standard includes
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

if(NOT qmlp_FIND_QUIETLY)
  message(STATUS "qmlp v${qmlp_VERSION} ${qmlp_VERSION_TYPE}")
endif()

# Avoid infinite recursion if tests include this as a subdirectory
if(DEFINED QUICKMLP_MASTER_PROJECT)
  set(QUICKMLP_TEST OFF)
endif()

# Check if qmlp is being used directly or via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND NOT DEFINED QUICKMLP_MASTER_PROJECT)
  ### Warn if not an out-of-source builds
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    set(lines
        "You are building in-place. If that is not what you intended to "
        "do, you can clean the source directory with:\n"
        "rm -r CMakeCache.txt CMakeFiles/ cmake_uninstall.cmake qmlpConfig.cmake "
        "qmlpConfigVersion.cmake tests/CMakeFiles/\n")
    message(AUTHOR_WARNING ${lines})
  endif()

  set(QUICKMLP_MASTER_PROJECT ON)

  if(OSX AND CMAKE_VERSION VERSION_LESS 3.7)
    # Bug in macOS CMake < 3.7 is unable to download catch
    message(WARNING "CMAKE 3.7+ needed on macOS to download catch, and newer HIGHLY recommended")
  elseif(WINDOWS AND CMAKE_VERSION VERSION_LESS 3.8)
    # Only tested with 3.8+ in CI.
    message(WARNING "CMAKE 3.8+ tested on Windows, previous versions untested")
  endif()

  message(STATUS "CMake ${CMAKE_VERSION}")

  if(CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()

  set(qmlp_system "")
else()
  set(QUICKMLP_MASTER_PROJECT OFF)
  set(qmlp_system SYSTEM)
endif()

# Options
option(QUICKMLP_TEST "Build qmlp test suite?" ${QUICKMLP_MASTER_PROJECT})
option(QUICKMLP_SHARED "Build qmlp as a shared library?" OFF)

# CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Third-party
add_subdirectory(third-party/cuda-kernel-loader)
add_subdirectory(third-party/json)

# Files
set(QUICKMLP_HEADERS
    include/qmlp/common.h
    include/qmlp/activation.h
    include/qmlp/iencoding.h
    include/qmlp/encoding_identity.h
    include/qmlp/fused_network.h
    include/qmlp/qmlp.h
    include/qmlp/errors.h
    include/qmlp/tensor.h
    )
    
set(QUICKMLP_RESOURCES
    include/qmlp/builtin-activations.json

    include/qmlp/kernels/loops.cuh
    include/qmlp/kernels/tensor.cuh
    include/qmlp/kernels/encoding_identity.cuh
    include/qmlp/kernels/layer.cuh
    include/qmlp/kernels/network_forward.cuh
    include/qmlp/kernels/network_backward.cuh
)

set(QUICKMLP_SOURCES
    src/qmlp.cpp
    src/activation.cpp
    src/tensor.cpp
    src/encoding_identity.cpp
    src/iencoding.cpp
    src/fused_network.cpp
    )

# Compare with grep and warn if mismatched
if(QUICKMLP_MASTER_PROJECT AND NOT CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB_RECURSE _qmlp_header_check
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS "include/qmlp/*")
  set(_qmlp_here_only ${QUICKMLP_HEADERS} ${QUICKMLP_RESOURCES})
  set(_qmlp_disk_only ${_qmlp_header_check})
  list(REMOVE_ITEM _qmlp_here_only ${_qmlp_header_check})
  list(REMOVE_ITEM _qmlp_disk_only ${QUICKMLP_HEADERS} ${QUICKMLP_RESOURCES})
  if(_qmlp_here_only)
    message(AUTHOR_WARNING "QUICKMLP_HEADERS+QUICKMLP_RESOURCES has extra files:" ${_qmlp_here_only})
  endif()
  if(_qmlp_disk_only)
    message(AUTHOR_WARNING "QUICKMLP_HEADERS+QUICKMLP_RESOURCES is missing files:" ${_qmlp_disk_only})
  endif()

  file(
    GLOB_RECURSE _qmlp_src_check
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS "src/*.cpp")
  set(_qmlp_here_only ${QUICKMLP_SOURCES})
  set(_qmlp_disk_only ${_qmlp_src_check})
  list(REMOVE_ITEM _qmlp_here_only ${_qmlp_src_check})
  list(REMOVE_ITEM _qmlp_disk_only ${QUICKMLP_SOURCES})
  if(_qmlp_here_only)
    message(AUTHOR_WARNING "QUICKMLP_SOURCES has extra files:" ${_qmlp_here_only})
  endif()
  if(_qmlp_disk_only)
    message(AUTHOR_WARNING "QUICKMLP_SOURCES is missing files:" ${_qmlp_disk_only})
  endif()
endif()

# CMake 3.12 added list(TRANSFORM <list> PREPEND
# But we can't use it yet
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/" QUICKMLP_HEADERS
               "${QUICKMLP_HEADERS}")
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/" QUICKMLP_RESOURCES
               "${QUICKMLP_RESOURCES}")
string(REPLACE "src/" "${CMAKE_CURRENT_SOURCE_DIR}/src/" QUICKMLP_SOURCES
               "${QUICKMLP_SOURCES}")

# Cache variable so this can be used in parent projects
set(qmlp_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/include"
    CACHE INTERNAL "Directory where qmlp headers are located")

# Backward compatible variable for add_subdirectory mode
if(NOT QUICKMLP_MASTER_PROJECT)
  set(QUICKMLP_INCLUDE_DIR
      "${qmlp_INCLUDE_DIR}"
      CACHE INTERNAL "")
endif()

# Note: when creating targets, you cannot use if statements at configure time -
# you need generator expressions, because those will be placed in the target file.
# You can also place ifs *in* the Config.in, but not here.

# This section builds targets
# Non-IMPORT targets cannot be defined twice
if(NOT TARGET qmlp)
  # Build the headers-only target (no Python included):
  # (long name used here to keep this from clashing in subdirectory mode)
  if (QUICKMLP_SHARED)
    add_library(qmlp SHARED
        ${QUICKMLP_HEADERS} ${QUICKMLP_RESOURCES} ${QUICKMLP_SOURCES})
  else()
    add_library(qmlp STATIC
        ${QUICKMLP_HEADERS} ${QUICKMLP_RESOURCES} ${QUICKMLP_SOURCES})
  endif()
  add_library(qmlp::qmlp ALIAS qmlp) # to match exported target

  set_target_properties(qmlp PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )

  target_include_directories(
    qmlp ${qmlp_system} PUBLIC ${qmlp_INCLUDE_DIR})
  target_include_directories(qmlp PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR} third-party/tinyformat)

  # CUDA Stuff
  target_link_libraries(qmlp PUBLIC
    CUDA::cudart ckl::ckl nlohmann_json::nlohmann_json)
  
else()
  # It is invalid to install a target twice, too.
  set(QUICKMLP_INSTALL OFF)
endif()


# BUILD_TESTING takes priority, but only if this is the master project
if(QUICKMLP_MASTER_PROJECT AND DEFINED BUILD_TESTING)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
else()
  if(QUICKMLP_TEST)
    add_subdirectory(tests)
  endif()
endif()

# Better symmetry with find_package(qmlp CONFIG) mode.
if(NOT QUICKMLP_MASTER_PROJECT)
  set(qmlp_FOUND
      TRUE
      CACHE INTERNAL "True if qmlp and all required components found on the system")
endif()
