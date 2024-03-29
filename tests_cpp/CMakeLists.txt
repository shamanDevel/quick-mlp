# CMakeLists.txt -- Build system for the qmlp test suite
#
# Copyright (c) 202 Sebastian Weiss <sebastian13.weiss@tum.de>
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


# Full set of test files
set(QUICKMLP_KERNEL_FILES

    )
set(QUICKMLP_TEST_FILES
    main.cpp
    catch.hpp
    test_eval_simple.cpp
    eigen_half.h
    test_against_eigen.h
    test_against_eigen1.cpp
    test_against_eigen2.cpp
    test_hashgrid.cpp
    )

add_executable(qmlp-tests
    ${QUICKMLP_TEST_FILES} ${QUICKMLP_KERNEL_FILES})
    
set_target_properties(qmlp-tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(qmlp-tests
    PRIVATE qmlp::qmlp-library CUDA::cudart)
    
# Eigen
set(BUILD_TESTING OFF)
add_subdirectory(Eigen)
target_link_libraries(qmlp-tests PRIVATE Eigen3::Eigen)

# Compile with compiler warnings turned on
if(MSVC)
    target_compile_options(qmlp-tests PRIVATE /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Intel|Clang)" AND NOT QUICKMLP_CUDA_TESTS)
    target_compile_options(
      qmlp-tests
      PRIVATE -Wall
              -Wextra
              -Wconversion
              -Wcast-qual
              -Wdeprecated
              -Wundef
              -Wnon-virtual-dtor)
endif()

# Copy to the bin/ order and run from there
add_custom_command(TARGET qmlp-tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:qmlp-tests> ${CMAKE_SOURCE_DIR}/bin/qmlp-tests${CMAKE_EXECUTABLE_SUFFIX}

    COMMENT "Copies the executable to bin/"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..

    VERBATIM
    )
set_property(TARGET qmlp-tests PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
