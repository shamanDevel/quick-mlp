cmake_minimum_required(VERSION 3.10)

####################################
# Find Python
####################################

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
#find_program(VIRTUALENV virtualenv)
#find_package(Python COMPONENTS Interpreter Development)
get_filename_component(PYTHON_DIRECTORY ${PYTHON_EXECUTABLE} DIRECTORY)


####################################
# Find Pytorch
####################################

# find installation path
if(NOT DEFINED TORCH_PATH)
    # query torch path from python
    if (WIN32)
        execute_process(COMMAND CMD /c python -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0])" OUTPUT_VARIABLE TORCH_FIRST_INCLUDE_DIR)
    else()
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0])" OUTPUT_VARIABLE TORCH_FIRST_INCLUDE_DIR)
    endif()
    get_filename_component(TORCH_ROOT ${TORCH_FIRST_INCLUDE_DIR}/../ ABSOLUTE)
    set(TORCH_PATH "${TORCH_ROOT}" CACHE FILEPATH "path to pytorch in the python installation")
    if(NOT (EXISTS ${TORCH_PATH}))
        message( FATAL_ERROR "Pytorch not found, is it not installed in the python distribution ${PYTHON_DIRECTORY}?")
    else()
        message(STATUS "Torch found at ${TORCH_PATH}")
    endif()
else()
    message(STATUS "Manually specifying torch path as ${TORCH_PATH}")
endif()
# ask Torch's CMake configuration
set(TORCH_CONFIG_PATH "${TORCH_PATH}/share/cmake/Torch" CACHE FILEPATH "possible path where TorchConfig.cmake is located")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CONFIG_PATH})
find_package(Torch REQUIRED)
# get libraries (hard coded), copied from torch.utils.cpp_extension.CUDAExtension
set(TORCH_LIBRARY_NAMES
    c10 c10_cuda torch torch_cpu torch_cuda torch_cuda_cpp torch_cuda_cu torch_python)
set(TORCH_LIBRARIES ${TORCH_LIBRARY})
FOREACH(LIB_NAME ${TORCH_LIBRARY_NAMES})
  set(LIB_VAR "TORCH_LIB_${LIB_NAME}") # Name of the variable which stores result of the search
  FIND_LIBRARY(${LIB_VAR} ${LIB_NAME} PATHS ${TORCH_PATH}/lib)
  list(APPEND TORCH_LIBRARIES ${${LIB_VAR}})
ENDFOREACH()
message(STATUS "Torch: full library list: ${TORCH_LIBRARIES}")
# get include directories
set(TORCH_INCLUDE_DIR "${TORCH_PATH}/include;${TORCH_PATH}/include/torch/csrc/api/include" CACHE FILEPATH "include directory for the pytorch headers")
message(STATUS "Torch: include directories: ${TORCH_INCLUDE_DIR}")

####################################
# BINDING LIBRARY
####################################

# your configuration
set(BINDINGS_FILES
    bind_activation.h
    bind_activation.cpp
    bind_encoding.h
    bind_encoding.cpp
	bindings.cpp
	)

# create the python extension
add_library(qmlp-pytorch-bindings SHARED
	${BINDINGS_FILES}
	)
target_include_directories(qmlp-pytorch-bindings
	PRIVATE ${TORCH_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR}
	)
target_link_libraries(qmlp-pytorch-bindings
	PRIVATE 
    qmlp
	${TORCH_LIBRARIES} ${PYTHON_LIBRARY})
set_target_properties(qmlp-pytorch-bindings PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
set_property(TARGET qmlp-pytorch-bindings PROPERTY CUDA_ARCHITECTURES 61 72)

message(STATUS "Binding output: ${CMAKE_SOURCE_DIR}/bin/${PYTHON_MODULE_PREFIX}${BINDINGS_NAME}${PYTHON_MODULE_EXTENSION}")
add_custom_command(TARGET qmlp-pytorch-bindings
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:qmlp-pytorch-bindings> ${CMAKE_SOURCE_DIR}/bin/${qmlp-pytorch-bindings}
	
	COMMENT "Copies the python extension to bin/"
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
	
	VERBATIM
	)
set_property(TARGET ${BINDINGS_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

