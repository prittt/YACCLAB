cmake_minimum_required(VERSION 2.4.0)

set (ProjectName "YACCLAB")

project (${ProjectName})

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build" CACHE PATH "Install prefix" FORCE)

if(MSVC)
  set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


# ----------------------------------------------------------------------------
#  Build & install layouts
# ----------------------------------------------------------------------------

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}" CACHE PATH "Output directory for applications" )


# ----------------------------------------------------------------------------
# Solution folders
# ----------------------------------------------------------------------------
if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

include_directories( "." )

# --------------------
# OpenCV 
# --------------------

FIND_PACKAGE(OpenCV REQUIRED core imgcodecs imgproc)
include_directories( ${OpenCV_INCLUDE_DIRS} )

if(MSVC)
	if("${OpenCV_STATIC}" STREQUAL "ON")
		message(STATUS "Setting static runtime for MSVC")
		foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO) 
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		endforeach(flag_var)
	endif()
endif()

# --------------------
# Project
# --------------------

file(GLOB SRCFILES ${SRCFILES} "*.cpp" "*.c")
file(GLOB HFILES ${HFILES} "*.h" "*.hpp")
	
set (addExe ${SRCFILES} ${HFILES})
add_executable(${ProjectName} ${addExe})
target_link_libraries (${ProjectName}  ${OpenCV_LIBS})
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Config file copy		
file(COPY "config.cfg" DESTINATION ${CMAKE_INSTALL_PREFIX})

# Download and unpack dataset
if (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/dataset.zip")
	message(STATUS "Downloading dataset...")
	file(DOWNLOAD http://imagelab.ing.unimore.it/files/YACCLAB_dataset.zip "${CMAKE_INSTALL_PREFIX}/dataset.zip")
	message(STATUS "Downloading done. Unpacking...")
	execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_INSTALL_PREFIX}/dataset.zip" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}" OUTPUT_QUIET)
	message(STATUS "Unpacking done.")
endif()
