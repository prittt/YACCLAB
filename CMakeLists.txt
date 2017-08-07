# Copyright(c) 2016 - 2017 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met :
#
# *Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and / or other materials provided with the distribution.
#
# * Neither the name of YACCLAB nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.0.0)

set (ProjectName "YACCLAB")
project (${ProjectName})

#Suppress creation of ZERO_CHECK
set(CMAKE_SUPPRESS_REGENERATION true)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Install prefix" FORCE)

#Set YACCLAB_dataset download variable
set(YACCLAB_DOWNLOAD_DATASET OFF CACHE BOOL "If flagged the YACCLAB dataset will be automatically downloaded")
set(YACCLAB_INPUT_DATASET_PATH "${CMAKE_INSTALL_PREFIX}/input" CACHE PATH "'input' dataset path") 
set(YACCLAB_OUTPUT_RESULTS_PATH "${CMAKE_INSTALL_PREFIX}/output" CACHE PATH "'output' results path") 


if(MSVC)
  set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

# ----------------------------------------------------------------------------
#  Check for c++11 support
# ----------------------------------------------------------------------------

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	if(COMPILER_SUPPORTS_CXX11)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	elseif(COMPILER_SUPPORTS_CXX0X)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	else()
		message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no complete C++11 support. Please use a different C++ compiler.")
	endif()
	# Set compilation flag to avoid creation of debug info in release configuration under Linux
	#if(CMAKE_COMPILER_IS_GNUCC)
	#	message(STATUS "Removed debug info in release mode")
	#	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
	#	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
	#endif()
else()
	if(MSVC)
		if(MSVC_VERSION LESS 1800)
			message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no complete C++11 support. Please use a different C++ compiler.")
		endif()
	else()
		message(STATUS "We are not able to check if ${CMAKE_CXX_COMPILER} compiler has complete C++11 support. YACCLAB may not work properly.")
	endif()
endif()


# ----------------------------------------------------------------------------
#  Build & install layouts
# ----------------------------------------------------------------------------

# Save libs and executables in the same place
set(YACCLAB_EXECUTABLE_OUTPUT_PATH "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Output directory for applications" )

# ----------------------------------------------------------------------------
# Solution folders
# ----------------------------------------------------------------------------
if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# --------------------
# OpenCV 
# --------------------

set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc" CACHE STRING "OpenCV packages required by YACCLAB are already set")
FIND_PACKAGE(OpenCV REQUIRED ${OpenCV_REQUIRED_PACKAGES})
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

include_directories("${CMAKE_SOURCE_DIR}/include")

file(GLOB SRCFILES ${SRCFILES} "src/*.cc" "src/*.cpp" "src/*.c")
file(GLOB HFILES ${HFILES} "include/*.h" "include/*.hpp")
file(GLOB INCFILES ${INCFILES} "include/*.inc")

# Set configuration file		
configure_file("${CMAKE_SOURCE_DIR}/doc/config_base.yaml" "${CMAKE_INSTALL_PREFIX}/config.yaml")

source_group("Include Files" FILES ${INCFILES})
source_group("Header Files" FILES ${HFILES})
source_group("Source Files" FILES ${SRCFILES})

set (addExe ${SRCFILES} ${HFILES} ${INCFILES} "${CMAKE_INSTALL_PREFIX}/config.yaml")
add_executable(${ProjectName} ${addExe})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${ProjectName}) # Set YACCLAB as startup group instead of ALL_BUILD
target_link_libraries (${ProjectName}  ${OpenCV_LIBS})

# Creation of output directory
if (NOT EXISTS "${YACCLAB_INPUT_DATASET_PATH}")
		file(MAKE_DIRECTORY ${YACCLAB_OUTPUT_RESULTS_PATH})
endif()
	
# Download and unpack dataset if necessary
if(YACCLAB_DOWNLOAD_DATASET)
	# Creation of input directory
	if (NOT EXISTS "${YACCLAB_INPUT_DATASET_PATH}")
		file(MAKE_DIRECTORY ${YACCLAB_INPUT_DATASET_PATH})
	endif()
	
	if (EXISTS "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip")
		message(STATUS "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip already exists, download skipped!")
	else()
		message(STATUS "Downloading dataset...")
		file(DOWNLOAD http://imagelab.ing.unimore.it/files/YACCLAB_dataset_new.zip "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip" STATUS status SHOW_PROGRESS)
		list( GET status 0 error_code )
		if( error_code )
			message(ERROR " Cmake is not able to download the YACCLAB dataset. \nPlease, retry or download it manually and put all unzipped dataset directories in the 'input' path:\n'" ${YACCLAB_INPUT_DATASET_PATH} "'. \n(The dataset is available at http://imagelab.ing.unimore.it/files/YACCLAB_dataset_new.zip).")
			if(EXISTS "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip")
				file(REMOVE "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip")
			endif()
		else()
			message(STATUS "Downloading done.")
		endif()
		
		if(EXISTS "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip")
			message(STATUS "Unpacking (it may take a while)...")
			execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${YACCLAB_INPUT_DATASET_PATH}/dataset.zip" WORKING_DIRECTORY "${YACCLAB_INPUT_DATASET_PATH}" OUTPUT_QUIET)
			message(STATUS "Unpacking done.")
		endif()
	endif()
	
endif()
