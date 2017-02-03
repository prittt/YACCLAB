# Copyright(c) 2016 - 2017 Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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
set(YACCLAB_DOWNLOAD_DATASET ON CACHE BOOL "If true the YACCLAB dataset will be downloaded automatically") 

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

include_directories("${CMAKE_SOURCE_DIR}/include")

file(GLOB SRCFILES ${SRCFILES} "src/*.cpp" "src/*.c")
file(GLOB HFILES ${HFILES} "include/*.h" "include/*.hpp")
	
set (addExe ${SRCFILES} ${HFILES})
add_executable(${ProjectName} ${addExe})
target_link_libraries (${ProjectName}  ${OpenCV_LIBS})
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Copy config file in the installation folder and rename it		
file(COPY "${CMAKE_SOURCE_DIR}/doc/StandardConfigurationFileDoNotModifyItButTheOneInTheInstallationFolder.cfg" DESTINATION ${CMAKE_INSTALL_PREFIX})
file(RENAME "${CMAKE_INSTALL_PREFIX}/StandardConfigurationFileDoNotModifyItButTheOneInTheInstallationFolder.cfg" "${CMAKE_INSTALL_PREFIX}/config.cfg")

# Download and unpack dataset if necessary
if(YACCLAB_DOWNLOAD_DATASET)
	if (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/input")
		if (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/dataset.zip")
			message(STATUS "Downloading dataset...")
			file(DOWNLOAD http://imagelab.ing.unimore.it/files/YACCLAB_dataset.zip "${CMAKE_INSTALL_PREFIX}/dataset.zip" STATUS status SHOW_PROGRESS)
			list( GET status 0 error_code )
			if( error_code )
				message(ERROR " Cmake is not able to download the YACCLAB dataset. \nPlease, retry or download it manually and put the 'input' unzipped folder in the installation path:\n'" ${CMAKE_INSTALL_PREFIX} "'. \n(The dataset is available at http://imagelab.ing.unimore.it/files/YACCLAB_dataset.zip).")
				if(EXISTS "${CMAKE_INSTALL_PREFIX}/dataset.zip")
					file(REMOVE "${CMAKE_INSTALL_PREFIX}/dataset.zip")
				endif()
			else()
				message(STATUS "Downloading done.")
			endif()
		endif()
		
		if(EXISTS "${CMAKE_INSTALL_PREFIX}/dataset.zip")
			message(STATUS "Unpacking (it may take a while)...")
			execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_INSTALL_PREFIX}/dataset.zip" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}" OUTPUT_QUIET)
			message(STATUS "Unpacking done.")
		endif()
	endif()
endif()
