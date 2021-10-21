# Copyright (c) 2021, the YACCLAB contributors, as 
# shown by the AUTHORS file. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.13)

# ----------------------------------------------------------------------------
#  Base variables
# ----------------------------------------------------------------------------

set (ProjectName "YACCLAB")
project (${ProjectName})

# ----------------------------------------------------------------------------
#  YACCLAB and CMAKE variables
# ----------------------------------------------------------------------------
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Install prefix" FORCE)

set(YACCLAB_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support if checked")
set(YACCLAB_ENABLE_3D OFF CACHE BOOL "Enable 3D support if checked")
set(YACCLAB_ENABLE_EPDT_19C OFF CACHE BOOL "Enable 3D algorithm based on Heuristic Decision Trees (may noticeably increase build time)")
set(YACCLAB_ENABLE_EPDT_22C OFF CACHE BOOL "Enable 3D algorithm based on Heuristic Decision Trees (may noticeably increase build time)")
set(YACCLAB_ENABLE_EPDT_26C OFF CACHE BOOL "Enable 3D algorithm based on Heuristic Decision Trees (may noticeably increase build time)")
set(YACCLAB_FORCE_CONFIG_GENERATION OFF CACHE BOOL "Force generation of config file, possibly overwriting the current one")
set(YACCLAB_DOWNLOAD_DATASET OFF CACHE BOOL "If flagged the YACCLAB dataset will be automatically downloaded")
set(YACCLAB_DOWNLOAD_DATASET_3D OFF CACHE BOOL "If flagged the YACCLAB 3D dataset will be automatically downloaded")
set(YACCLAB_INPUT_DATASET_PATH "${CMAKE_INSTALL_PREFIX}/input" CACHE PATH "'input' dataset path") 
set(YACCLAB_OUTPUT_RESULTS_PATH "${CMAKE_INSTALL_PREFIX}/output" CACHE PATH "'output' results path") 
set(YACCLAB_OS "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}" CACHE INTERNAL STRING)
set(YACCLAB_CI_CPU2D OFF CACHE INTERNAL "This variable is used to handle travis-ci parameters, do not change its value")
set(YACCLAB_IS_JENKINS OFF CACHE INTERNAL "This variable is used to handle jenkins parameters, do not change its value")
#set(YACCLAB_CXX_STANDARD 14 CACHE STRING "Version of the C++ standard (must be >= 14)") 
#set_property(CACHE YACCLAB_CXX_STANDARD PROPERTY STRINGS 14 17 20 23)

if(MSVC)
  set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()


# ----------------------------------------------------------------------------
#  Check for c++14 support
# ----------------------------------------------------------------------------

if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang" OR CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
#    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
#    if(NOT CMAKE_CXX_FLAGS MATCHES "-m32" AND NOT CMAKE_CXX_FLAGS MATCHES "-m64")
#        set(CMAKE_CXX_FLAGS "-m32 ${CMAKE_CXX_FLAGS}" CACHE STRING "32-bit compiled" FORCE)
#    endif()
#    if(NOT CMAKE_C_FLAGS MATCHES "-m32" AND NOT CMAKE_C_FLAGS MATCHES "-m64")
#	set(CMAKE_C_FLAGS "-m32 ${CMAKE_C_FLAGS}" CACHE STRING "32-bit compiled" FORCE)
#    endif()
#endif()

# The following commands would serve to simplify the check on the compiler version but do not yet seem to work perfectly
set(CMAKE_CXX_STANDARD_REQUIRED YES) # Force the following command to generate an error when it fails
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS NO)
   
#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
#    include(CheckCXXCompilerFlag)
#    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
#
#	if(COMPILER_SUPPORTS_CXX14 AND NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+14")
#		set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}" CACHE STRING "Enable c++14" FORCE)
#	elseif(NOT COMPILER_SUPPORTS_CXX14)
#		message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
#	endif()
#	
#	message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
#	message(STATUS "C flags: ${CMAKE_C_FLAGS}")
#elseif(MSVC)
#		if(MSVC_VERSION LESS 1900)
#			message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
#		endif()
#else()
#	message(STATUS "We are not able to check if ${CMAKE_CXX_COMPILER} compiler has C++14 support. YACCLAB may not work properly.")
#endif()

# To be able to use __cplusplus macro to check C++ version
if(MSVC)
	add_compile_options("/Zc:__cplusplus")
endif()

# ----------------------------------------------------------------------------
#  Check for CUDA support
# ----------------------------------------------------------------------------
if(YACCLAB_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)	
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		add_definitions(-DYACCLAB_WITH_CUDA)
        
        set(CMAKE_CUDA_STANDARD_REQUIRED YES) # Force the following command to generate an error when it fails
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_EXTENSIONS NO)
        
		# Detect and set CUDA architecture
		include(FindCUDA/select_compute_arch)
		CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
		string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
		string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
		string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
		SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
		message(STATUS "CUDA detected architecture: ${CUDA_ARCH_LIST}")
		
        #find_package(CUDA)
        #CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS)
        #message(STATUS ${ARCH_FLAGS})
        #SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
        
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
                
	else()
		message(WARNING "No CUDA support")
	endif()
endif()


# ----------------------------------------------------------------------------
#  Build & install layouts
# ----------------------------------------------------------------------------
set(YACCLAB_EXECUTABLE_OUTPUT_PATH "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Output directory for applications")


# ----------------------------------------------------------------------------
# Solution folders
# ----------------------------------------------------------------------------

if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()


# ----------------------------------------------------------------------------
# OpenCV 
# ----------------------------------------------------------------------------

#set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc" CACHE STRING "OpenCV packages required by YACCLAB are already set")
#string(FIND "${OpenCV_REQUIRED_PACKAGES}" "cudafeatures2d" cudafeatures2d_pos)
#if (YACCLAB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND ${cudafeatures2d_pos} EQUAL -1) 
#	set(OpenCV_REQUIRED_PACKAGES "${OpenCV_REQUIRED_PACKAGES};cudafeatures2d" CACHE STRING "OpenCV packages required by YACCLAB are already set" FORCE)    
#endif()
set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc")
if (YACCLAB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER) 
	set(OpenCV_REQUIRED_PACKAGES "${OpenCV_REQUIRED_PACKAGES};cudafeatures2d")    
endif()
message(STATUS "OpenCV required packages: ${OpenCV_REQUIRED_PACKAGES}")

FIND_PACKAGE(OpenCV REQUIRED ${OpenCV_REQUIRED_PACKAGES})
include_directories( ${OpenCV_INCLUDE_DIRS} )

if(MSVC)
	if("${OpenCV_STATIC}" STREQUAL "ON" OR "${OpenCV_SHARED}" STREQUAL "OFF")
		message(STATUS "Setting static runtime for MSVC")
		foreach(flag_var CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		endforeach(flag_var)
		if (YACCLAB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
			foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO) 
				string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
			endforeach(flag_var)
		endif()
	endif()
endif()


# ----------------------------------------------------------------------------
# OpenMP
# ----------------------------------------------------------------------------

find_package(OpenMP)
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
#if(OpenMP_CXX_FOUND)
#    add_compile_options("${OpenMP_CXX_FLAGS}")
#    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
#endif()


# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------
message(STATUS "OS: ${YACCLAB_OS}, OpenCV: ${OpenCV_VERSION}")
add_definitions(-DOPENCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR})
add_definitions(-DYACCLAB_OS="${YACCLAB_OS}")

# ----------------------------------------------------------------------------
# Project
# ----------------------------------------------------------------------------

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/cuda/include")
include_directories("${CMAKE_INSTALL_PREFIX}") # To avoid linking error when a new file is added to the project directly from Visual Studio IDE

#SET_SOURCE_FILES_PROPERTIES("${CMAKE_SOURCE_DIR}/src/labeling_bolelli_2019.cc" PROPERTIES COMPILE_FLAGS "-O1")

add_definitions(-D_CRT_SECURE_NO_WARNINGS) #To suppress 'fopen' opencv warning/bug  


# ----------------------------------------------------------------------------
# Configuration file
# ----------------------------------------------------------------------------

if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/config.yaml" OR YACCLAB_FORCE_CONFIG_GENERATION)
    if(YACCLAB_CI_CPU)
        configure_file("${CMAKE_SOURCE_DIR}/doc/config_ci_cpu.yaml" "${CMAKE_INSTALL_PREFIX}/config.yaml.tmp")
    elseif(YACCLAB_CI_CPU2D)
        configure_file("${CMAKE_SOURCE_DIR}/doc/config_ci_cpu2d.yaml" "${CMAKE_INSTALL_PREFIX}/config.yaml.tmp")
    elseif(YACCLAB_IS_JENKINS)	
        configure_file("${CMAKE_SOURCE_DIR}/doc/config_jenkins.yaml" "${CMAKE_INSTALL_PREFIX}/config.yaml.tmp")
    else()
        file(READ "${CMAKE_SOURCE_DIR}/doc/config_2d.yaml" YACCLAB_CONFIG_2D)

        if(YACCLAB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
            file(READ "${CMAKE_SOURCE_DIR}/doc/config_cuda_2d.yaml" YACCLAB_CONFIG_CUDA_2D)
            if(YACCLAB_ENABLE_3D)
                file(READ "${CMAKE_SOURCE_DIR}/doc/config_cuda_3d.yaml" YACCLAB_CONFIG_CUDA_3D)
            endif()
        endif()

        if(YACCLAB_ENABLE_3D)
            if(YACCLAB_ENABLE_EPDT_19C)
                file(READ "${CMAKE_SOURCE_DIR}/doc/config_epdt_19c.yaml" YACCLAB_CONFIG_EPDT_19C)
            endif()
            if(YACCLAB_ENABLE_EPDT_22C)
                file(READ "${CMAKE_SOURCE_DIR}/doc/config_epdt_22c.yaml" YACCLAB_CONFIG_EPDT_22C)
            endif()
            if(YACCLAB_ENABLE_EPDT_26C)
                file(READ "${CMAKE_SOURCE_DIR}/doc/config_epdt_26c.yaml" YACCLAB_CONFIG_EPDT_26C)
            endif()
            set(YACCLAB_CONFIG_EPDT_ALGORITHMS "${YACCLAB_CONFIG_EPDT_19C}${YACCLAB_CONFIG_EPDT_22C}${YACCLAB_CONFIG_EPDT_26C}")
            configure_file("${CMAKE_SOURCE_DIR}/doc/config_3d.yaml" "${CMAKE_INSTALL_PREFIX}/config_3d.yaml.tmp")
            file(READ "${CMAKE_INSTALL_PREFIX}/config_3d.yaml.tmp" YACCLAB_CONFIG_3D)
            file(REMOVE "${CMAKE_INSTALL_PREFIX}/config_3d.yaml.tmp")
        endif()
        
        configure_file("${CMAKE_SOURCE_DIR}/doc/config_base.yaml" "${CMAKE_INSTALL_PREFIX}/config.yaml.tmp")                
    endif()
    
    file(READ "${CMAKE_INSTALL_PREFIX}/config.yaml.tmp" YACCLAB_CONFIG_FILE)
    file(REMOVE "${CMAKE_INSTALL_PREFIX}/config.yaml.tmp")
    
    #file(CONFIGURE OUTPUT "${CMAKE_INSTALL_PREFIX}/config.yaml" CONTENT "${YACCLAB_CONFIG_FILE}" NEWLINE_STYLE LF)     to be used with CMake >= 18
    file(GENERATE OUTPUT "${CMAKE_INSTALL_PREFIX}/config.yaml" CONTENT "${YACCLAB_CONFIG_FILE}")    
endif()

add_executable(${ProjectName} "${CMAKE_INSTALL_PREFIX}/config.yaml")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${ProjectName}) # Set YACCLAB as startup group instead of ALL_BUILD
target_link_libraries (${ProjectName}  ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ProjectName} OpenMP::OpenMP_CXX)
endif()

# ----------------------------------------------------------------------------
# Output and datasests
# ----------------------------------------------------------------------------

# Set YACCLAB as startup group instead of ALL_BUILD
#set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT YACCLAB)

#target_link_libraries (${ProjectName} ${OpenCV_LIBS})

add_subdirectory(src)
add_subdirectory(include)
if(YACCLAB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda/src)
    add_subdirectory(cuda/include)
    include_directories("${CUDA_INCLUDE_DIRS}")
endif()

source_group("Header Files" REGULAR_EXPRESSION "h$|hpp$")
source_group("Source Files" REGULAR_EXPRESSION "cc$|cpp$|c$")
source_group("Include Files" REGULAR_EXPRESSION "inc$|inc.h$")

if(YACCLAB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    source_group("Cuda Files" REGULAR_EXPRESSION "cu$")
endif()
# Creation of the output directory
if (NOT EXISTS "${YACCLAB_OUTPUT_RESULTS_PATH}")
		file(MAKE_DIRECTORY ${YACCLAB_OUTPUT_RESULTS_PATH})
endif()

message(STATUS ${CUDA_NVCC_FLAGS})

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

if (YACCLAB_DOWNLOAD_DATASET_3D)
	# Creation of input directory
	if (NOT EXISTS "${YACCLAB_INPUT_DATASET_PATH}")
		file(MAKE_DIRECTORY ${YACCLAB_INPUT_DATASET_PATH})
	endif()

	if (EXISTS "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip")
		message(STATUS "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip already exists, download skipped!")
	else()
		message(STATUS "Downloading 3D dataset...")
		file(DOWNLOAD http://imagelab.ing.unimore.it/files/YACCLAB_dataset3D.zip "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip" STATUS status SHOW_PROGRESS)
		list( GET status 0 error_code )
		if( error_code )
			message(ERROR " Cmake is not able to download the YACCLAB 3D dataset. \nPlease, retry or download it manually and put all unzipped dataset directories in the 'input' path:\n'" ${YACCLAB_INPUT_DATASET_PATH} "'. \n(The dataset is available at http://imagelab.ing.unimore.it/files/YACCLAB_dataset3D.zip).")
			if(EXISTS "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip")
				file(REMOVE "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip")
			endif()
		else()
			message(STATUS "Downloading done.")
		endif()
		
		if(EXISTS "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip")
			message(STATUS "Unpacking (it may take a while)...")
			execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${YACCLAB_INPUT_DATASET_PATH}/dataset3D.zip" WORKING_DIRECTORY "${YACCLAB_INPUT_DATASET_PATH}" OUTPUT_QUIET)
			message(STATUS "Unpacking done.")
		endif()
	endif()
endif()
