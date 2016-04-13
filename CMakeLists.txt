cmake_minimum_required(VERSION 2.0.0)


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
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )


# ----------------------------------------------------------------------------
# Solution folders:
# ----------------------------------------------------------------------------
if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

include_directories( "." )

# ------------------------------------
#opencv
# --------------------------------------------------

FIND_PACKAGE(OpenCV REQUIRED core imgproc)
include_directories( ${OpenCV_INCLUDE_DIRS} )
	 

# --------------------
# Parte del progetto
# --------------------

file(GLOB SRCFILES ${SRCFILES} "*.cpp" "*.c")
file(GLOB HFILES ${HFILES} "*.h" "*.hpp")
	
set (addExe ${SRCFILES} ${HFILES})
add_executable(${ProjectName} ${addExe})
target_link_libraries (${ProjectName}  ${OpenCV_LIBS})
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Config file copy		
file(COPY "config.cfg" DESTINATION ${CMAKE_BINARY_DIR}/Debug)
file(COPY "config.cfg" DESTINATION ${CMAKE_BINARY_DIR}/Release)

# Download and unzip dataset
message(STATUS "Downloading dataset...")
file(DOWNLOAD http://imagelab.ing.unimore.it/files/YACCLAB_dataset.tgz "${CMAKE_INSTALL_PREFIX}/dataset.tgz")
message(STATUS "Downloading done. Unpacking...")
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_INSTALL_PREFIX}/dataset.tgz" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
message(STATUS "Unpacking done.")