cmake_minimum_required(VERSION 3.0.0)


set (ProjectName "YACCLAB")

project (${ProjectName})

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

if(MSVC)
  set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
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

set(OpenCV_STATIC OFF)
#set(OpenCVBuildPAth "$ENV{OPENCV_DIR}")
set (OpenCVBuildPAth  "C:/users/federico/Desktop/ConnectedComponents/opencv/build")
FIND_PACKAGE(OpenCV REQUIRED NO_MODULE
		PATHS ${OpenCVBuildPAth}
		NO_DEFAULT_PATH)
include_directories( ${OpenCV_INCLUDE_DIRS} )
	 

# --------------------
# Parte del progetto
# --------------------

file(GLOB SRCFILES ${SRCFILES} "*.cpp" "*.c")

file(GLOB HFILES ${HFILES} "*.h" "*.hpp")

#	source_group (Source FILES ${SRCFILES_t})
#	source_group (Headers FILES ${HFILES_t})

	
set (addExe ${SRCFILES} ${HFILES})
add_executable(${ProjectName} ${addExe})
target_link_libraries (${ProjectName}  ${OpenCV_LIBS})
include_directories (${CMAKE_CURRENT_BINARY_DIR})



# ---------------------------------
# config file copy		
	file(COPY "files.txt" DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
	file(COPY "files.txt" DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)

# ------------------------------------



# ----------------------------------------------------------------------------
# Summary:
# ----------------------------------------------------------------------------
message("")
message("End  =====================================")