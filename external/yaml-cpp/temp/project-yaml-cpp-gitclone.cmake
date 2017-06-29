if("master" STREQUAL "")
  message(FATAL_ERROR "Tag for git checkout should not be empty.")
endif()

set(run 0)

if("C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/stamp/project-yaml-cpp-gitinfo.txt" IS_NEWER_THAN "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/stamp/project-yaml-cpp-gitclone-lastrun.txt")
  set(run 1)
endif()

if(NOT run)
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: 'C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/stamp/project-yaml-cpp-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src'")
endif()

set(git_options)

# disable cert checking if explicitly told not to do it
set(tls_verify "")
if(NOT "x" STREQUAL "x" AND NOT tls_verify)
  list(APPEND git_options
    -c http.sslVerify=false)
endif()

set(git_clone_options)

set(git_shallow "")
if(git_shallow)
  list(APPEND git_clone_options --depth 1 --no-single-branch)
endif()

set(git_progress "")
if(git_progress)
  list(APPEND git_clone_options --progress)
endif()

set(git_config "")
foreach(config IN LISTS git_config)
  list(APPEND git_clone_options --config ${config})
endforeach()

# try the clone 3 times incase there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe" ${git_options} clone ${git_clone_options} --origin "origin" "https://github.com/jbeder/yaml-cpp.git" "src"
    WORKING_DIRECTORY "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/jbeder/yaml-cpp.git'")
endif()

# Use `git checkout <branch>` even though this risks ambiguity with a
# local path.  Unfortunately we cannot use `git checkout <tree-ish> --`
# because that will not search for remote branch names, a common use case.
execute_process(
  COMMAND "C:/Program Files/Git/cmd/git.exe" ${git_options} checkout master
  WORKING_DIRECTORY "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

execute_process(
  COMMAND "C:/Program Files/Git/cmd/git.exe" ${git_options} submodule init 
  WORKING_DIRECTORY "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to init submodules in: 'C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src'")
endif()

execute_process(
  COMMAND "C:/Program Files/Git/cmd/git.exe" ${git_options} submodule update --recursive --init 
  WORKING_DIRECTORY "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/stamp/project-yaml-cpp-gitinfo.txt"
    "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/stamp/project-yaml-cpp-gitclone-lastrun.txt"
  WORKING_DIRECTORY "C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'C:/Users/Michele Cancilla/Desktop/NewYACCLAB/external/yaml-cpp/stamp/project-yaml-cpp-gitclone-lastrun.txt'")
endif()

