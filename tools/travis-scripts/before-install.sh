#!/bin/bash

# Copyright(c) 2016 - 2017 Federico Bolelli
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

# exit this script if any commmand fails 
set -e

function install_linux_environment()
{  
  echo "############################################### Install Linux Environment ###############################################"

  echo -e "\n\n------------------------------------------> Install ubuntu-toolchain for gcc-4.8:"
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Update apt"
  sudo apt-get -qq update -y
  #sudo apt-get upgrade -y #Don't do that
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Install gcc-4.8:"
  if [ "$CXX" = "g++" ]; then sudo apt-get -qq install g++-4.8; fi
  if [ "$CXX" = "g++" ]; then export CXX="g++-4.8" CC="gcc-4.8"; fi
  sudo update-alternatives --quiet --install /usr/bin/g++ g++ /usr/bin/g++-4.8 90
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Install OpenCV-3.1.0 (only if they weren't cached) and dependent packages:"
  sudo apt-get -qq -y install build-essential
  sudo apt-get -qq -y install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
  # Not necessary dependency:
  sudo apt-get -qq -y install python-dev python-numpy libtbb2 libtbb-dev libpng-dev libdc1394-22-dev
  #sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

  echo -e "(OpenCV cache test):"
  if [ -d opencv-3.1.0 -a "$(ls -A opencv-3.1.0/)" ]; then
    echo -e "    OpenCV already installed"
  else
    echo -e "    OpenCV not installed yet, downloading it ... "
    # Download v3.1.0 .zip file and extract.
    #curl -L --progress-bar https://github.com/Itseez/opencv/archive/3.1.0.zip > opencv.zip
	curl -L https://github.com/Itseez/opencv/archive/3.1.0.zip > opencv.zip
    echo -e "    DONE"
    unzip -qq opencv.zip
    rm opencv.zip
    cd opencv-3.1.0
  
    # Create a new 'build' folder.
    mkdir build
    cd build
  
    # Create 'install_dir' folder
	mkdir install_dir
  
    # Set build instructions for Ubuntu distro.
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=./install_dir -D WITH_FFMPEG=OFF -D WITH_OPENCL=OFF -D WITH_QT=OFF -D WITH_IPP=OFF -D WITH_MATLAB=OFF -D WITH_OPENGL=OFF -D WITH_QT=OFF -D WITH_TIFF=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D WITH_TBB=OFF -D WITH_V4L=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D BUILD_SHARE_LIBS=OFF ..
  
    # Run 'make' with four threads.
    make -j4
  
    # Install to OS.
    sudo make install

	# We move the following commands out of there in order to export OpenCV configuration variable also when OpenCV libraries were cached
	# sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
    # sudo ldconfig
	
	echo "OpenCV installed."
  
    # Return to the repo "root" folder
    cd ../../
  fi
  
  # Add configuration to OpenCV to tell it where the library files are located on the file system (/usr/local/lib)
  export LD_LIBRARY_PATH=./opencv-3.1.0/build/install_dir
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Install Gnuplot and dependent packages"
  #sudo apt-get install -y build-essentials g++ gcc # Already installed
  #TODO install dependency required by gnuplot: libcerf and Qt5 (Core, Gui, Network, Svg, PrintSupport)
  wget https://sourceforge.net/projects/gnuplot/files/gnuplot/5.0.0/gnuplot-5.0.0.tar.gz/download -O gnuplot-5.0.0.tar.gz
  tar -xzf gnuplot-5.0.0.tar.gz
  rm gnuplot-5.0.0.tar.gz
  cd gnuplot-5.0.0
  ./configure --prefix=/usr/local > /dev/null
  # Note that gnuplot build process exit with some warnings (delete redirection to see it)
  make -s > /dev/null
  sudo make install > /dev/null
  cd ../../
  gnuplot --version
  echo -e "------------------------------------------> DONE!" 
}

function install_osx_environment()
{
  echo "############################################### Install OSX Environment ###############################################"

  echo -e "\n\n------------------------------------------> Clean and Update brew"
  #Clean brew cache to avoid memory waste
  brew cleanup > /dev/null
  #brew cleanup > brew_cleanup.log
  rm -rf "'brew cache'"
  
  #Update brew and packages
  #brew update -y > /dev/null
  #brew upgrade -y > /dev/null
  #brew update -y > brew_update.log
  #brew upgrade -y > brew_upgrade.log
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Check CMake version"
  cmake --version
  echo -e "------------------------------------------> DONE!" 

  echo -e "\n\n------------------------------------------> Install OpenCV-3.1.0 (only if they weren't cached) and dependent packages:"

  echo -e "(OpenCV cache test):"
  if [ -d opencv-3.1.0 -a "$(ls -A opencv-3.1.0/)" ]; then
    echo -e "    OpenCV already installed"
  else
    echo -e "    OpenCV not installed yet, downloading it ... "
    # Download v3.1.0 .zip file and extract.
    #curl -L --progress-bar https://github.com/Itseez/opencv/archive/3.1.0.zip > opencv.zip
	curl -L https://github.com/Itseez/opencv/archive/3.1.0.zip > opencv.zip
    echo -e "    DONE"
	unzip -qq opencv.zip
    rm opencv.zip
    cd opencv-3.1.0

    # Create a new 'build' folder.
    mkdir build
    cd build
  
    # Create 'install_dir' folder
	mkdir install_dir
    
	# Set build instructions for Ubuntu distro.
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=./install_dir -D WITH_FFMPEG=OFF -D WITH_OPENCL=OFF -D WITH_QT=OFF -D WITH_IPP=OFF -D WITH_MATLAB=OFF -D WITH_OPENGL=OFF -D WITH_TIFF=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D WITH_TBB=OFF -D WITH_CUDA=OFF -D WITH_V4L=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D BUILD_SHARED_LIBS=OFF ..
  
    # Run 'make' with four threads.
    make -j
  
    # Install to OS.
    sudo make install
  
    echo "OpenCV installed."
  
    # Return to the repo "root" folder
    cd ../../
  fi
  # Add configuration to OpenCV to tell it where the library files are located on the file system (/usr/local/lib). Is this reallyu necessary on OSX?
  export LD_LIBRARY_PATH=./opencv-3.1.0/build/install_dir
  echo -e "------------------------------------------> DONE!"
	
  echo -e "\n\n------------------------------------------> Install Gnuplot and dependent packages:"
  brew install gnuplot > gnuplot_install.log
  gnuplot --version
  echo -e "------------------------------------------> DONE!" 
	
}

function pass(){
	echo "pass"
}

# Set up environment according os and target
function install_environement_for_pull_request()
{
    if [ "$TRAVIS_OS_NAME" == "linux" ]; then
        if [ "$BUILD_TARGET" == "linux" ]; then
            install_linux_environment
        fi

        #if [ "$BUILD_TARGET" == "android" ]; then
            # not supported yet
        #fi
    fi

    if [ "$TRAVIS_OS_NAME" == "osx" ]; then
        install_osx_environment
    fi
}


# build pull request
#if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
    install_environement_for_pull_request
#fi
