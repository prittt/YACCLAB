#!/bin/bash

# exit this script if any commmand fails 
set -e

function install_linux_environment()
{  
  echo "############################################### Install Linux Environment ###############################################"

  echo -e "\n\n------------------------------------------> Install ubuntu-toolchain for gcc-4.8:"
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Update apt"
  sudo apt-get update -y
  #sudo apt-get upgrade -y #Don't do that
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Install gcc-4.8:"
  if [ "$CXX" = "g++" ]; then sudo apt-get install -qq g++-4.8; fi
  if [ "$CXX" = "g++" ]; then export CXX="g++-4.8" CC="gcc-4.8"; fi
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 90
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Install OpenCV-3.1.0 and dependent packages:"
  sudo apt-get install -y build-essential
  sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
  # Not necessary dependency:
  sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libpng-dev libdc1394-22-dev
  #sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

  # Download v3.1.0 .zip file and extract.
  curl -L --progress-bar https://github.com/Itseez/opencv/archive/3.1.0.zip > opencv.zip
  unzip -qq opencv.zip
  rm opencv.zip
  cd opencv-3.1.0
  
  # Create a new 'build' folder.
  mkdir build
  cd build
  
  # Set build instructions for Ubuntu distro.
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_FFMPEG=OFF -D WITH_OPENCL=OFF -D WITH_QT=OFF -D WITH_IPP=OFF -D WITH_MATLAB=OFF -D WITH_OPENGL=OFF -D WITH_QT=OFF -D WITH_TIFF=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D WITH_TBB=OFF -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF ..
  
  # Run 'make' with four threads.
  make -j4
  
  # Install to OS.
  sudo make install
  
  # Add configuration to OpenCV to tell it where the library files are located on the file system (/usr/local/lib)
  sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
  
  sudo ldconfig
  echo "OpenCV installed."
  
  # Return to the repo "root" folder
  cd ../../
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Install Gnuplot and dependent packages"
  #sudo apt-get install -y build-essentials g++ gcc # Already installed
  #TODO install dependency required by gnuplot: libcerf and Qt5 (Core, Gui, Network, Svg, PrintSupport)
  wget https://sourceforge.net/projects/gnuplot/files/gnuplot/5.0.0/gnuplot-5.0.0.tar.gz/download -O gnuplot-5.0.0.tar.gz
  tar -xzf gnuplot-5.0.0.tar.gz
  rm gnuplot-5.0.0.tar.gz
  cd gnuplot-5.0.0
  #TODO redirection doesn't work? 
  ./configure --prefix=/usr/local
  make -s
  sudo make install
  cd ../../
  gnuplot --version
  echo -e "------------------------------------------> DONE!" 
}

function install_osx_environment()
{
  echo "############################################### Install OSX Environment ###############################################"

  echo -e "\n\n------------------------------------------> Clean and Update brew"
  #Clean brew cache to avoid memory waste
  brew cleanup
  rm -rf "'brew cache'"
  
  #Update brew and packages
  brew update -y
  brew upgrade -y
  echo -e "------------------------------------------> DONE!" 
  
  echo -e "\n\n------------------------------------------> Check CMake version"
  cmake --version
  echo -e "------------------------------------------> DONE!" 

  echo -e "\n\n------------------------------------------> Install OpenCV-3.1.0 and dependent packages:"
  # Download v3.1.0 .zip file and extract.
  curl -L --progress-bar https://github.com/Itseez/opencv/archive/3.1.0.zip > opencv.zip
  unzip -qq opencv.zip
  rm opencv.zip
  cd opencv-3.1.0
  
  # Create a new 'build' folder.
  mkdir build
  cd build
  
  # Set build instructions for Ubuntu distro.
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_FFMPEG=OFF -D WITH_OPENCL=OFF -D WITH_QT=OFF -D WITH_IPP=OFF -D WITH_MATLAB=OFF -D WITH_OPENGL=OFF -D WITH_QT=OFF -D WITH_TIFF=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D WITH_TBB=OFF -D WITH_CUDA=OFF -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF ..  
  
  # Run 'make' with four threads.
  make -j
  
  # Install to OS.
  sudo make install
  
  echo "OpenCV installed."
  
  # Return to the repo "root" folder
  cd ../../
  echo -e "------------------------------------------> DONE!" 
	
  echo -e "\n\n------------------------------------------> Install Gnuplot and dependent packages:"
  brew install gnuplot
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
