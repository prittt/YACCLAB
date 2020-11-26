
echo "############################################### Install OSX Environment ###############################################"


echo -e "\n\n------------------------------------------> Clean and Update brew" 
#Clean brew cache to avoid memory waste
brew cleanup > /dev/null
#brew cleanup > brew_cleanup.log
rm -rf "'brew cache'"
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
#Update brew and packages
#brew update -y > /dev/null
#brew upgrade -y > /dev/null
#brew update -y > brew_update.log
#brew upgrade -y > brew_upgrade.log
echo -e "------------------------------------------> DONE!" 

echo -e "\n\n------------------------------------------> Install cmake-3.13 (only if it wasn't cached):"
#export DEPS_DIR="${TRAVIS_BUILD_DIR}/deps"
#mkdir ${DEPS_DIR} && cd ${DEPS_DIR}
echo -e "(CMake cache test):"
if [ -d cmake-install -a "$(ls -A cmake-install/)" ]; then
    echo -e "    CMake already installed"
else
    echo -e "    CMake not installed yet, downloading it ... "
    curl -L https://cmake.org/files/v3.13/cmake-3.13.0-Darwin-x86_64.tar.gz > cmake-3.13.tar.gz
    echo -e "    DONE"
    tar -xzf cmake-3.13.tar.gz
    mv cmake-3.13.0-Darwin-x86_64 cmake-install
    #export PATH=${DEPS_DIR}/cmake-install/CMake.app/Contents/bin:$PATH
    #cd ${TRAVIS_BUILD_DIR}
    echo -e "------------------------------------------> DONE!" 

    echo -e "\n\n------------------------------------------> Check CMake version"
    cmake-install/CMake.app/Contents/bin/cmake --version
fi
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

    # Set build instructions for OSX (x64 build).
    ../../cmake-install/CMake.app/Contents/bin/cmake -D CMAKE_C_FLAGS=-m64 -D CMAKE_CXX_FLAGS=-m64 -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=./install_dir -D WITH_FFMPEG=OFF -D WITH_OPENCL=OFF -D WITH_QT=OFF -D WITH_IPP=OFF -D WITH_MATLAB=OFF -D WITH_OPENGL=OFF -D WITH_TIFF=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D WITH_TBB=OFF -D WITH_CUDA=OFF -D WITH_V4L=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D BUILD_SHARED_LIBS=OFF -D BUILD_ZLIB=OFF -D BUILD_opencv_core=ON -D BUILD_opencv_imgproc=ON -D BUILD_opencv_imgcodecs=ON -D BUILD_opencv_videoio=OFF -D BUILD_opencv_highgui=OFF -D BUILD_opencv_video=OFF -D BUILD_opencv_calib3d=OFF -D BUILD_opencv_features2d=OFF -D BUILD_opencv_objdetect=OFF -D BUILD_opencv_ml=OFF -D BUILD_opencv_flann=OFF -D BUILD_opencv_photo=ON -D BUILD_opencv_stitching=OFF -D BUILD_opencv_cudaarithm=OFF -D BUILD_opencv_cudabgsegm=OFF -D BUILD_opencv_cudacodec=OFF -D BUILD_opencv_cudafeatures2d=OFF -D BUILD_opencv_cudafilters=OFF -D BUILD_opencv_cudaimgproc=OFF -D BUILD_opencv_cudalegacy=OFF -D BUILD_opencv_cudaobjdetect=OFF -D BUILD_opencv_cudaoptflow=OFF -D BUILD_opencv_cudastereo=OFF -D BUILD_opencv_cudawarping=OFF -D BUILD_opencv_cudev=OFF -D BUILD_opencv_shape=OFF -D BUILD_opencv_superres=OFF -D BUILD_opencv_videostab=OFF -D BUILD_opencv_viz=OFF -D WITH_OPENEXR=OFF ..

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

echo -e "\n\n------------------------------------------> Install Gnuplot" 
brew install gnuplot > gnuplot_install.log
gnuplot --version
echo -e "------------------------------------------> DONE!" 
  
  
# Download dataset
echo - e "\n\n------------------------------------------> Download of YACCLAB reduced dataset (only if it wasn't cached):" 
if [ -d input -a "$(ls -A input/)" ]; then
    echo -e "    Yacclab dataset already downloaded"
else
    echo -e "    Yacclab dataset not cached, downloading it..."
    wget https://imagelab.ing.unimore.it/files/YACCLAB_dataset3D_reduced.zip -O dataset.zip 
    unzip -qq dataset.zip 
    rm dataset.zip  
    wget imagelab.ing.unimore.it/files/YACCLAB_dataset_reduced.zip -O dataset.zip
    unzip -qq dataset.zip
    rm dataset.zip  
    echo -e "------------------------------------------> DONE!"
fi
  
