pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
               stage('linux_gpu') {
                    agent {
                        docker { 
                            label 'docker && gpu'
                            image 'stal12/opencv4.4-cuda9.2'
                            args '--gpus 1'
                        }
                    }
                    stages {
                        stage('Build & Run') {
                            steps {
							  timeout(15) {
                                echo 'Building..'
                                cmakeBuild buildDir: 'build', cmakeArgs: '-D BUILD_TARGET=GPU -D BUILD_TESTS=ON', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                              } 
							              }
                        }
                        stage('linux_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('windows_gpu') {
                    agent {
                        label 'windows && gpu'
                    }
                    stages {
                        stage('Build') {
                            steps {
                              timeout(15) {
                                echo 'Building..'
                                cmakeBuild buildDir: 'build', cmakeArgs: '-D BUILD_TARGET=GPU -D BUILD_TESTS=ON',  installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                              }
                            }
                        }
                        stage('Test') {
                            steps {
                              timeout(15) {
                                echo 'Testing..'
                                bat 'cd build && ctest -C Debug -VV'
                              }
                            }
                        }
                        stage('windows_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
            }
        }
    }
}
