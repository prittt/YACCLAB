pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
               stage('ubuntu16_gpu') {
                    agent {
                        docker { 
                            label 'docker && gpu'
                            image 'aimagelab/opencv4.4-cuda9.2-ubuntu16.04'
                            args '--gpus 1'
                        }
                    }
                    stages {
                        stage('Clean') {
                            steps {
                                timeout(15) {
                                    echo 'Cleaning..'
                                    sh 'rm -rf bin || true'
                                }
                            }
                        }
                        stage('Build') {
                            steps {
                                timeout(120) {
                                    echo 'Building..'
                                    sh 'chmod +x tools/jenkins-scripts/run-script.sh'
                                    sh 'export BUILD_TARGET=linux && tools/jenkins-scripts/run-script.sh'
                                }
                            }
                        }
                        stage('Run') {
                            steps {
                                timeout(600) {
                                    echo 'Running..'
                                    sh 'cd bin && ./YACCLAB'
                                }
                             }
                        }
                        stage('ubuntu16_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('ubuntu20_gpu') {
                    agent {
                        docker { 
                            label 'docker && gpu'
                            image 'aimagelab/opencv4.4-cuda11.4.1-ubuntu20.04:withunzip'
                            args '--gpus 1'
                        }
                    }
                    stages {
                        stage('Clean') {
                            steps {
                                timeout(15) {
                                    echo 'Cleaning..'
                                    sh 'rm -rf bin || true'
                                }
                            }
                        }
                        stage('Build') {
                            steps {
                                timeout(120) {
                                    echo 'Building..'
                                    sh 'chmod +x tools/jenkins-scripts/run-script.sh'
                                    sh 'export BUILD_TARGET=linux && tools/jenkins-scripts/run-script.sh'
                                }
                            }
                        }
                        stage('Run') {
                            steps {
                                timeout(600) {
                                    echo 'Running..'
                                    sh 'cd bin && ./YACCLAB'
                                }
                            }
                        }
                        stage('ubuntu20_gpu_end') {
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
