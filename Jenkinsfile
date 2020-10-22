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
                        stage('Clean') {
                            steps {
                                timeout(15) {
                                    echo 'Cleaning..'
                                    sh 'rm -r bin'
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
                                sh 'cd bin && ./YACCLAB'
                            }
                        }
                        stage('linux_gpu_end') {
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
