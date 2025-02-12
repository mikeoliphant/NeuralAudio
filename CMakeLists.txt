cmake_minimum_required(VERSION 3.5)

project(NeuralAudio VERSION 0.0.1)

add_subdirectory(NeuralAudio)
add_subdirectory(NeuralAudioCAPI)

option(BUILD_UTILS "Build NeuralAudio utils" OFF)
if(BUILD_UTILS)
    message(STATUS "Building NeuralAudio Utils")
    add_subdirectory(Utils)
endif()
