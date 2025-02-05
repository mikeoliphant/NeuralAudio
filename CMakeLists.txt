cmake_minimum_required(VERSION 3.5)

project(NeuralAudio VERSION 0.0.1)

add_subdirectory(NeuralAudio)

option(BUILD_UTILS "Build NeuralAudio utils" OFF)
if(BUILD_UTILS)
    message(STATUS "Building NeuralAudio Utils")

    if(BUILD_STATIC_RTNEURAL)
    else()
        message(STATUS "Turning static RTNeural build on for Utils")
        set(BUILD_STATIC_RTNEURAL ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(Utils)
endif()
