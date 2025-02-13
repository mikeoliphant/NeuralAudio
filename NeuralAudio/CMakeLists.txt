set(CMAKE_CXX_STANDARD 20)

add_definitions(-DNAM_SAMPLE_FLOAT)
add_definitions(-DDSP_SAMPLE_FLOAT)

option(BUILD_STATIC_RTNEURAL "Build Static RTNeural" OFF)
if(BUILD_STATIC_RTNEURAL)
    message(STATUS "Building static RTNeural models")
    add_definitions(-DBUILD_STATIC_RTNEURAL)
else()
    message(STATUS "NOT Building static RTNeural models")
endif()

option(BUILD_NAMCORE "Build NAM Core" OFF)
if(BUILD_NAMCORE)
    message(STATUS "Building NAM Core implementation")
    add_definitions(-DBUILD_NAMCORE)
else()
    message(STATUS "NOT Building NAM Core implementation")
endif()

set(WAVENET_FRAMES "64" CACHE STRING "WaveNet frame size")

add_definitions(-DWAVENET_MAX_NUM_FRAMES=${WAVENET_FRAMES})

message(STATUS "WaveNet frame size is: ${WAVENET_FRAMES}")

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

set(SOURCES
	NeuralModel.h
	NeuralModel.cpp
	NAMModel.h
	RTNeuralModel.h
	Activation.h
	WaveNet.h
	WaveNetDynamic.h
	LSTM.h
	LSTMDynamic.h
	InternalModel.h
	TemplateHelper.h)

if(BUILD_NAMCORE)
	set(NAM_SOURCES	../deps/NeuralAmpModelerCore/NAM/activations.h
		../deps/NeuralAmpModelerCore/NAM/activations.cpp
		../deps/NeuralAmpModelerCore/NAM/lstm.h
		../deps/NeuralAmpModelerCore/NAM/lstm.cpp
		../deps/NeuralAmpModelerCore/NAM/dsp.h
		../deps/NeuralAmpModelerCore/NAM/dsp.cpp
		../deps/NeuralAmpModelerCore/NAM/wavenet.cpp
		../deps/NeuralAmpModelerCore/NAM/wavenet.h)
endif()

if(BUILD_STATIC_RTNEURAL)
set(RTNEURAL_WN_SOURCES ../deps/RTNeural-NAM/wavenet/wavenet_layer.hpp
	../deps/RTNeural-NAM/wavenet/wavenet_layer_array.hpp
	../deps/RTNeural-NAM/wavenet/wavenet_model.hpp
	../deps/RTNeural-NAM/wavenet/arena.hpp)
endif()

add_library(NeuralAudio STATIC ${SOURCES} ${NAM_SOURCES} ${RTNEURAL_WN_SOURCES})

target_include_directories(NeuralAudio PUBLIC ..)
target_include_directories(NeuralAudio SYSTEM PUBLIC ../deps/NeuralAmpModelerCore)
target_include_directories(NeuralAudio SYSTEM PUBLIC ../deps/RTNeural)
target_include_directories(NeuralAudio SYSTEM PUBLIC ../deps/math_approx)
target_include_directories(NeuralAudio SYSTEM PUBLIC ../deps/RTNeural-NAM/wavenet)
target_include_directories(NeuralAudio SYSTEM PUBLIC ../deps/RTNeural/modules/Eigen)
target_include_directories(NeuralAudio SYSTEM PUBLIC ../deps/RTNeural/modules/json)

set_property(TARGET NeuralAudio PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../deps/RTNeural RTNeural)
add_subdirectory(../deps/math_approx math_approx)
target_link_libraries(NeuralAudio LINK_PUBLIC RTNeural math_approx)

source_group(NeuralAudio ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})
source_group(RTNeural-NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RTNEURAL_WN_SOURCES})