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

set(LSTM_MATH "FastMath" CACHE STRING "LSTM math functions")
add_definitions(-DLSTM_MATH=${LSTM_MATH})
message(STATUS "LSTM math is: ${LSTM_MATH}")

set(WAVENET_MATH "FastMath" CACHE STRING "WaveNet math functions")
add_definitions(-DWAVENET_MATH=${WAVENET_MATH})
message(STATUS "WaveNet math is: ${WAVENET_MATH}")

set(WAVENET_FRAMES "64" CACHE STRING "WaveNet frame size")

add_definitions(-DWAVENET_MAX_NUM_FRAMES=${WAVENET_FRAMES})

message(STATUS "WaveNet frame size is: ${WAVENET_FRAMES}")

set(BUFFER_PADDING "24" CACHE STRING "Convoluation buffer padding size")

add_definitions(-DLAYER_ARRAY_BUFFER_PADDING=${BUFFER_PADDING})

message(STATUS "Convoluation buffer padding size: ${BUFFER_PADDING}")

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

set(SOURCES
	NeuralModel.h
	NeuralModel.cpp
	NAMModel.h
	RTNeuralModel.h
	RTNeuralLoader.cpp
	RTNeuralLoader.h
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
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/NeuralAmpModelerCore)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/math_approx)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural-NAM/wavenet)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural/modules/Eigen)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural/modules/json)

set_property(TARGET NeuralAudio PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../deps/RTNeural RTNeural)
add_subdirectory(../deps/math_approx math_approx)
target_link_libraries(NeuralAudio LINK_PUBLIC RTNeural math_approx)

source_group(NeuralAudio ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})
source_group(RTNeural-NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RTNEURAL_WN_SOURCES})