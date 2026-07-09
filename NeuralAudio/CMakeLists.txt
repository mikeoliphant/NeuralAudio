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

option(BUILD_NAMCORE "Build NAM Core" ON)
if(BUILD_NAMCORE)
    message(STATUS "Building NAM Core implementation")
    add_definitions(-DBUILD_NAMCORE)
else()
    message(STATUS "NOT Building NAM Core implementation")
endif()

option(NAM_USE_INLINE_GEMM "Use inline matrix multiplications in NAM Core" OFF)
if(NAM_USE_INLINE_GEMM)
    message(STATUS "Using NAM Core inline matrix multiplications")
    add_definitions(-DNAM_USE_INLINE_GEMM)
else()
    message(STATUS "NOT using NAM Core inline matrix multiplications")
endif()

option(NAM_ENABLE_A2_FAST "Use NAM A2 fast-path WaveNet" ON)
if(NAM_ENABLE_A2_FAST)
    message(STATUS "Using NAM A2 fast-path WaveNet")
    add_definitions(-DNAM_ENABLE_A2_FAST)
else()
    message(STATUS "NOT using NAM A2 fast-path WaveNet")
endif()

option(BUILD_INTERNAL_STATIC_WAVENET "Build Internal static WaveNet models" ON)
if(BUILD_INTERNAL_STATIC_WAVENET)
    message(STATUS "Building Internal static WaveNet models")
    add_definitions(-DBUILD_INTERNAL_STATIC_WAVENET)
else()
    message(STATUS "NOT building Internal static WaveNet models")
endif()

option(BUILD_STATIC_INTERNAL_NAMA2 "Build Internal static A2 WaveNet models" ON)
if(BUILD_STATIC_INTERNAL_NAMA2)
    message(STATUS "Building Internal static A2 WaveNet models")
    add_definitions(-DBUILD_STATIC_INTERNAL_NAMA2)
else()
    message(STATUS "NOT building Internal static A2 WaveNet models")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "15.0")
	option(ENABLE_MULTIFRAME_8X8_CONVOLUTION "Enable multi-frame 8x8 convolution" ON)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL  "21.0")
	option(ENABLE_MULTIFRAME_8X8_CONVOLUTION "Enable multi-frame 8x8 convolution" ON)
else()
	option(ENABLE_MULTIFRAME_8X8_CONVOLUTION "Enable multi-frame 8x8 convolution" OFF)
endif()

if(ENABLE_MULTIFRAME_8X8_CONVOLUTION)
    message(STATUS "Using multi-frame 8x8 convolution")
    add_definitions(-DENABLE_MULTIFRAME_8X8_CONVOLUTION)
else()
    message(STATUS "NOT using multi-frame 8x8 convolution")
endif()

set(WAVENET_MATH "FastMath" CACHE STRING "WaveNet math functions")
add_definitions(-DWAVENET_MATH=${WAVENET_MATH})
message(STATUS "WaveNet math is: ${WAVENET_MATH}")

option(BUILD_INTERNAL_STATIC_LSTM "Build Internal static LSTM models" OFF)
if(BUILD_INTERNAL_STATIC_LSTM)
    message(STATUS "Building Internal static LSTM models")
    add_definitions(-DBUILD_INTERNAL_STATIC_LSTM)
else()
    message(STATUS "NOT building Internal static LSTM models")
endif()

set(LSTM_MATH "FastMath" CACHE STRING "LSTM math functions")
add_definitions(-DLSTM_MATH=${LSTM_MATH})
message(STATUS "LSTM math is: ${LSTM_MATH}")

set(DEFAULT_QUALITY_SCALE "1.0" CACHE STRING "Default model quality scale factor")

add_definitions(-DDEFAULT_QUALITY_SCALE=${DEFAULT_QUALITY_SCALE})

message(STATUS "Default model quality scale factor is: ${DEFAULT_QUALITY_SCALE}")

set(DEFAULT_INPUT_DBU "12" CACHE STRING "Default dBu level for input scaling")

add_definitions(-DDEFAULT_INPUT_DBU=${DEFAULT_INPUT_DBU})

message(STATUS "Default input dBu is: ${DEFAULT_INPUT_DBU}")

set(WAVENET_FRAMES "64" CACHE STRING "WaveNet frame size")

add_definitions(-DWAVENET_MAX_NUM_FRAMES=${WAVENET_FRAMES})

message(STATUS "WaveNet frame size is: ${WAVENET_FRAMES}")

set(BUFFER_PADDING "24" CACHE STRING "Convolution buffer padding size")

add_definitions(-DLAYER_ARRAY_BUFFER_PADDING=${BUFFER_PADDING})

message(STATUS "Convolution buffer padding size: ${BUFFER_PADDING}")

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

set(SOURCES
	NeuralModel.h
	NeuralModel.cpp
	NeuralModelImpl.h
	NAMModel.h
	RTNeuralModel.h
	RTNeuralLoader.cpp
	RTNeuralLoader.h
	Activation.h
	MatMul.h
	WaveNet.h
	WaveNetDynamic.h
	LSTM.h
	LSTMDynamic.h
	InternalModel.h
	CompositeModel.h
	TemplateHelper.h)

if(BUILD_NAMCORE)
	set(NAM_SOURCES	../deps/NeuralAmpModelerCore/NAM/activations.h
		../deps/NeuralAmpModelerCore/NAM/activations.cpp
		../deps/NeuralAmpModelerCore/NAM/conv1d.cpp
		../deps/NeuralAmpModelerCore/NAM/conv1d.h
		../deps/NeuralAmpModelerCore/NAM/film.h 
		../deps/NeuralAmpModelerCore/NAM/gating_activations.h
		../deps/NeuralAmpModelerCore/NAM/get_dsp.h
		../deps/NeuralAmpModelerCore/NAM/get_dsp.cpp
		../deps/NeuralAmpModelerCore/NAM/lstm.h
		../deps/NeuralAmpModelerCore/NAM/registry.h
		../deps/NeuralAmpModelerCore/NAM/ring_buffer.h
		../deps/NeuralAmpModelerCore/NAM/ring_buffer.cpp
		../deps/NeuralAmpModelerCore/NAM/slimmable.h
		../deps/NeuralAmpModelerCore/NAM/lstm.cpp
		../deps/NeuralAmpModelerCore/NAM/dsp.h
		../deps/NeuralAmpModelerCore/NAM/dsp.cpp
		../deps/NeuralAmpModelerCore/NAM/container.cpp
		../deps/NeuralAmpModelerCore/NAM/container.h
		../deps/NeuralAmpModelerCore/NAM/wavenet/slimmable.cpp
		../deps/NeuralAmpModelerCore/NAM/wavenet/detail.h
		../deps/NeuralAmpModelerCore/NAM/wavenet/params.h
		../deps/NeuralAmpModelerCore/NAM/wavenet/model.h
		../deps/NeuralAmpModelerCore/NAM/wavenet/model.cpp
		../deps/NeuralAmpModelerCore/NAM/wavenet/a2_fast.h
		../deps/NeuralAmpModelerCore/NAM/wavenet/a2_fast.cpp)
endif()

add_library(NeuralAudio OBJECT ${SOURCES} ${NAM_SOURCES})

target_include_directories(NeuralAudio PUBLIC ..)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/NeuralAmpModelerCore)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/math_approx)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural/modules/Eigen)
target_include_directories(NeuralAudio SYSTEM PRIVATE ../deps/RTNeural/modules/json)

set_property(TARGET NeuralAudio PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../deps/RTNeural RTNeural)
add_subdirectory(../deps/math_approx math_approx)
target_link_libraries(NeuralAudio LINK_PUBLIC RTNeural math_approx)

source_group(NeuralAudio ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})
source_group(RTNeural-NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RTNEURAL_WN_SOURCES})
