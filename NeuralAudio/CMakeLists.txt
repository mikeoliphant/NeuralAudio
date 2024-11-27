set(CMAKE_CXX_STANDARD 20)

add_definitions(-DNAM_SAMPLE_FLOAT)
add_definitions(-DDSP_SAMPLE_FLOAT)

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

set(SOURCES
	NeuralModel.h
	NeuralModel.cpp
	NAMModel.h
	RTNeuralModel.h)

set(NAM_SOURCES	../deps/NeuralAmpModelerCore/NAM/activations.h
	../deps/NeuralAmpModelerCore/NAM/activations.cpp
	../deps/NeuralAmpModelerCore/NAM/version.h
	../deps/NeuralAmpModelerCore/NAM/lstm.h
	../deps/NeuralAmpModelerCore/NAM/lstm.cpp
	../deps/NeuralAmpModelerCore/NAM/dsp.h
	../deps/NeuralAmpModelerCore/NAM/dsp.cpp
	../deps/NeuralAmpModelerCore/NAM/get_dsp.cpp
	../deps/NeuralAmpModelerCore/NAM/util.cpp
	../deps/NeuralAmpModelerCore/NAM/util.h
	../deps/NeuralAmpModelerCore/NAM/wavenet.cpp
	../deps/NeuralAmpModelerCore/NAM/wavenet.h
	../deps/NeuralAmpModelerCore/NAM/convnet.cpp
	../deps/NeuralAmpModelerCore/NAM/convnet.h)

set(RTNEURAL_WN_SOURCES ../deps/RTNeural-NAM/wavenet/wavenet_layer.hpp
	../deps/RTNeural-NAM/wavenet/wavenet_layer_array.hpp
	../deps/RTNeural-NAM/wavenet/wavenet_model.hpp
	../deps/RTNeural-NAM/wavenet/arena.hpp)

add_library(NeuralAudio STATIC ${SOURCES} ${NAM_SOURCES} ${RTNEURAL_WN_SOURCES})

target_include_directories(NeuralAudio PUBLIC ..)
target_include_directories(NeuralAudio PUBLIC ../deps/NeuralAmpModelerCore)
target_include_directories(NeuralAudio PUBLIC ../deps/RTNeural)
target_include_directories(NeuralAudio PUBLIC ../deps/RTNeural-NAM/wavenet)
target_include_directories(NeuralAudio PUBLIC ../deps/RTNeural/modules/Eigen)
target_include_directories(NeuralAudio PUBLIC ../deps/RTNeural/modules/json)

set_property(TARGET NeuralAudio PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../deps/RTNeural RTNeural)
target_link_libraries(NeuralAudio LINK_PUBLIC RTNeural)

source_group(NeuralAudio ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})
source_group(RTNeural-NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RTNEURAL_WN_SOURCES})