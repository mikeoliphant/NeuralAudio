include_directories(SYSTEM ../deps/RTNeural/modules/Eigen)
include_directories(SYSTEM ../deps/json)

set(SOURCES
	NeuralModel.h)

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

add_library(NeuralAudio STATIC ${SOURCES} ${NAM_SOURCES})

source_group(NeuralAudio ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})