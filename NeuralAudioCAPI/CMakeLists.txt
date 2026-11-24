set(CMAKE_CXX_STANDARD 20)

set(SOURCES NeuralAudioCApi.h NeuralAudioCApi.cpp)
	
add_library(NeuralAudioCAPI SHARED ${SOURCES})

include_directories(NeuralAudioCAPI ../NeuralAudio)

target_link_libraries(NeuralAudioCAPI PRIVATE NeuralAudio)

source_group(NeuralAudioCAPI FILES ${SOURCES})
