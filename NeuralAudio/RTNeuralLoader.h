#pragma once

#ifdef BUILD_RTNEURAL

#include "NeuralModel.h"
#include "NeuralModelImpl.h"

namespace NeuralAudio
{
#ifdef BUILD_STATIC_RTNEURAL
	extern void EnsureRTNeuralModelDefsAreLoaded();
	extern NeuralModelImpl* RTNeuralLoadNAMLSTM(const nlohmann::json& modelJson, NeuralModelLoader* loader);
#endif
	extern NeuralModelImpl* RTNeuralLoadKeras(const nlohmann::json& modelJson, NeuralModelLoader* loader);
}
#endif