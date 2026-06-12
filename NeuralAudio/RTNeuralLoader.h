#pragma once

#include "NeuralModel.h"
#include "NeuralModelImpl.h"

namespace NeuralAudio
{
	extern void EnsureRTNeuralModelDefsAreLoaded();
	extern NeuralModelImpl* RTNeuralLoadNAMWaveNet(const nlohmann::json& modelJson, NeuralModelLoader* loader);
	extern NeuralModelImpl* RTNeuralLoadNAMLSTM(const nlohmann::json& modelJson, NeuralModelLoader* loader);
	extern NeuralModelImpl* RTNeuralLoadKeras(const nlohmann::json& modelJson, NeuralModelLoader* loader);
}