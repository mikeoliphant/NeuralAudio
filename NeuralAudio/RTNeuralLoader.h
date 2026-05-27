#pragma once

#include "NeuralModel.h"
#include "NeuralModelImpl.h"

namespace NeuralAudio
{
	extern void EnsureRTNeuralModelDefsAreLoaded();
	extern NeuralModel* RTNeuralLoadNAMWaveNet(const nlohmann::json& modelJson, NeuralModelLoader* loader);
	extern NeuralModel* RTNeuralLoadNAMLSTM(const nlohmann::json& modelJson, NeuralModelLoader* loader);
	extern NeuralModel* RTNeuralLoadKeras(const nlohmann::json& modelJson, NeuralModelLoader* loader);
}