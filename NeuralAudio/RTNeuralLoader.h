#pragma once

#include "NeuralModel.h"

namespace NeuralAudio
{
	extern void EnsureRTNeuralModelDefsAreLoaded();
	extern NeuralModel* RTNeuralLoadNAMWaveNet(const nlohmann::json& modelJson);
	extern NeuralModel* RTNeuralLoadNAMLSTM(const nlohmann::json& modelJson);
	extern NeuralModel* RTNeuralLoadKeras(const nlohmann::json& modelJson);
}