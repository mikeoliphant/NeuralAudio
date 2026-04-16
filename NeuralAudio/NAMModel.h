#pragma once

#include "NeuralModel.h"
#include <NAM/activations.h>
#include <NAM/get_dsp.h>
#include <NAM/dsp.h>
#include <NAM/registry.h>

namespace NeuralAudio
{
	class NAMModel : public NeuralModel
	{
	public:
		NAMModel()
		{
			nam::activations::Activation::enable_fast_tanh();
		}

		~NAMModel()
		{
			if (namModel)
				namModel.reset();
		}

		EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::NAMCore;
		}

		bool LoadFromJson(const nlohmann::json& modelJson)
		{
			if (namModel)
				namModel.reset();

			ReadNAMConfig(modelJson);

			namModel = nam::get_dsp(modelJson);

			return true;
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			namModel->process(&input, &output, (int)numSamples);
		}

		void Prewarm()
		{
			namModel->prewarm();
		}

	private:
		std::unique_ptr<nam::DSP> namModel = nullptr;
	};
}