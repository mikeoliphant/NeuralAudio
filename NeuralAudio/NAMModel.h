#pragma once

#include "NeuralModel.h"
#include <NAM/activations.h>
#include <NAM/dsp.h>

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

		bool LoadFromFile(std::filesystem::path modelPath)
		{
			namModel = nam::get_dsp(modelPath);

			return true;
		}

		float GetRecommendedOutputDBAdjustment()
		{
			if (namModel->HasLoudness())
			{
				return namModel->GetLoudness();
			}
			else
			{
				return 1.0f;
			}
		}

		void Process(float* input, float* output, int numSamples)
		{
			namModel->process(input, output, numSamples);
		}

	private:
		std::unique_ptr<nam::DSP> namModel = nullptr;
	};
}