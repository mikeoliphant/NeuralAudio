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

		void Process(std::vector<float> input, std::vector<float> output)
		{
			namModel->process(input.data(), output.data(), input.size());
		}

	private:
		std::unique_ptr<nam::DSP> namModel;
	};
}