#pragma once

#include <filesystem>
#include <NAM/activations.h>
#include <NAM/dsp.h>

namespace NeuralAudio
{
	class NeuralModel
	{
	public:
		static NeuralModel* CreateFromFile(std::filesystem::path modelPath)
		{
			nam::activations::Activation::enable_fast_tanh();

			NeuralModel model = NeuralModel();
			model.namModel = nam::get_dsp(modelPath);
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