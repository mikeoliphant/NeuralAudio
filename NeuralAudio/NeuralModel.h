#pragma once

#include <filesystem>

namespace NeuralAudio
{
	class NAMModel;

	class NeuralModel
	{
	public:
		static NeuralModel* CreateFromFile(std::filesystem::path modelPath);

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return 1.0f;
		}

		virtual void Process(float* input, float* output, int numSamples)
		{
		}

	private:
	};
}