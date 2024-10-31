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

		virtual void Process(std::vector<float> input, std::vector<float> output)
		{
		}

	private:
		NAMModel* namModel;
	};
}