#pragma once

#include <filesystem>

namespace NeuralAudio
{
	class NeuralModel
	{
	public:
		static NeuralModel* CreateFromFile(std::filesystem::path modelPath);
		static void SetPreferNAM(bool val)
		{
			preferNAM = val;
		}

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return 0;
		}

		virtual void Process(float* input, float* output, int numSamples)
		{
		}

	private:
		inline static bool preferNAM = false;
	};
}