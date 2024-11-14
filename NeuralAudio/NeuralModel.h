#pragma once

#include <filesystem>

namespace NeuralAudio
{
	enum ModelLoadMode
	{
		PreferRTNeural,
		PreferNAMCore,
	};

	class NeuralModel
	{
	public:
		static NeuralModel* CreateFromFile(std::filesystem::path modelPath);

		static void SetLSTMLoadMode(ModelLoadMode val)
		{
			lstmLoadMode = val;
		}

		static void SetWaveNetLoadMode(ModelLoadMode val)
		{
			wavenetLoadMode = val;
		}

		virtual float GetRecommendedInputDBAdjustment()
		{
			return 0;
		}

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return 0;
		}

		virtual float GetSampleRate()
		{
			return 48000;
		}

		virtual void Process(float* input, float* output, int numSamples)
		{
		}

		virtual void Prewarm()
		{
		}

	private:
		inline static ModelLoadMode lstmLoadMode = ModelLoadMode::PreferRTNeural;
		inline static ModelLoadMode wavenetLoadMode = ModelLoadMode::PreferRTNeural;
	};
}