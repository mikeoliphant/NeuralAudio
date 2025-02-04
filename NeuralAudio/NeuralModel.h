#pragma once

#include <filesystem>
#include "json.hpp"

namespace NeuralAudio
{
	enum EModelLoadMode
	{
		Internal,
		RTNeural,
		NAMCore
	};

	class NeuralModel
	{
	public:
		static NeuralModel* CreateFromFile(std::filesystem::path modelPath);

		virtual ~NeuralModel()
		{
		}

		static void SetLSTMLoadMode(EModelLoadMode val)
		{
			lstmLoadMode = val;
		}

		static void SetWaveNetLoadMode(EModelLoadMode val)
		{
			wavenetLoadMode = val;
		}

		static void SetAudioInputLevelDBu(float audioDBu)
		{
			audioInputLevelDBu = audioDBu;
		}

		static void SetDefaultMaxAudioBufferSize(int maxSize)
		{
			defaultMaxAudioBufferSize = maxSize;
		}

		virtual EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::Internal;
		}

		virtual void SetMaxAudioBufferSize(int maxSize)
		{
		}

		virtual float GetRecommendedInputDBAdjustment()
		{
			return audioInputLevelDBu - modelInputLevelDBu;
		}

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return -18 - modelLoudnessDB;
		}

		virtual float GetSampleRate()
		{
			return sampleRate;
		}

		virtual void Process(float* input, float* output, int numSamples)
		{
		}

		virtual void Prewarm()
		{
		}

	protected:
		void ReadNAMConfig(nlohmann::json& modelJson);
		void ReadKerasConfig(nlohmann::json& modelJson);

		float modelInputLevelDBu = 12;
		float modelOutputLevelDBu = 12;
		float modelLoudnessDB = -18;
		float sampleRate = 48000;

		inline static float audioInputLevelDBu = 12;
		inline static EModelLoadMode lstmLoadMode = EModelLoadMode::Internal;
		inline static EModelLoadMode wavenetLoadMode = EModelLoadMode::Internal;
		inline static int defaultMaxAudioBufferSize = 128;
	};
}