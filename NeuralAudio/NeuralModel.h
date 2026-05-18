#pragma once

#include <filesystem>
#include <istream>
#include "json.hpp"

#ifndef DEFAULT_QUALITY_SCALE
#define DEFAULT_QUALITY_SCALE 1.0
#endif

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
		static NeuralModel* CreateFromStream(std::basic_istream<char>& stream, std::filesystem::path extension);

		virtual ~NeuralModel()
		{
		}

		static bool SetLSTMLoadMode(EModelLoadMode val)
		{
			if (!SupportsLSTMLoadMode(val))
				return false;

			lstmLoadMode = val;

			return true;
		}

		static bool SetWaveNetLoadMode(EModelLoadMode val)
		{
			if (!SupportsWaveNetLoadMode(val))
				return false;

			wavenetLoadMode = val;

			return true;
		}

		static bool SupportsWaveNetLoadMode(EModelLoadMode mode);
		static bool SupportsLSTMLoadMode(EModelLoadMode mode);

		static void SetAudioInputLevelDBu(float audioDBu)
		{
			audioInputLevelDBu = audioDBu;
		}

		static void SetDefaultMaxAudioBufferSize(int maxSize)
		{
			defaultMaxAudioBufferSize = maxSize;
		}

		static void SetDefaultQualityScaleFactor(float scaleFactor)
		{
			defaultQualityScaleFactor = scaleFactor;
		}

		virtual EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::Internal;
		}

		virtual bool HasQualityScaling()
		{
			return false;
		}

		virtual float GetQualityScaleFactor()
		{
			return 1.0f;
		}

		virtual void SetQualityScaleFactor(float scaleFactor)
		{
			(void)scaleFactor;
		}

		virtual bool IsStatic()
		{
			return false;
		}

		virtual void SetMaxAudioBufferSize(const int maxSize)
		{
			(void)maxSize;
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

		virtual int GetReceptiveFieldSize()
		{
			return -1;	// No fixed receptive field size (ie: for LSTM)
		}

		virtual std::string GetModelVersion()
		{
			return modelVersion;
		}

		virtual std::string GetMetadata(const std::string& fieldName)
		{
			auto it = std::find_if(metadata.begin(), metadata.end(), [&](const auto& pair)
			{
				return pair.first == fieldName;
			});

			if (it != metadata.end())
			{
				return it->second;
			}

			return "";
		}

		virtual void Process(float* input, float* output, size_t numSamples)
		{
			(void)input;
			(void)output;
			(void)numSamples;
		}

		virtual void Prewarm()
		{
		}

	protected:
		void ReadNAMConfig(const nlohmann::json& modelJson);
		void ReadKerasConfig(const nlohmann::json& modelJson);

		float modelInputLevelDBu = 12;
		float modelOutputLevelDBu = 12;
		float modelLoudnessDB = -18;
		float sampleRate = 48000;
		std::string modelVersion = "";
		std::vector<std::pair<std::string, std::string>> metadata;

		inline static float audioInputLevelDBu = 12;
		inline static EModelLoadMode lstmLoadMode = EModelLoadMode::Internal;
		inline static EModelLoadMode wavenetLoadMode = EModelLoadMode::Internal;
		inline static int defaultMaxAudioBufferSize = 128;
		inline static float defaultQualityScaleFactor = DEFAULT_QUALITY_SCALE;

		void AddMetadata(std::string fieldName, std::string fieldValue)
		{
			metadata.push_back({fieldName, fieldValue});
		}

		void AddMetadata(const nlohmann::json& metadataJson)
		{
			for (auto& [key, value] : metadataJson.items())
			{
				if (!value.is_null())
				{
					AddMetadata(key, value.dump());
				}
			}
		}

		void Prewarm(size_t numSamples, size_t blockSize)
		{
			std::vector<float> input;
			input.resize(blockSize);
			std::fill(input.begin(), input.end(), 0.0f);

			std::vector<float> output;
			output.resize(blockSize);

			for (size_t block = 0; block < (numSamples / blockSize); block++)
			{
				Process(input.data(), output.data(), blockSize);
			}
		}
	};
}