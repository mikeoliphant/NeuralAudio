#pragma once

#include <filesystem>
#include <istream>
#include <algorithm>
#include <string>
#include <vector>
#include "json.hpp"

#ifndef DEFAULT_QUALITY_SCALE
#define DEFAULT_QUALITY_SCALE 1.0
#endif

#ifndef DEFAULT_INPUT_DBU
#define DEFAULT_INPUT_DBU 12
#endif

namespace NeuralAudio
{
	enum EModelLoadMode
	{
		Internal,
		RTNeural,
		NAMCore
	};

	enum ECompositeModelLoadMode
	{
		LoadAll,
		OnDemand
	};

	class NeuralModel
	{
	public:
		virtual ~NeuralModel()
		{
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

		virtual bool IsQualityChangeRealtimeSafe(float newScaleFactor)
		{
			(void)newScaleFactor;

			return true;
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

		virtual bool HasLoudness() const
		{
			return hasLoudnessKnown_;
		}

		virtual bool HasInputLevel() const
		{
			return hasInputLevelKnown_;
		}

		virtual bool HasOutputLevel() const
		{
			return hasOutputLevelKnown_;
		}

		virtual float GetLoudnessDB() const
		{
			return modelLoudnessDB;
		}

		virtual float GetInputLevelDBu() const
		{
			return modelInputLevelDBu;
		}

		virtual float GetOutputLevelDBu() const
		{
			return modelOutputLevelDBu;
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
		float audioInputLevelDBu = 12;
		float modelInputLevelDBu = 12;
		float modelOutputLevelDBu = 12;
		float modelLoudnessDB = -18;
		float sampleRate = 48000;
		std::string modelVersion = "";
		std::vector<std::pair<std::string, std::string>> metadata;
		bool hasLoudnessKnown_ = false;
		bool hasInputLevelKnown_ = false;
		bool hasOutputLevelKnown_ = false;
	};

	class NeuralModelLoader
	{
		public:
			NeuralModel* CreateFromFile(const std::filesystem::path& modelPath, bool doPrewarm = true);
			NeuralModel* CreateFromStream(std::basic_istream<char>& stream, const std::filesystem::path& extension, bool doPrewarm = true);
			NeuralModel* CreateFromJson(nlohmann::json& modelJson, const std::filesystem::path& extension, bool doPrewarm = true);

			bool SetLSTMLoadMode(EModelLoadMode val)
			{
				if (!SupportsLSTMLoadMode(val))
					return false;

				lstmLoadMode = val;

				return true;
			}

			bool SetWaveNetLoadMode(EModelLoadMode val)
			{
				if (!SupportsWaveNetLoadMode(val))
					return false;

				wavenetLoadMode = val;

				return true;
			}

			ECompositeModelLoadMode GetCompositeModelLoadMode()
			{
				return compositeLoadMode;
			}

			void SetCompositeModelLoadMode(ECompositeModelLoadMode loadMode)
			{
				compositeLoadMode = loadMode;
			}

			bool SupportsWaveNetLoadMode(EModelLoadMode mode);
			bool SupportsLSTMLoadMode(EModelLoadMode mode);

			void SetAudioInputLevelDBu(float audioDBu)
			{
				audioInputLevelDBu = audioDBu;
			}

			float GetAudioInputLevelDBu()
			{
				return audioInputLevelDBu;
			}

			void SetDefaultMaxAudioBufferSize(int maxSize)
			{
				defaultMaxAudioBufferSize = maxSize;
			}

			int GetDefaultMaxAudioBufferSize()
			{
				return defaultMaxAudioBufferSize;
			}

			void SetDefaultQualityScaleFactor(float scaleFactor)
			{
				defaultQualityScaleFactor = scaleFactor;
			}

			float GetDefaultQualityScaleFactor()
			{
				return defaultQualityScaleFactor;
			}

			void SetExternalSampleRate(int sampleRate)
			{
				this->externalSampleRate = sampleRate;
			}

		protected:
			EModelLoadMode lstmLoadMode = EModelLoadMode::Internal;
			EModelLoadMode wavenetLoadMode = EModelLoadMode::Internal;
			ECompositeModelLoadMode compositeLoadMode = ECompositeModelLoadMode::LoadAll;
			float audioInputLevelDBu = (float)DEFAULT_INPUT_DBU;
			int defaultMaxAudioBufferSize = 128;
			float defaultQualityScaleFactor = (float)DEFAULT_QUALITY_SCALE;
			int externalSampleRate = 48000;
	};

}