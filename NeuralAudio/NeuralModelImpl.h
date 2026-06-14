#pragma once

#include "NeuralModel.h"

namespace NeuralAudio
{
	class NeuralModelImpl : public NeuralModel
	{
		public:
			using NeuralModel::Prewarm;

			void SetModelLoader(NeuralModelLoader* modelLoader)
			{
				this->loader = modelLoader;
			}

			bool HadInitialPrewarm()
			{
				return hadInitialPrewarm;
			}

			void SetHadInitialPrewarm()
			{
				hadInitialPrewarm = true;
			}

		protected:
			void ReadNAMConfig(const nlohmann::json& modelJson)
			{
				modelVersion = modelJson.at("version");

				if (modelJson.contains("sample_rate") && modelJson.at("sample_rate").is_number())
				{
					sampleRate = modelJson.at("sample_rate");
				}

				if (modelJson.contains("metadata"))
				{
					auto& metadataJson = modelJson.at("metadata");

					AddMetadata(metadataJson);

					if (metadataJson.contains("loudness") && metadataJson.at("loudness").is_number())
					{
						modelLoudnessDB = (float)metadataJson.at("loudness");
					}

					if (metadataJson.contains("input_level_dbu") && metadataJson.at("input_level_dbu").is_number())
					{
						modelInputLevelDBu = metadataJson.at("input_level_dbu");
					}

					if (metadataJson.contains("output_level_dbu") && metadataJson.at("output_level_dbu").is_number())
					{
						modelOutputLevelDBu = metadataJson.at("output_level_dbu");
					}
				}
			}

			void ReadKerasConfig(const nlohmann::json& modelJson)
			{
				if (modelJson.contains("samplerate") && modelJson.at("samplerate").is_number())
				{
					sampleRate = modelJson.at("samplerate");
				}

				if (modelJson.contains("in_gain") && modelJson.at("in_gain").is_number())
				{
					modelInputLevelDBu = modelJson.at("in_gain");
				}

				if (modelJson.contains("out_gain") && modelJson.at("out_gain").is_number())
				{
					modelLoudnessDB = -18 - (float)modelJson.at("out_gain");
				}
			}

			void AddMetadata(std::string fieldName, std::string fieldValue)
			{
				metadata.push_back({ fieldName, fieldValue });
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

			NeuralModelLoader *loader;
			bool hadInitialPrewarm = false;
	};
}