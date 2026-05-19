#pragma once

#include "json.hpp"
#include "NeuralModel.h"

namespace NeuralAudio
{
	class NeuralModelImpl : public NeuralModel
	{
		protected:
			void ReadNAMConfig(const nlohmann::json& modelJson)
			{
				modelVersion = modelJson.at("version");

				if (modelJson.contains("metadata"))
				{
					nlohmann::json metadataJson = modelJson.at("metadata");

					AddMetadata(metadataJson);
				}

				if (modelJson.contains("sample_rate") && modelJson.at("sample_rate").is_number())
				{
					sampleRate = modelJson.at("sample_rate");
				}

				if (modelJson.contains("metadata"))
				{
					nlohmann::json metaData = modelJson.at("metadata");

					if (metaData.contains("loudness") && metaData.at("loudness").is_number())
					{
						modelLoudnessDB = (float)metaData.at("loudness");
					}

					if (metaData.contains("input_level_dbu") && metaData.at("input_level_dbu").is_number())
					{
						modelInputLevelDBu = metaData.at("input_level_dbu");
					}

					if (metaData.contains("output_level_dbu") && metaData.at("output_level_dbu").is_number())
					{
						modelOutputLevelDBu = metaData.at("output_level_dbu");
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
	};
}