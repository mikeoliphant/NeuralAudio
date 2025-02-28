#include <list>
#include "NeuralModel.h"
#ifdef BUILD_NAMCORE
#include "NAMModel.h"
#endif
#ifdef BUILD_STATIC_RTNEURAL
#include "RTNeuralLoader.h"
#endif
#include "RTNeuralModel.h"
#include "InternalModel.h"

namespace NeuralAudio
{
	static bool modelDefsAreLoaded;

	static std::list<InternalWaveNetDefinitionBase*> internalWavenetModelDefs;
	static std::list<InternalLSTMDefinitionBase*> internalLSTMModelDefs;

	static void EnsureModelDefsAreLoaded()
	{
		if (!modelDefsAreLoaded)
		{
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<16, 8>);	// Standard
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<12, 6>);	// Lite
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<8, 4>);	// Feather
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<4, 2>);	// Nano

			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 8>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 12>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 16>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 24>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<2, 8>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<2, 12>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<2, 16>);

#ifdef BUILD_STATIC_RTNEURAL
			EnsureRTNeuralModelDefsAreLoaded();
#endif

			modelDefsAreLoaded = true;
		}
	}

	static InternalWaveNetDefinitionBase* FindInternalWaveNetDefinition(size_t numChannels, size_t headSize)
	{
		for (auto const& model : internalWavenetModelDefs)
		{
			if ((numChannels == model->GetNumChannels()) && (headSize == model->GetHeadSize()))
				return model;
		}

		return nullptr;
	}

	static InternalLSTMDefinitionBase* FindInternalLSTMDefinition(size_t numLayers, size_t hiddenSize)
	{
		for (auto const& model : internalLSTMModelDefs)
		{
			if ((numLayers == model->GetNumLayers()) && (hiddenSize == model->GetHiddenSize()))
				return model;
		}

		return nullptr;
	}

	static std::vector<int> stdDilations = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
	static std::vector<int> liteDilations = { 1, 2, 4, 8, 16, 32, 64 };
	static std::vector<int> liteDilations2 = { 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

	static bool CheckDilations(const nlohmann::json dilationJson, std::vector<int>& checkDilations)
	{
		if (dilationJson.size() != checkDilations.size())
			return false;

		for (size_t i = 0; i < dilationJson.size(); i++)
		{
			if (dilationJson[i] != checkDilations[i])
				return false;
		}

		return true;
	}

	NeuralModel* NeuralModel::CreateFromFile(std::filesystem::path modelPath)
	{
		std::ifstream jsonStream(modelPath, std::ifstream::binary);

		return CreateFromStream(jsonStream, modelPath.extension());
	}

	NeuralModel* NeuralModel::CreateFromStream(std::basic_istream<char>& jsonStream, std::filesystem::path extension)
	{
		EnsureModelDefsAreLoaded();

		nlohmann::json modelJson;
		jsonStream >> modelJson;

		NeuralModel* newModel = nullptr;

		if (extension == ".nam")
		{
			std::string arch = modelJson["architecture"];

#ifdef BUILD_NAMCORE
			if (wavenetLoadMode == EModelLoadMode::NAMCore)
			{
				NAMModel* model = new NAMModel;

				model->LoadFromJson(modelJson);

				newModel = model;
			}
#endif

			if (newModel == nullptr)
			{
				if (arch == "WaveNet")
				{
					nlohmann::json config = modelJson["config"];

					if (config["layers"].size() == 2)
					{
						nlohmann::json firstLayerConfig = config["layers"][0];
						nlohmann::json secondLayerConfig = config["layers"][1];

						if (!firstLayerConfig["gated"] && !secondLayerConfig["gated"] && !firstLayerConfig["head_bias"] && secondLayerConfig["head_bias"])
						{
							bool isOfficialArchitecture = false;

							if (firstLayerConfig["channels"] == 16)
							{
								if (CheckDilations(firstLayerConfig["dilations"], stdDilations) && CheckDilations(secondLayerConfig["dilations"], stdDilations))
								{
									isOfficialArchitecture = true;
								}
							}
							else
							{
								if (CheckDilations(firstLayerConfig["dilations"], liteDilations) && CheckDilations(secondLayerConfig["dilations"], liteDilations2))
								{
									isOfficialArchitecture = true;
								}
							}

							if (isOfficialArchitecture)
							{
								if (wavenetLoadMode == EModelLoadMode::RTNeural)
								{
#ifdef BUILD_STATIC_RTNEURAL
									newModel = RTNeuralLoadNAMWaveNet(modelJson);
#endif
								}

								if (newModel == nullptr)
								{
									auto modelDef = FindInternalWaveNetDefinition(firstLayerConfig["channels"], firstLayerConfig["head_size"]);

									if (modelDef != nullptr)
									{
										auto model = modelDef->CreateModel();

										model->LoadFromNAMJson(modelJson);

										newModel = model;
									}
								}
							}
						}
					}

					if (newModel == nullptr)
					{
						// Use a dynamic model if we had no static definition
						InternalWaveNetModelDyn* model = new InternalWaveNetModelDyn;

						if (model->LoadFromNAMJson(modelJson))
						{
							newModel = model;
						}
					}
				}
				else if (arch == "LSTM")
				{
					nlohmann::json config = modelJson["config"];

#ifdef BUILD_STATIC_RTNEURAL
					if (lstmLoadMode == EModelLoadMode::RTNeural)
					{
						newModel = RTNeuralLoadNAMLSTM(modelJson);
					}
#endif

					if (newModel == nullptr)
					{
						auto modelDef = FindInternalLSTMDefinition(config["num_layers"], config["hidden_size"]);

						if (modelDef != nullptr)
						{
							auto model = modelDef->CreateModel();
							model->LoadFromNAMJson(modelJson);

							newModel = model;
						}

						// Use a dynamic model if we had no static definition
						if (newModel == nullptr)
						{
							InternalLSTMModelDyn* model = new InternalLSTMModelDyn;

							if (model->LoadFromNAMJson(modelJson))
							{
								newModel = model;
							}
						}
					}
				}
			}
		}
		else if ((extension == ".json") || (extension == ".aidax"))
		{
			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size() - 1;
			const std::string modelType = layers.at(0).at("type");
			const int hiddenSize = layers.at(0).at("shape").back();

			if (modelType == "lstm")
			{
#ifdef BUILD_STATIC_RTNEURAL
				if (lstmLoadMode == EModelLoadMode::RTNeural)
				{
					newModel = RTNeuralLoadKeras(modelJson);
				}
#endif

				if (newModel == nullptr && lstmLoadMode == EModelLoadMode::Internal)
				{
					if (numLayers == 1)
					{
						auto modelDef = FindInternalLSTMDefinition(numLayers, hiddenSize);

						if (modelDef != nullptr)
						{
							auto model = modelDef->CreateModel();

							if (model->LoadFromKerasJson(modelJson))
							{
								newModel = model;
							}
						}
					}

					if (newModel == nullptr)
					{
						// Use a dynamic model if we had no static definition
						InternalLSTMModelDyn* model = new InternalLSTMModelDyn;

						if (model->LoadFromKerasJson(modelJson))
						{
							newModel = model;
						}
					}
				}
			}
	
			if (newModel == nullptr)
			{
				// Use a dynamic model for other model types
				RTNeuralModelDyn* model = new RTNeuralModelDyn;
				model->LoadFromKerasJson(modelJson);

				newModel = model;
			}
		}

		if (newModel != nullptr)
		{
			newModel->Prewarm();
		}

		return newModel;
	}

	void NeuralModel::ReadNAMConfig(const nlohmann::json& modelJson)
	{
		if (modelJson.contains("samplerate") && modelJson["samplerate"].is_number_float())
		{
			sampleRate = modelJson["samplerate"];
		}

		if (modelJson.contains("sample_rate") && modelJson["sample_rate"].is_number_float())
		{
			sampleRate = modelJson["sample_rate"];
		}

		if (modelJson.contains("metadata"))
		{
			nlohmann::json metaData = modelJson["metadata"];

			if (metaData.contains("loudness") && metaData["loudness"].is_number_float())
			{
				modelLoudnessDB = (float)metaData["loudness"];
			}

			if (metaData.contains("input_level_dbu") && metaData["input_level_dbu"].is_number_float())
			{
				modelInputLevelDBu = metaData["input_level_dbu"];
			}

			if (metaData.contains("output_level_dbu") && metaData["output_level_dbu"].is_number_float())
			{
				modelOutputLevelDBu = metaData["output_level_dbu"];
			}
		}
	}

	void NeuralModel::ReadKerasConfig(const nlohmann::json& modelJson)
	{
		if (modelJson.contains("samplerate") && modelJson["samplerate"].is_number_float())
		{
			sampleRate = modelJson["samplerate"];
		}

		if (modelJson.contains("in_gain") && modelJson["in_gain"].is_number_float())
		{
			modelInputLevelDBu = modelJson["in_gain"];
		}

		if (modelJson.contains("out_gain") && modelJson["out_gain"].is_number_float())
		{
			modelLoudnessDB = -18 - (float)modelJson["out_gain"];
		}
	}
}