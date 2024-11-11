#include <list>
#include "NeuralModel.h"
#include "NAMModel.h"
#include "RTNeuralModel.h"

namespace NeuralAudio
{
	static bool modelDefsAreLoaded;

	static std::list<RTNeuralLSTMDefinitionBase*> lstmModelDefs;
	static std::list<RTNeuralWaveNetDefinitionBase*> wavenetModelDefs;

	static void EnsureModelDefsAreLoaded()
	{
		if (!modelDefsAreLoaded)
		{
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 8>);
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 12>);
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 16>);
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 24>);
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 8>);
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 12>);
			lstmModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 16>);

			wavenetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<16, 8>);	// Standard
			wavenetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<12, 6>);	// Lite
			wavenetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<8, 4>);	// Feather
			wavenetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<4, 2>);	// Nano

			modelDefsAreLoaded = true;
		}
	}

	static RTNeuralLSTMDefinitionBase* FindLSTMDefinition(size_t numLayers, size_t hiddenSize)
	{
		for (auto const& model : lstmModelDefs)
		{
			if ((numLayers == model->GetNumLayers()) && (hiddenSize == model->GetHiddenSize()))
				return model;
		}

		return nullptr;
	}

	static RTNeuralWaveNetDefinitionBase* FindWaveNetDefinition(size_t numChannels, size_t headSize)
	{
		for (auto const& model : wavenetModelDefs)
		{
			if ((numChannels == model->GetNumChannels()) && (headSize == model->GetHeadSize()))
				return model;
		}

		return nullptr;
	}

	static std::vector<int> stdDilations = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
	static std::vector<int> liteDilations = { 1, 2, 4, 8, 16, 32, 64 };
	static std::vector<int> liteDilations2 = { 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

	static bool CheckDilations(nlohmann::json dilationJson, std::vector<int>& checkDilations)
	{
		if (dilationJson.size() != checkDilations.size())
			return false;

		for (int i = 0; i < dilationJson.size(); i++)
		{
			if (dilationJson[i] != checkDilations[i])
				return false;
		}
	}

	NeuralModel* NeuralModel::CreateFromFile(std::filesystem::path modelPath)
	{
		EnsureModelDefsAreLoaded();

		std::ifstream jsonStream(modelPath, std::ifstream::binary);

		nlohmann::json modelJson;
		jsonStream >> modelJson;

		NeuralModel* newModel = nullptr;

		if (modelPath.extension() == ".nam")
		{
			std::string arch = modelJson["architecture"];

			if (!preferNAM)
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
								auto modelDef = FindWaveNetDefinition(firstLayerConfig["channels"], firstLayerConfig["head_size"]);

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
				else if (arch == "LSTM")
				{
					nlohmann::json config = modelJson["config"];

					auto modelDef = FindLSTMDefinition(config["num_layers"], config["hidden_size"]);

					if (modelDef != nullptr)
					{
						RTNeuralModel* model = modelDef->CreateModel();
						model->LoadFromNAMJson(modelJson);

						newModel = model;
					}
					else
					{
						RTNeuralModelDyn* model = new RTNeuralModelDyn;
						model->LoadFromNAMJson(modelJson);

						newModel = model;
					}
				}
			}

			// If we couldn't load the model using RTNeural, use NAM core
			if (newModel == nullptr)
			{
				NAMModel* model = new NAMModel;

				model->LoadFromJson(modelJson);

				newModel = model;
			}
		}
		else if (modelPath.extension() == ".json")
		{
			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size() - 1;
			const std::string modelType = layers.at(0).at("type");
			const int hidden_size = layers.at(0).at("shape").back();

			if (modelType == "lstm")
			{
				auto modelDef = FindLSTMDefinition(numLayers, hidden_size);

				if (modelDef != nullptr)
				{
					RTNeuralModel* model = modelDef->CreateModel();

					model->LoadFromKerasJson(modelJson);

					newModel = model;
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
}