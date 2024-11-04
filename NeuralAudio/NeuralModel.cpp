#include <list>
#include "NeuralModel.h"
#include "NAMModel.h"
#include "RTNeuralModel.h"

namespace NeuralAudio
{
	static std::list<RTNeuralModelDefinitionBase*> modelDefs;

	static bool modelDefsAreLoaded;
	static void EnsureModelDefsAreLoaded()
	{
		if (!modelDefsAreLoaded)
		{
			modelDefs.push_back(new RTNeuralModelDefinitionT<1, 8>);
			modelDefs.push_back(new RTNeuralModelDefinitionT<1, 12>);
			modelDefs.push_back(new RTNeuralModelDefinitionT<1, 16>);
			modelDefs.push_back(new RTNeuralModelDefinitionT<1, 24>);
			modelDefs.push_back(new RTNeuralModelDefinitionT<2, 8>);
			modelDefs.push_back(new RTNeuralModelDefinitionT<2, 12>);
			modelDefs.push_back(new RTNeuralModelDefinitionT<2, 16>);

			modelDefsAreLoaded = true;
		}
	}

	static RTNeuralModelDefinitionBase* FindModelDefinition(size_t numLayers, size_t hiddenSize)
	{
		for (auto const& model : modelDefs)
		{
			if ((numLayers == model->GetNumLayers()) && (hiddenSize == model->GetHiddenSize()))
				return model;
		}

		return nullptr;
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

			if ((arch == "WaveNet") || preferNAM)
			{
				NAMModel* model = new NAMModel;

				model->LoadFromJson(modelJson);

				newModel = model;
			}
			else if (arch == "LSTM")
			{
				nlohmann::json config = modelJson["config"];

				auto modelDef = FindModelDefinition(config["num_layers"], config["hidden_size"]);

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
		else if (modelPath.extension() == ".json")
		{
			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size() - 1;
			const std::string modelType = layers.at(0).at("type");
			const int hidden_size = layers.at(0).at("shape").back();

			if (modelType == "lstm")
			{
				auto modelDef = FindModelDefinition(numLayers, hidden_size);

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