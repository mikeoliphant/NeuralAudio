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

	static RTNeuralModelDefinitionBase* FindModelDefinition(int numLayers, int hiddenSize)
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

		if (modelPath.extension() == ".nam")
		{
			std::string arch = modelJson["architecture"];

			if ((arch == "WaveNet") || preferNAM)
			{
				NAMModel* model = new NAMModel;

				model->LoadFromJson(modelJson);

				return model;
			}
			else if (arch == "LSTM")
			{
				nlohmann::json config = modelJson["config"];

				auto modelDef = FindModelDefinition(config["num_layers"], config["hidden_size"]);

				if (modelDef != nullptr)
				{
					RTNeuralModel* model = modelDef->CreateModel();
					model->LoadFromNAMJson(modelJson);

					return model;
				}

				RTNeuralModelDyn* model = new RTNeuralModelDyn;
				model->LoadFromNAMJson(modelJson);

				return model;
			}
		}
		else if (modelPath.extension() == ".json")
		{
			RTNeuralModel* model = modelDefs.front()->CreateModel();

			model->LoadFromKerasJson(modelJson);

			return model;
		}

		return nullptr;
	}
}