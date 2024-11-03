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
			modelDefs.push_back(new RTNeuralModelDefinitionT<1, 16>);

			modelDefsAreLoaded = true;
		}
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
				RTNeuralModel *model = modelDefs.front()->CreateModel();
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