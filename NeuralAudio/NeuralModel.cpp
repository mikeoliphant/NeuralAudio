#include "NeuralModel.h"
#include "NAMModel.h"

namespace NeuralAudio
{
	NeuralModel* NeuralModel::CreateFromFile(std::filesystem::path modelPath)
	{
		NAMModel* model = new NAMModel;

		model->LoadFromFile(modelPath);

		return model;
	}
}