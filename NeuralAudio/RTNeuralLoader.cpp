#include <list>
#include "RTNeuralModel.h"

namespace NeuralAudio
{
#ifdef BUILD_STATIC_RTNEURAL
		std::list<RTNeuralLSTMDefinitionBase*> rtNeuralLSTMModelDefs;

		void EnsureRTNeuralModelDefsAreLoaded()
		{
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 8>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 12>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 16>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 24>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 8>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 12>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 16>);
		}

		RTNeuralLSTMDefinitionBase* FindRTNeuralLSTMDefinition(size_t numLayers, size_t hiddenSize)
		{
			for (auto const& model : rtNeuralLSTMModelDefs)
			{
				if ((numLayers == model->GetNumLayers()) && (hiddenSize == model->GetHiddenSize()))
					return model;
			}

			return nullptr;
		}

		NeuralModelImpl* RTNeuralLoadNAMLSTM(const nlohmann::json& modelJson, NeuralModelLoader *loader)
		{
			auto& config = modelJson.at("config");

			auto modelDef = FindRTNeuralLSTMDefinition(config.at("num_layers"), config.at("hidden_size"));

			if (modelDef != nullptr)
			{
				RTNeuralModel* model = modelDef->CreateModel();

				model->SetModelLoader(loader);
				model->LoadFromNAMJson(modelJson);

				return model;
			}

			// If we didn't have a static model that matched, use RTNeural's dynamic model
			RTNeuralModelDyn* dynModel = new RTNeuralModelDyn;

			dynModel->SetModelLoader(loader);
			dynModel->LoadFromNAMJson(modelJson);

			return dynModel;

			return nullptr;
		}

		NeuralModelImpl* RTNeuralLoadKeras(const nlohmann::json& modelJson, NeuralModelLoader* loader)
		{
			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size() - 1;
			const std::string modelType = layers.at(0).at("type");
			const int hiddenSize = layers.at(0).at("shape").back();

			auto modelDef = FindRTNeuralLSTMDefinition(numLayers, hiddenSize);

			if (modelDef != nullptr)
			{
				RTNeuralModel* model = modelDef->CreateModel();

				model->SetModelLoader(loader);
				model->LoadFromKerasJson(modelJson);

				return model;
			}

			return nullptr;
		}
#endif
}

