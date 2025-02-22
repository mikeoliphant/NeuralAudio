#include <list>
#include "RTNeuralModel.h"

namespace NeuralAudio
{
#ifdef BUILD_STATIC_RTNEURAL
		std::list<RTNeuralLSTMDefinitionBase*> rtNeuralLSTMModelDefs;
		std::list<RTNeuralWaveNetDefinitionBase*> rtNeuralWaveNetModelDefs;

		void EnsureRTNeuralModelDefsAreLoaded()
		{
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 8>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 12>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 16>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<1, 24>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 8>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 12>);
			rtNeuralLSTMModelDefs.push_back(new RTNeuralLSTMDefinitionT<2, 16>);

			rtNeuralWaveNetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<16, 8>);	// Standard
			rtNeuralWaveNetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<12, 6>);	// Lite
			rtNeuralWaveNetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<8, 4>);	// Feather
			rtNeuralWaveNetModelDefs.push_back(new RTNeuralWaveNetDefinitionT<4, 2>);	// Nano
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

		RTNeuralWaveNetDefinitionBase* FindRTNeuralWaveNetDefinition(size_t numChannels, size_t headSize)
		{
			for (auto const& model : rtNeuralWaveNetModelDefs)
			{
				if ((numChannels == model->GetNumChannels()) && (headSize == model->GetHeadSize()))
					return model;
			}

			return nullptr;
		}

		NeuralModel* RTNeuralLoadNAMWaveNet(const nlohmann::json& modelJson)
		{
			nlohmann::json config = modelJson["config"];

			nlohmann::json firstLayerConfig = config["layers"][0];
			nlohmann::json secondLayerConfig = config["layers"][1];
			
			auto modelDef = FindRTNeuralWaveNetDefinition(firstLayerConfig["channels"], firstLayerConfig["head_size"]);

			if (modelDef != nullptr)
			{
				auto model = modelDef->CreateModel();

				model->LoadFromNAMJson(modelJson);

				return model;
			}

			return nullptr;
		}

		NeuralModel* RTNeuralLoadNAMLSTM(const nlohmann::json& modelJson)
		{
			nlohmann::json config = modelJson["config"];

			auto modelDef = FindRTNeuralLSTMDefinition(config["num_layers"], config["hidden_size"]);

			if (modelDef != nullptr)
			{
				RTNeuralModel* model = modelDef->CreateModel();
				model->LoadFromNAMJson(modelJson);

				if (model != nullptr)
					return model;

				// If we didn't have a static model that matched, use RTNeural's dynamic model
				RTNeuralModelDyn* dynModel = new RTNeuralModelDyn;
				dynModel->LoadFromNAMJson(modelJson);

				return dynModel;
			}

			return nullptr;
		}

		NeuralModel* RTNeuralLoadKeras(const nlohmann::json& modelJson)
		{
			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size() - 1;
			const std::string modelType = layers.at(0).at("type");
			const int hiddenSize = layers.at(0).at("shape").back();

			auto modelDef = FindRTNeuralLSTMDefinition(numLayers, hiddenSize);

			if (modelDef != nullptr)
			{
				RTNeuralModel* model = modelDef->CreateModel();

				model->LoadFromKerasJson(modelJson);

				return model;
			}

			return nullptr;
		}
#endif
}

