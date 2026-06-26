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
#include "CompositeModel.h"

namespace NeuralAudio
{
	static bool modelDefsAreLoaded;

	static std::list<InternalWaveNetDefinitionBase*> internalWavenetModelDefs;
	static std::list<InternalLSTMDefinitionBase*> internalLSTMModelDefs;

	static void EnsureModelDefsAreLoaded()
	{
		if (!modelDefsAreLoaded)
		{
#ifdef BUILD_INTERNAL_STATIC_WAVENET
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<16, 8>);	// Standard
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<12, 6>);	// Lite
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<8, 4>);	// Feather
			internalWavenetModelDefs.push_back(new InternalWaveNetDefinitionT<4, 2>);	// Nano
#endif

#ifdef BUILD_INTERNAL_STATIC_LSTM
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 8>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 12>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 16>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<1, 24>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<2, 8>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<2, 12>);
			internalLSTMModelDefs.push_back(new InternalLSTMDefinitionT<2, 16>);
#endif

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

	static bool CheckDilations(const nlohmann::json& dilationJson, std::vector<int>& checkDilations)
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

	static void OversampleNAMConfig(nlohmann::json& modelJson, int externalSampleRate)
	{
		std::string arch = modelJson.at("architecture");

		if (arch != "WaveNet")
			return;

		int modelSampleRate = 48000;

		if (modelJson.contains("sample_rate") && modelJson.at("sample_rate").is_number())
		{
			modelSampleRate = (int)modelJson.at("sample_rate").get<float>();
		}

		if (modelSampleRate == externalSampleRate)
			return;	// No oversampling required

		if ((externalSampleRate % modelSampleRate) != 0)
			return;	// Not an integer multiple

		int oversampleFactor = externalSampleRate / modelSampleRate;

		auto& config = modelJson.at("config");

		for (auto& layer : config.at("layers"))
		{
			for (auto& dilation : layer.at("dilations"))
			{
				dilation = dilation.get<int>() * oversampleFactor;
			}

			if (layer.contains("head"))
			{
				auto& head = layer.at("head");

				head["head_dilation"] = oversampleFactor;
			}
		}
	}

	bool NeuralModelLoader::SupportsWaveNetLoadMode(EModelLoadMode mode)
	{
		if (mode == EModelLoadMode::NAMCore)
#ifdef BUILD_NAMCORE
			return true;
#else
			return false;
#endif

		if (mode == EModelLoadMode::RTNeural)
			return false;

		return true;
	}

	bool NeuralModelLoader::SupportsLSTMLoadMode(EModelLoadMode mode)
	{
		if (mode == EModelLoadMode::NAMCore)
#ifdef BUILD_NAMCORE
			return true;
#else
			return false;
#endif

		return true;
	}

	bool NAMIsA2(std::string version)
	{
		int major = 0, minor = 0, patch = 0;
		char dot;
		std::stringstream ss(version);

		ss >> major >> dot >> minor >> dot >> patch;

		return (major > 0) || (minor > 5) || ((minor == 5) && (patch > 4));
	}

	NeuralModel* NeuralModelLoader::CreateFromFile(const std::filesystem::path& modelPath, bool doPrewarm)
	{
		if (!std::filesystem::exists(modelPath))
			return nullptr;

		std::ifstream jsonStream(modelPath, std::ifstream::binary);

		return CreateFromStream(jsonStream, modelPath.extension(), doPrewarm);
	}


	NeuralModel* NeuralModelLoader::CreateFromStream(std::basic_istream<char>& jsonStream, const std::filesystem::path& extension, bool doPrewarm)
	{
		nlohmann::json modelJson;
		jsonStream >> modelJson;

		return CreateFromJson(modelJson, extension, doPrewarm);
	}

	NeuralModel* NeuralModelLoader::CreateFromJson(nlohmann::json& modelJson, const std::filesystem::path& extension, bool doPrewarm)
	{
		EnsureModelDefsAreLoaded();

		NeuralModelImpl* newModel = nullptr;

		if (extension == ".nam")
		{
			OversampleNAMConfig(modelJson, externalSampleRate);

			std::string arch = modelJson.at("architecture");

#ifdef BUILD_NAMCORE
			std::string version = modelJson.at("version");

			if ((wavenetLoadMode == EModelLoadMode::NAMCore) || NAMIsA2(version))
			{
				if (arch == "SlimmableContainer")	// Packed A2 multi-model file 
				{
					ScalableCompositeModel* model = new ScalableCompositeModel;

					model->SetModelLoader(this);
					model->LoadFromJson(modelJson);

					newModel = model;
				}
				else
				{
					NAMModel* model = new NAMModel;

					model->SetModelLoader(this);
					model->LoadFromJson(modelJson);

					newModel = model;
				}
			}
#endif

			auto& config = modelJson.at("config");

			if (newModel == nullptr)
			{
				if (arch == "WaveNet")
				{
					if (config.at("layers").size() == 2)
					{
						auto& firstLayerConfig = config.at("layers").at(0);
						auto& secondLayerConfig = config.at("layers").at(1);

						if (!firstLayerConfig.at("gated") && !secondLayerConfig.at("gated") && !firstLayerConfig.at("head_bias") && secondLayerConfig.at("head_bias"))
						{
							bool isOfficialArchitecture = false;

							if (firstLayerConfig.at("channels") == 16)
							{
								if (CheckDilations(firstLayerConfig.at("dilations"), stdDilations) && CheckDilations(secondLayerConfig.at("dilations"), stdDilations))
								{
									isOfficialArchitecture = true;
								}
							}
							else
							{
								if (CheckDilations(firstLayerConfig.at("dilations"), liteDilations) && CheckDilations(secondLayerConfig.at("dilations"), liteDilations2))
								{
									isOfficialArchitecture = true;
								}
							}

							if (isOfficialArchitecture)
							{
								if (newModel == nullptr)
								{
									auto modelDef = FindInternalWaveNetDefinition(firstLayerConfig.at("channels"), firstLayerConfig.at("head_size"));

									if (modelDef != nullptr)
									{
										auto model = modelDef->CreateModel();

										model->SetModelLoader(this);
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

						model->SetModelLoader(this);

						if (model->LoadFromNAMJson(modelJson))
						{
							newModel = model;
						}
					}
				}
				else if (arch == "LSTM")
				{
#ifdef BUILD_STATIC_RTNEURAL
					if (lstmLoadMode == EModelLoadMode::RTNeural)
					{
						newModel = RTNeuralLoadNAMLSTM(modelJson, this);
					}
#endif

					if (newModel == nullptr)
					{
						auto modelDef = FindInternalLSTMDefinition(config.at("num_layers"), config.at("hidden_size"));

						if (modelDef != nullptr)
						{
							auto model = modelDef->CreateModel();

							model->SetModelLoader(this);
							model->LoadFromNAMJson(modelJson);

							newModel = model;
						}

						// Use a dynamic model if we had no static definition
						if (newModel == nullptr)
						{
							InternalLSTMModelDyn* model = new InternalLSTMModelDyn;

							model->SetModelLoader(this);

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
					newModel = RTNeuralLoadKeras(modelJson, this);
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

		if ((newModel != nullptr) && doPrewarm)
		{
			newModel->Prewarm();
		}

		return newModel;
	}
}