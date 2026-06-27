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
			internalWavenetModelDefs.push_back(new InternalA1WaveNetDefinitionT<16, 8>);	// Standard
			internalWavenetModelDefs.push_back(new InternalA1WaveNetDefinitionT<12, 6>);	// Lite
			internalWavenetModelDefs.push_back(new InternalA1WaveNetDefinitionT<8, 4>);	// Feather
			internalWavenetModelDefs.push_back(new InternalA1WaveNetDefinitionT<4, 2>);	// Nano
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

	static InternalWaveNetDefinitionBase* FindInternalWaveNetDefinition(size_t NumChannels, size_t HeadSize)
	{
		for (auto const& model : internalWavenetModelDefs)
		{
			if ((NumChannels == model->GetNumChannels()) && (HeadSize == model->GetHeadSize()))
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

	static std::vector<int> a2KernelSizes = { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6 };
	static std::vector<int> a2Dilations = { 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239 };

	static bool CheckIntegerSequence(const nlohmann::json& sequenceJson, std::vector<int>& sequenceVector)
	{
		if (sequenceJson.size() != sequenceVector.size())
			return false;

		for (size_t i = 0; i < sequenceJson.size(); i++)
		{
			if (sequenceJson[i] != sequenceVector[i])
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

	bool ApproxEqual(float a, float b, float epsilon = 1e-5f)
	{
		return std::fabs(a - b) <= epsilon;
	}

	bool IsActive(const nlohmann::json& json, const char* name)
	{
		if (!json.contains(name))
			return true;

		return json.at(name).value("active", false);
	}

	bool HasNonNull(const nlohmann::json& json, const char* name)
	{
		return json.contains(name) && !json.at(name).is_null();
	}

	bool NAMIsA2Standard(const nlohmann::json& modelJson)
	{
		if (!modelJson.contains("architecture"))
			return false;

		if (modelJson.at("architecture") != "WaveNet")
			return false;

		if (!modelJson.contains("config"))
			return false;

		auto& config = modelJson.at("config");

		if (HasNonNull(config, "head"))
			return false;

		if (config.contains("condition_dsp"))
			return false;

		if (config.value("in_channels", 1) != 1)
			return false;

		if (!config.contains("layers"))
			return false;

		if (config.at("layers").size() != 1)
			return false;

		const auto& layerConfig = config.at("layers").at(0);

		if (layerConfig.value("input_size", 0) != 1)
			return false;

		if (layerConfig.value("condition_size", 0) != 1)
			return false;

		const int channels = layerConfig.value("channels", 0);

		if ((channels != 3) && (channels != 8))
			return false;

		if (layerConfig.value("bottleneck", channels) != channels)
			return false;

		if (!layerConfig.contains("kernel_sizes"))
			return false;

		if (!CheckIntegerSequence(layerConfig.at("kernel_sizes"), a2KernelSizes))
			return false;

		if (!layerConfig.contains("dilations"))
			return false;

		if (!CheckIntegerSequence(layerConfig.at("dilations"), a2Dilations))
			return false;

		if (!layerConfig.contains("activation"))
			return false;

		for (const auto& activation : layerConfig.at("activation"))
		{
			if (!activation.contains("type"))
				return false;

			if (activation.at("type") != "LeakyReLU")
				return false;

			if (!ApproxEqual(activation.value("negative_slope", 0.01f), 0.01f))
				return false;
		}

		if (layerConfig.contains("secondary_activation"))
		{
			for (const auto& activation : layerConfig.at("secondary_activation"))
			{
				if (!activation.is_null())
					return false;
			}
		}

		if (layerConfig.contains("gating_mode"))
		{
			for (const auto& gating : layerConfig.at("gating_mode"))
			{
				if (!gating.is_null() && (gating != "none"))
					return false;
			}
		}

		if (!layerConfig.contains("head"))
			return false;

		const auto& head = layerConfig.at("head");

		if (head.value("out_channels", 1) != 1)
			return false;

		if (head.value("kernel_size", 16) != 16)
			return false;

		if (head.value("head_dilation", 1) != 1)
			return false;

		if (!head.value("bias", true))
			return false;

		if (!IsActive(layerConfig, "layer1x1"))
			return false;

		if (layerConfig.at("layer1x1").value("groups", 1) != 1)
			return false;

		for (auto& key : { "head1x1", "conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film",
							"activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film" })
		{
			if (IsActive(layerConfig, key))
				return false;
		}

		if (layerConfig.value("groups_input", 1) != 1)
			return false;

		if (layerConfig.value("groups_input_mixin", 1) != 1)
			return false;

		if (HasNonNull(layerConfig, "slimmable"))
			return false;

		return true;
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

			if (arch == "SlimmableContainer")	// Packed A2 multi-model file 
			{
				ScalableCompositeModel* model = new ScalableCompositeModel;

				model->SetModelLoader(this);
				model->LoadFromJson(modelJson);

				newModel = model;
			}


			if (newModel == nullptr)
			{
#ifdef BUILD_NAMCORE
				std::string version = modelJson.at("version");

				bool loadA2WithNAMCore = true;

#ifdef BUILD_STATIC_INTERNAL_NAMA2
				loadA2WithNAMCore = false;
#endif
				if ((wavenetLoadMode == EModelLoadMode::NAMCore) || (NAMIsA2(version) && (loadA2WithNAMCore || !NAMIsA2Standard(modelJson))))
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
					if (config.at("layers").size() == 1)
					{
#ifdef BUILD_STATIC_INTERNAL_NAMA2
						auto& layerConfig = config.at("layers").at(0);

						if (CheckIntegerSequence(layerConfig.at("dilations"), a2Dilations))
						{
							if (layerConfig.at("channels") == 3)
							{
								auto model = new InternalWaveNetModelT<NeuralAudio::WaveNetModelT<NeuralAudio::WaveNetLayerArrayT<1, 1, 1, 16, 1, 3, A2KernelSizes, A2Dilations, true, EActivationType::LeakyReLU>>>();

								if (model != nullptr)
								{
									model->SetModelLoader(this);
									model->LoadFromNAMJson(modelJson);

									newModel = model;
								}
							}
							else if (layerConfig.at("channels") == 8)
							{
								auto model = new InternalWaveNetModelT <NeuralAudio::WaveNetModelT<NeuralAudio::WaveNetLayerArrayT<1, 1, 1, 16, 1, 8, A2KernelSizes, A2Dilations, true, EActivationType::LeakyReLU>>>();

								if (model != nullptr)
								{
									model->SetModelLoader(this);
									model->LoadFromNAMJson(modelJson);

									newModel = model;
								}
							}
						}
#endif
					}
					else if (config.at("layers").size() == 2)
					{
						auto& firstLayerConfig = config.at("layers").at(0);
						auto& secondLayerConfig = config.at("layers").at(1);

						if (!firstLayerConfig.at("gated") && !secondLayerConfig.at("gated") && !firstLayerConfig.at("head_bias") && secondLayerConfig.at("head_bias"))
						{
							bool isOfficialArchitecture = false;

							if (firstLayerConfig.at("channels") == 16)
							{
								if (CheckIntegerSequence(firstLayerConfig.at("dilations"), stdDilations) && CheckIntegerSequence(secondLayerConfig.at("dilations"), stdDilations))
								{
									isOfficialArchitecture = true;
								}
							}
							else
							{
								if (CheckIntegerSequence(firstLayerConfig.at("dilations"), liteDilations) && CheckIntegerSequence(secondLayerConfig.at("dilations"), liteDilations2))
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