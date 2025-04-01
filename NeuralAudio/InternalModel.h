#pragma once

#include "NeuralModel.h"
#include "WaveNet.h"
#include "WaveNetDynamic.h"
#include "LSTM.h"
#include "LSTMDynamic.h"

namespace NeuralAudio
{
	using IStdDilations = NeuralAudio::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
	using ILiteDilations1 = NeuralAudio::Dilations<1, 2, 4, 8, 16, 32, 64>;
	using ILiteDilations2 = NeuralAudio::Dilations<128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;

	class InternalModel : public NeuralModel
	{
	public:
		bool LoadFromKerasJson(const nlohmann::json& modelJson)
		{
			ReadKerasConfig(modelJson);

			return CreateModelFromKerasJson(modelJson);

			return true;
		}

		virtual bool CreateModelFromKerasJson(const nlohmann::json& modelJson)
		{
			(void)modelJson;

			return false;
		}

		virtual bool LoadFromNAMJson(const nlohmann::json& modelJson)
		{
			ReadNAMConfig(modelJson);

			return CreateModelFromNAMJson(modelJson);
		}

		virtual bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
		{
			(void)modelJson;

			return false;
		}
	};


	template <int numChannels, int headSize>
	class InternalWaveNetModelT : public InternalModel
	{
		using ModelType = typename std::conditional<numChannels == 16,
			NeuralAudio::WaveNetModelT<
				NeuralAudio::WaveNetLayerArrayT<1, 1, headSize, numChannels, 3, IStdDilations, false>,
				NeuralAudio::WaveNetLayerArrayT<numChannels, 1, 1, headSize, 3, IStdDilations, true>>,
			NeuralAudio::WaveNetModelT<
				NeuralAudio::WaveNetLayerArrayT<1, 1, headSize, numChannels, 3, ILiteDilations1, false>,
				NeuralAudio::WaveNetLayerArrayT<numChannels, 1, 1, headSize, 3, ILiteDilations2, true>>
			>::type;

	public:
		InternalWaveNetModelT()
			: model(nullptr)
		{
		}

		~InternalWaveNetModelT()
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}
		}

		bool IsStatic()
		{
			return true;
		}

		bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			nlohmann::json config = modelJson.at("config");

			model->SetWeights(modelJson.at("weights"));

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			size_t offset = 0;

			while (numSamples > 0)
			{
				size_t toProcess = std::min(numSamples, model->GetMaxFrames());

				model->Process(input + offset, output + offset, toProcess);

				offset += toProcess;
				numSamples -= toProcess;
			}
		}

		void Prewarm()
		{
			model->Prewarm();
		}

	private:
		ModelType* model = nullptr;
	};


	class InternalWaveNetDefinitionBase
	{
	public:
		virtual InternalModel* CreateModel()
		{
			return nullptr;
		}

		virtual size_t GetNumChannels()
		{
			return 0;
		}

		virtual size_t GetHeadSize()
		{
			return 0;
		}
	};

	template <int numChannels, int headSize>
	class InternalWaveNetDefinitionT : public InternalWaveNetDefinitionBase
	{
	public:
		InternalModel* CreateModel()
		{
			return new InternalWaveNetModelT<numChannels, headSize>;
		}

		virtual size_t GetNumChannels()
		{
			return numChannels;
		}

		virtual size_t GetHeadSize()
		{
			return headSize;
		}
	};

	class InternalWaveNetModelDyn : public InternalModel
	{
	public:
		InternalWaveNetModelDyn()
		{
		}

		~InternalWaveNetModelDyn()
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}
		}

		EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::Internal;
		}

		bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
		{
			nlohmann::json config = modelJson.at("config");

			std::vector<WaveNetLayerArray> layerArrays;

			for (size_t i = 0; i < config.at("layers").size(); i++)
			{
				nlohmann::json layerConfig = config.at("layers").at(i);

				layerArrays.push_back(WaveNetLayerArray(layerConfig.at("input_size"), layerConfig.at("condition_size"), layerConfig.at("head_size"),
					layerConfig.at("channels"), layerConfig.at("kernel_size"), layerConfig.at("head_bias"), layerConfig.at("dilations")));
			}

			model = new WaveNetModel(layerArrays);

			model->SetWeights(modelJson.at("weights"));

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			model->SetMaxFrames(maxSize);
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			size_t offset = 0;

			while (numSamples > 0)
			{
				size_t toProcess = std::min(numSamples, model->GetMaxFrames());

				model->Process(input + offset, output + offset, toProcess);

				offset += toProcess;
				numSamples -= toProcess;
			}
		}

		void Prewarm()
		{
			model->Prewarm();
		}

	private:
		WaveNetModel* model = nullptr;
	};


	template <int NumLayers, int HiddenSize>
	class InternalLSTMModelT : public InternalModel
	{
	public:
		InternalLSTMModelT()
			: model(nullptr)
		{
		}

		~InternalLSTMModelT()
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}
		}

		bool IsStatic()
		{
			return true;
		}

		bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new LSTMModelT<NumLayers, HiddenSize>;

			nlohmann::json config = modelJson.at("config");

			model->SetNAMWeights(modelJson.at("weights"));

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		std::vector<float> FlattenWeights(const nlohmann::json& weights)
		{
			std::vector<float> vec;

			for (size_t i = 0; i < weights.size(); i++)
			{
				if (weights[i].is_array())
				{
					auto subVec = FlattenWeights(weights[i]);
					vec.insert(vec.end(), subVec.begin(), subVec.end());
				}
				else
				{
					vec.push_back(weights[i]);
				}
			}

			return vec;
		}

		bool CreateModelFromKerasJson(const nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new LSTMModelT<NumLayers, HiddenSize>;

			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size();

			if (numLayers < 2)
				return false;

			auto lastLayer = layers[numLayers - 1];

			if (lastLayer.at("type") != "dense")
				return false;

			LSTMDef lstmDef;

			lstmDef.HeadWeights = FlattenWeights(lastLayer.at("weights").at(0));
			lstmDef.HeadBias = lastLayer.at("weights").at(1).at(0);

			for (size_t i = 0; i < (numLayers - 1); i++)
			{
				auto layer = layers[i];

				if (layer.at("type") != "lstm")
					return false;

				LSTMLayerDef layerDef;

				layerDef.InputWeights = FlattenWeights(layer.at("weights").at(0));
				layerDef.HiddenWeights = FlattenWeights(layer.at("weights").at(1));
				layerDef.BiasWeights = FlattenWeights(layer.at("weights").at(2));

				lstmDef.Layers.push_back(layerDef);
			}

			model->SetWeights(lstmDef);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			(void)maxSize;
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			model->Process(input, output, numSamples);
		}

		void Prewarm()
		{
			NeuralModel::Prewarm(2048, 64);
		}

	private:
		LSTMModelT<NumLayers, HiddenSize>* model = nullptr;
	};


	class InternalLSTMDefinitionBase
	{
	public:
		virtual InternalModel* CreateModel()
		{
			return nullptr;
		}

		virtual size_t GetNumLayers()
		{
			return 0;
		}

		virtual size_t GetHiddenSize()
		{
			return 0;
		}
	};

	template <int NumLayers, int HiddenSize>
	class InternalLSTMDefinitionT : public InternalLSTMDefinitionBase
	{
	public:
		InternalModel* CreateModel()
		{
			return new InternalLSTMModelT<NumLayers, HiddenSize>;
		}

		virtual size_t GetNumLayers()
		{
			return NumLayers;
		}

		virtual size_t GetHiddenSize()
		{
			return HiddenSize;
		}
	};

	class InternalLSTMModelDyn : public InternalModel
	{
	public:
		InternalLSTMModelDyn()
			: model(nullptr)
		{
		}

		~InternalLSTMModelDyn()
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}
		}

		bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			nlohmann::json config = modelJson.at("config");

			model = new LSTMModel(config.at("num_layers"), config.at("hidden_size"));

			model->SetNAMWeights(modelJson.at("weights"));

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		std::vector<float> FlattenWeights(const nlohmann::json& weights)
		{
			std::vector<float> vec;

			for (size_t i = 0; i < weights.size(); i++)
			{
				if (weights[i].is_array())
				{
					auto subVec = FlattenWeights(weights[i]);
					vec.insert(vec.end(), subVec.begin(), subVec.end());
				}
				else
				{
					vec.push_back(weights[i]);
				}
			}

			return vec;
		}

		bool CreateModelFromKerasJson(const nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size();
			const size_t hiddenSize = layers.at(0).at("shape").back();

			if (numLayers < 2)
				return false;

			auto lastLayer = layers[numLayers - 1];

			if (lastLayer.at("type") != "dense")
				return false;

			model = new LSTMModel(numLayers - 1, hiddenSize);

			LSTMDef lstmDef;

			lstmDef.HeadWeights = FlattenWeights(lastLayer.at("weights").at(0));
			lstmDef.HeadBias = lastLayer.at("weights").at(1).at(0);

			for (size_t i = 0; i < (numLayers - 1); i++)
			{
				auto layer = layers[i];

				if (layer.at("type") != "lstm")
					return false;

				LSTMLayerDef layerDef;

				layerDef.InputWeights = FlattenWeights(layer.at("weights").at(0));
				layerDef.HiddenWeights = FlattenWeights(layer.at("weights").at(1));
				layerDef.BiasWeights = FlattenWeights(layer.at("weights").at(2));

				lstmDef.Layers.push_back(layerDef);
			}

			model->SetWeights(lstmDef);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			(void)maxSize;
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			model->Process(input, output, numSamples);
		}

		void Prewarm()
		{
			NeuralModel::Prewarm(2048, 64);
		}

	private:
		LSTMModel* model = nullptr;
	};
}


