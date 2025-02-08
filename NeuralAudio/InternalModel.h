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
		bool LoadFromKerasJson(nlohmann::json& modelJson)
		{
			ReadKerasConfig(modelJson);

			return CreateModelFromKerasJson(modelJson);

			return true;
		}

		virtual bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			return CreateModelFromKerasJson(modelJson);
		}

		virtual bool LoadFromNAMJson(nlohmann::json& modelJson)
		{
			ReadNAMConfig(modelJson);

			return CreateModelFromNAMJson(modelJson);
		}

		virtual bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
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

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			nlohmann::json config = modelJson["config"];

			model->SetWeights(modelJson["weights"]);

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			model->SetMaxFrames(defaultMaxAudioBufferSize);
		}

		void Process(float* input, float* output, int numSamples)
		{
			int offset = 0;

			while (numSamples > 0)
			{
				int toProcess = std::min(numSamples, model->GetMaxFrames());

				model->Process(input + offset, output + offset, toProcess);

				offset += toProcess;
				numSamples -= toProcess;
			}
		}

		void Prewarm()
		{
			const int numSamples = model->GetMaxFrames();

			std::vector<float> input;
			input.resize(numSamples);
			std::fill(input.begin(), input.end(), 0);

			std::vector<float> output;
			output.resize(numSamples);

			for (int block = 0; block < (4096 / numSamples); block++)
			{
				model->Process(input.data(), output.data(), numSamples);
			}
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

		virtual int GetNumChannels()
		{
			return 0;
		}

		virtual int GetHeadSize()
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

		virtual int GetNumChannels()
		{
			return numChannels;
		}

		virtual int GetHeadSize()
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

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			nlohmann::json config = modelJson["config"];

			std::vector<WaveNetLayerArray> layerArrays;

			for (size_t i = 0; i < config["layers"].size(); i++)
			{
				nlohmann::json layerConfig = config["layers"][i];

				layerArrays.push_back(WaveNetLayerArray(layerConfig["input_size"], layerConfig["condition_size"], layerConfig["head_size"],
					layerConfig["channels"], layerConfig["kernel_size"], layerConfig["head_bias"], layerConfig["dilations"]));
			}

			model = new WaveNetModel(layerArrays);

			model->SetWeights(modelJson["weights"]);

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			model->SetMaxFrames(defaultMaxAudioBufferSize);
		}

		void Process(float* input, float* output, int numSamples)
		{
			int offset = 0;

			while (numSamples > 0)
			{
				int toProcess = std::min(numSamples, model->GetMaxFrames());

				model->Process(input + offset, output + offset, toProcess);

				offset += toProcess;
				numSamples -= toProcess;
			}
		}

		void Prewarm()
		{
			const int numSamples = model->GetMaxFrames();

			std::vector<float> input;
			input.resize(numSamples);
			std::fill(input.begin(), input.end(), 0);

			std::vector<float> output;
			output.resize(numSamples);

			for (int block = 0; block < (4096 / numSamples); block++)
			{
				model->Process(input.data(), output.data(), numSamples);
			}
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

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new LSTMModelT<NumLayers, HiddenSize>;

			nlohmann::json config = modelJson["config"];

			model->SetNAMWeights(modelJson["weights"]);

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		std::vector<float> FlattenWeights(const nlohmann::json& weights)
		{
			std::vector<float> vec;

			for (auto i = 0; i < weights.size(); i++)
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

		bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new LSTMModelT<NumLayers, HiddenSize>;

			nlohmann::json config = modelJson["config"];

			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size();

			if (numLayers < 2)
				return false;

			auto lastLayer = layers[numLayers - 1];

			if (lastLayer["type"] != "dense")
				return false;

			LSTMDef lstmDef;

			lstmDef.HeadWeights = FlattenWeights(lastLayer["weights"][0]);
			lstmDef.HeadBias = lastLayer["weights"][1][0];

			for (int i = 0; i < (numLayers - 1); i++)
			{
				auto layer = layers[i];

				if (layer["type"] != "lstm")
					return false;

				LSTMLayerDef layerDef;

				layerDef.InputWeights = FlattenWeights(layer["weights"][0]);
				layerDef.HiddenWeights = FlattenWeights(layer["weights"][1]);
				layerDef.BiasWeights = FlattenWeights(layer["weights"][2]);

				lstmDef.Layers.push_back(layerDef);
			}

			model->SetWeights(lstmDef);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			//model->prepare(maxSize);
		}

		void Process(float* input, float* output, int numSamples)
		{
			model->Process(input, output, numSamples);
		}

		void Prewarm()
		{
			//constexpr int numSamples = 64;

			//std::vector<float> input;
			//input.resize(numSamples);
			//std::fill(input.begin(), input.end(), 0);

			//std::vector<float> output;
			//output.resize(numSamples);

			//for (int block = 0; block < (4096 / numSamples); block++)
			//{
			//	model->Process(input.data(), output.data(), numSamples);
			//}
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

		virtual int GetNumLayers()
		{
			return 0;
		}

		virtual int GetHiddenSize()
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

		virtual int GetNumLayers()
		{
			return NumLayers;
		}

		virtual int GetHiddenSize()
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

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			nlohmann::json config = modelJson["config"];

			model = new LSTMModel(config["num_layers"], config["hidden_size"]);

			model->SetNAMWeights(modelJson["weights"]);

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		std::vector<float> FlattenWeights(const nlohmann::json& weights)
		{
			std::vector<float> vec;

			for (auto i = 0; i < weights.size(); i++)
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

		bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			nlohmann::json config = modelJson["config"];

			const auto layers = modelJson.at("layers");
			const size_t numLayers = layers.size();
			const int hiddenSize = layers.at(0).at("shape").back();

			if (numLayers < 2)
				return false;

			auto lastLayer = layers[numLayers - 1];

			if (lastLayer["type"] != "dense")
				return false;

			model = new LSTMModel(numLayers, hiddenSize);

			LSTMDef lstmDef;

			lstmDef.HeadWeights = FlattenWeights(lastLayer["weights"][0]);
			lstmDef.HeadBias = lastLayer["weights"][1][0];

			for (int i = 0; i < (numLayers - 1); i++)
			{
				auto layer = layers[i];

				if (layer["type"] != "lstm")
					return false;

				LSTMLayerDef layerDef;

				layerDef.InputWeights = FlattenWeights(layer["weights"][0]);
				layerDef.HiddenWeights = FlattenWeights(layer["weights"][1]);
				layerDef.BiasWeights = FlattenWeights(layer["weights"][2]);

				lstmDef.Layers.push_back(layerDef);
			}

			model->SetWeights(lstmDef);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			//model->prepare(maxSize);
		}

		void Process(float* input, float* output, int numSamples)
		{
			model->Process(input, output, numSamples);
		}

		void Prewarm()
		{
			//constexpr int numSamples = 64;

			//std::vector<float> input;
			//input.resize(numSamples);
			//std::fill(input.begin(), input.end(), 0);

			//std::vector<float> output;
			//output.resize(numSamples);

			//for (int block = 0; block < (4096 / numSamples); block++)
			//{
			//	model->Process(input.data(), output.data(), numSamples);
			//}
		}

	private:
		LSTMModel* model = nullptr;
	};
}


