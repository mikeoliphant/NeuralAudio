#pragma once

#include "NeuralModel.h"
#include "WaveNet.h"
#include "LSTM.h"

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
			ReadRTNeuralConfig(modelJson);

			return CreateModelFromKerasJson(modelJson);

			return true;
		}

		virtual bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			return false;
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
			NeuralAudio::WaveNetModel<
				NeuralAudio::WaveNetLayerArray<1, 1, headSize, numChannels, 3, IStdDilations, false>,
				NeuralAudio::WaveNetLayerArray<numChannels, 1, 1, headSize, 3, IStdDilations, true>>,
			NeuralAudio::WaveNetModel<
				NeuralAudio::WaveNetLayerArray<1, 1, headSize, numChannels, 3, ILiteDilations1, false>,
				NeuralAudio::WaveNetLayerArray<numChannels, 1, 1, headSize, 3, ILiteDilations2, true>>
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
			int toProcess = std::min(numSamples, model->GetMaxFrames());

			while (numSamples > 0)
			{
				model->Process(input, output, toProcess);

				input += toProcess;
				output += toProcess;
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



	template <int HiddenSize>
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

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new LSTMModel<HiddenSize>;

			nlohmann::json config = modelJson["config"];

			model->SetWeights(modelJson["weights"]);

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

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
		LSTMModel<HiddenSize>* model = nullptr;
	};


	class InternalLSTMDefinitionBase
	{
	public:
		virtual InternalModel* CreateModel()
		{
			return nullptr;
		}

		virtual int GetHiddenSize()
		{
			return 0;
		}
	};

	template <int HiddenSize>
	class InternalLSTMDefinitionT : public InternalLSTMDefinitionBase
	{
	public:
		InternalModel* CreateModel()
		{
			return new InternalLSTMModelT<HiddenSize>;
		}

		virtual int GetHiddenSize()
		{
			return HiddenSize;
		}
	};
}


