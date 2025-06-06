#pragma once

#include "NeuralModel.h"
#include <RTNeural/RTNeural.h>
#ifdef BUILD_STATIC_RTNEURAL
#include "wavenet_model.hpp"
#endif
#include "TemplateHelper.h"

namespace NeuralAudio
{
	struct FastMathsProvider
	{
		template <typename Matrix>
		static auto tanh(const Matrix& x)
		{
			return x.array().tanh();
		}

		template <typename Matrix>
		static auto sigmoid(const Matrix& x)
		{
			using T = typename Matrix::Scalar;

			return ((x.array() / (T)2).array().tanh() + (T)1) / (T)2;
		}

		template <typename Matrix>
		static auto exp(const Matrix& x)
		{
			return x.array().exp();
		}
	};

	class RTNeuralModel : public NeuralModel
	{
	public:
		EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::RTNeural;
		}

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

#ifdef BUILD_STATIC_RTNEURAL
	template <int numLayers, int hiddenSize>
	class RTNeuralLSTMModelT : public RTNeuralModel
	{
		using ModelType = typename std::conditional<numLayers == 1,
			RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, hiddenSize, RTNeural::SampleRateCorrectionMode::None, FastMathsProvider>, RTNeural::DenseT<float, hiddenSize, 1>>,
			RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, hiddenSize, RTNeural::SampleRateCorrectionMode::None, FastMathsProvider>,
				RTNeural::LSTMLayerT<float, hiddenSize, hiddenSize, RTNeural::SampleRateCorrectionMode::None, FastMathsProvider>, RTNeural::DenseT<float, hiddenSize, 1>>
		>::type;

	public:
		RTNeuralLSTMModelT()
			: model(nullptr)
		{
		}

		~RTNeuralLSTMModelT()
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

		bool CreateModelFromKerasJson(const nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			model->parseJson(modelJson, false);
			model->reset();

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

			std::vector<float> weights = modelJson.at("weights");

			const int networkInputSize = 1;
			const int networkOutputSize = 1;
			const int gateSize = 4 * hiddenSize;

			auto iter = weights.begin();

			ForEachIndex<numLayers>([&](auto layer)
				{
					const int layerInputSize = (layer == 0) ? networkInputSize : hiddenSize;

					Eigen::MatrixXf inputPlusHidden = Eigen::Map<Eigen::MatrixXf>(&(*iter), layerInputSize + hiddenSize, gateSize);

					auto& lstmLayer = model->template get<layer>();

					// Input weights
					std::vector<std::vector<float>> inputWeights;

					inputWeights.resize(layerInputSize);

					for (size_t col = 0; col < layerInputSize; col++)
					{
						inputWeights[col].resize(gateSize);

						for (size_t row = 0; row < gateSize; row++)
						{
							inputWeights[col][row] = inputPlusHidden(col, row);
						}
					}

					lstmLayer.setWVals(inputWeights);

					// Recurrent weights
					std::vector<std::vector<float>> hiddenWeights;

					hiddenWeights.resize(hiddenSize);

					for (size_t col = 0; col < hiddenSize; col++)
					{
						hiddenWeights[col].resize(gateSize);

						for (size_t row = 0; row < gateSize; row++)
						{
							hiddenWeights[col][row] = inputPlusHidden(col + layerInputSize, row);
						}
					}

					lstmLayer.setUVals(hiddenWeights);

					iter += (gateSize * (layerInputSize + hiddenSize));

					// Bias weights
					std::vector<float> biasWeights = std::vector<float>(iter, iter + gateSize);

					lstmLayer.setBVals(biasWeights);

					iter += gateSize;

					// initial internal state values follow here in NAM, but aren't supported by RTNeural

					iter += hiddenSize * 2;	// (hidden state and cell state)
				});

			// Dense layer weights
			auto& denseLayer = model->template get<numLayers>();

			std::vector<std::vector<float>> denseWeights;
			denseWeights.resize(1);
			denseWeights[0] = std::vector<float>(iter, iter + hiddenSize);

			denseLayer.setWeights(denseWeights);

			iter += hiddenSize;

			// Dense layer bias
			auto denseBias = std::vector<float>(iter, iter + networkOutputSize);
			denseLayer.setBias(&(*iter));

			iter += networkOutputSize;

			model->reset();

			return true;
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			for (size_t i = 0; i < numSamples; i++)
				output[i] = model->forward(input + i);
		}

		void Prewarm()
		{
			float sample = 0;

			for (size_t i = 0; i < 2048; i++)
				model->forward(&sample);
		}

	private:
		ModelType* model = nullptr;
	};

	using StdDilations = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
	using LiteDilations1 = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64>;
	using LiteDilations2 = wavenet::Dilations<128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;

	template <int numChannels, int headSize>
	class RTNeuralWaveNetModelT : public RTNeuralModel
	{
		using ModelType = typename std::conditional<numChannels == 16,
			wavenet::Wavenet_Model<float, 1,
				wavenet::Layer_Array<float, 1, 1, headSize, numChannels, 3, StdDilations, false, FastMathsProvider>,
				wavenet::Layer_Array<float, numChannels, 1, 1, headSize, 3, StdDilations, true, FastMathsProvider>>,
			wavenet::Wavenet_Model<float, 1,
				wavenet::Layer_Array<float, 1, 1, headSize, numChannels, 3, LiteDilations1, false, FastMathsProvider>,
				wavenet::Layer_Array<float, numChannels, 1, 1, headSize, 3, LiteDilations2, true, FastMathsProvider>>
			>::type;

	public:
		RTNeuralWaveNetModelT()
			: model(nullptr)
		{
		}

		~RTNeuralWaveNetModelT()
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

			model->load_weights(modelJson);

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		void SetMaxAudioBufferSize(int maxSize)
		{
			model->prepare(maxSize);
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			model->forward(input, output, (int)numSamples);
		}

		void Prewarm()
		{
			float sample = 0;

			for (size_t i = 0; i < 2048; i++)
				model->forward(sample);
		}

	private:
		ModelType* model = nullptr;
	};

	class RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
		{
			return nullptr;
		}
	};

	class RTNeuralLSTMDefinitionBase : public RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
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

	template <int numLayers, int hiddenSize>
	class RTNeuralLSTMDefinitionT : public RTNeuralLSTMDefinitionBase
	{
	public:
		RTNeuralModel* CreateModel()
		{
			return new RTNeuralLSTMModelT<numLayers, hiddenSize>;
		}

		size_t GetNumLayers()
		{
			return numLayers;
		}

		size_t GetHiddenSize()
		{
			return hiddenSize;
		}
	};

	class RTNeuralWaveNetDefinitionBase : public RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
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
	class RTNeuralWaveNetDefinitionT : public RTNeuralWaveNetDefinitionBase
	{
	public:
		RTNeuralModel* CreateModel()
		{
			return new RTNeuralWaveNetModelT<numChannels, headSize>;
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

#endif

	class RTNeuralModelDyn : public RTNeuralModel
	{
	public:
		RTNeuralModelDyn()
		{
		}

		~RTNeuralModelDyn()
		{
			if (model)
				model.reset();
		}

		EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::RTNeural;
		}

		bool CreateModelFromKerasJson(const nlohmann::json& modelJson)
		{
			model = RTNeural::json_parser::parseJson<float, FastMathsProvider>(modelJson, false);
			model->reset();

			return true;
		}

		bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
		{
			model = std::make_unique<RTNeural::Model<float>>(1);

			nlohmann::json config = modelJson.at("config");

			const size_t numLayers = config.at("num_layers");
			const size_t inputSize = config.at("input_size");
			const size_t hiddenSize = config.at("hidden_size");

			std::vector<float> weights = modelJson.at("weights");

			const size_t networkInputSize = inputSize;
			const size_t networkOutputSize = inputSize;
			const size_t gateSize = 4 * hiddenSize;

			auto iter = weights.begin();

			for (size_t layer = 0; layer < numLayers; layer++)
			{
				const size_t layerInputSize = (layer == 0) ? networkInputSize : hiddenSize;

				Eigen::MatrixXf inputPlusHidden = Eigen::Map<Eigen::MatrixXf>(&(*iter), layerInputSize + hiddenSize, gateSize);

				auto lstmLayer = new RTNeural::LSTMLayer<float>((int)layerInputSize, (int)hiddenSize);

				model->addLayer(lstmLayer);

				// Input weights
				std::vector<std::vector<float>> inputWeights;

				inputWeights.resize(layerInputSize);

				for (size_t col = 0; col < layerInputSize; col++)
				{
					inputWeights[col].resize(gateSize);

					for (size_t row = 0; row < gateSize; row++)
					{
						inputWeights[col][row] = inputPlusHidden(col, row);
					}
				}

				lstmLayer->setWVals(inputWeights);

				// Recurrent weights
				std::vector<std::vector<float>> hiddenWeights;

				hiddenWeights.resize(hiddenSize);

				for (size_t col = 0; col < hiddenSize; col++)
				{
					hiddenWeights[col].resize(gateSize);

					for (size_t row = 0; row < gateSize; row++)
					{
						hiddenWeights[col][row] = inputPlusHidden(col + layerInputSize, row);
					}
				}

				lstmLayer->setUVals(hiddenWeights);

				iter += (gateSize * (layerInputSize + hiddenSize));

				// Bias weights
				std::vector<float> biasWeights = std::vector<float>(iter, iter + gateSize);

				lstmLayer->setBVals(biasWeights);

				iter += gateSize;

				// initial internal state values follow here in NAM, but aren't supported by RTNeural

				iter += hiddenSize * 2;	// (hidden state and cell state)

				//// LSTM hidden state
				//auto hiddenState = std::vector<float>(iter, iter + hiddenSize);

				//iter += hiddenSize;

				//// LSTM cell state
				//auto cellState = std::vector<float>(iter, iter + hiddenSize);

				//lstmLayer->setHCVals(hiddenState, cellState);

				//iter += hiddenSize;
			}

			// Dense layer weights
			auto denseLayer = new RTNeural::Dense<float>((int)hiddenSize, (int)networkOutputSize);
			model->addLayer(denseLayer);

			std::vector<std::vector<float>> denseWeights;
			denseWeights.resize(1);
			denseWeights[0] = std::vector<float>(iter, iter + hiddenSize);

			denseLayer->setWeights(denseWeights);

			iter += hiddenSize;

			// Dense layer bias
			auto denseBias = std::vector<float>(iter, iter + networkOutputSize);
			denseLayer->setBias(&(*iter));

			iter += networkOutputSize;

			model->reset();

			return true;
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			for (size_t i = 0; i < numSamples; i++)
				output[i] = model->forward(input + i);
		}

		void Prewarm()
		{
			float sample = 0;

			for (size_t i = 0; i < 2048; i++)
				model->forward(&sample);
		}

	private:
		std::unique_ptr<RTNeural::Model<float>> model;
	};
}
