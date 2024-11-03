#pragma once

#include "NeuralModel.h"
#include <RTNeural/RTNeural.h>

namespace NeuralAudio
{
	struct DefaultMathsProvider
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
		virtual float GetRecommendedInputDBAdjustment()
		{
			return inputGain;
		}

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return outputGain;
		}

		float GetSampleRate()
		{
			return sampleRate;
		}

		bool LoadFromKerasJson(nlohmann::json& modelJson)
		{
			if (modelJson.contains("samplerate"))
			{
				sampleRate = modelJson["samplerate"];
			}

			if (modelJson.contains("in_gain"))
			{
				inputGain = modelJson["in_gain"];
			}

			if (modelJson.contains("out_gain"))
			{
				outputGain = modelJson["out_gain"];
			}

			return CreateModelFromKerasJson(modelJson);

			return true;
		}

		virtual bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			return false;
		}

		virtual bool LoadFromNAMJson(nlohmann::json& modelJson)
		{
			if (modelJson.contains("samplerate"))
			{
				sampleRate = modelJson["samplerate"];
			}

			if (modelJson.contains("in_gain"))
			{
				inputGain = modelJson["in_gain"];
			}

			if (modelJson.contains("out_gain"))
			{
				outputGain = modelJson["out_gain"];
			}

			return CreateModelFromNAMJson(modelJson);
		}

		virtual bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			return false;
		}

	protected:
		float sampleRate = 48000;
		float inputGain = 0;
		float outputGain = 0;
	};

	template <std::size_t ... Is, typename F>
	void ForEachIndex(std::index_sequence<Is...>, F&& f)
	{
		int dummy[] = { 0, /* Handles empty Is. following cast handle evil operator comma */
					   (static_cast<void>(f(std::integral_constant<std::size_t, Is>())), 0)... };
		static_cast<void>(dummy); // avoid warning for unused variable
	}

	template <std::size_t N, typename F>
	void ForEachIndex(F&& f)
	{
		ForEachIndex(std::make_index_sequence<N>(), std::forward<F>(f));
	}

	template <int numLayers, int hiddenSize>
	class RTNeuralModelT : public RTNeuralModel
	{
		using ModelType = typename std::conditional<numLayers == 1,
			RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, hiddenSize>, RTNeural::DenseT<float, hiddenSize, 1>>,
			RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, hiddenSize>, RTNeural::LSTMLayerT<float, hiddenSize, hiddenSize>, RTNeural::DenseT<float, hiddenSize, 1>>
		>::type;

	public:
		RTNeuralModelT()
			: model(nullptr)
		{
		}

		~RTNeuralModelT()
		{
			if (model != nullptr)
			{
				delete model;
				model == nullptr;
			}
		}

		bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			model->parseJson(modelJson, true);
			model->reset();

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

			std::vector<float> weights = modelJson["weights"];

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

		float GetRecommendedOutputDBAdjustment()
		{
			return 0;
		}

		void Process(float* input, float* output, int numSamples)
		{
			for (int i = 0; i < numSamples; i++)
				output[i] = model->forward(input + i);
		}

	private:
		ModelType* model = nullptr;
	};

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

		bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			model = RTNeural::json_parser::parseJson<float>(modelJson, true);
			model->reset();

			return true;
		}

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			model = std::make_unique<RTNeural::Model<float>>(1);

			nlohmann::json config = modelJson["config"];

			const int numLayers = config["num_layers"];
			const int inputSize = config["input_size"];
			const int hiddenSize = config["hidden_size"];

			std::vector<float> weights = modelJson["weights"];

			const int networkInputSize = 1;
			const int networkOutputSize = 1;
			const int gateSize = 4 * hiddenSize;

			auto iter = weights.begin();

			for (int layer = 0; layer < numLayers; layer++)
			{
				const int layerInputSize = (layer == 0) ? networkInputSize : hiddenSize;

				Eigen::MatrixXf inputPlusHidden = Eigen::Map<Eigen::MatrixXf>(&(*iter), layerInputSize + hiddenSize, gateSize);

				auto lstmLayer = new RTNeural::LSTMLayer<float>(layerInputSize, hiddenSize);

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
			}

			// Dense layer weights
			auto denseLayer = new RTNeural::Dense<float>(hiddenSize, networkOutputSize);
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

		void Process(float* input, float* output, int numSamples)
		{
			for (int i = 0; i < numSamples; i++)
				output[i] = model->forward(input + i);
		}

	private:
		std::unique_ptr<RTNeural::Model<float>> model;
	};

	class RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
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

	template <int numLayers, int hiddenSize>
	class RTNeuralModelDefinitionT : public RTNeuralModelDefinitionBase
	{
	public:
		RTNeuralModel* CreateModel()
		{
			return new RTNeuralModelT<numLayers, hiddenSize>;
		}

		int GetNumLayers()
		{
			return numLayers;
		}

		int GetHiddenSize()
		{
			return hiddenSize;
		}
	};
}
