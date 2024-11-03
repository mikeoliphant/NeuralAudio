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
		virtual bool LoadFromKerasJson(nlohmann::json& modelJson)
		{
			return false;
		}

		virtual bool LoadFromNAMJson(nlohmann::json& modelJson)
		{
			return false;
		}
	};

	template <int numLayers, int hiddenSize>
	class RTNeuralModelT : public RTNeuralModel
	{
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

		bool LoadFromKerasJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, numLayers, hiddenSize>, RTNeural::DenseT<float, hiddenSize, 1>>();

			model->parseJson(modelJson, true);
			model->reset();

			return true;
		}

		bool LoadFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, numLayers, hiddenSize>, RTNeural::DenseT<float, hiddenSize, 1>>();

			nlohmann::json config = modelJson["config"];

			std::vector<float> weights = modelJson["weights"];

			const int networkInputSize = 1;
			const int networkOutputSize = 1;
			const int gateSize = 4 * hiddenSize;

			auto iter = weights.begin();

			for (int layer = 0; layer < numLayers; layer++)
			{
				const int layerInputSize = (layer == 0) ? networkInputSize : hiddenSize;

				Eigen::MatrixXf inputPlusHidden = Eigen::Map<Eigen::MatrixXf>(&(*iter), layerInputSize + hiddenSize, gateSize);

				auto& lstmLayer = model->get<0>();

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

				// hidden state values follow here in NAM, but aren't supported by RTNeural
				iter += hiddenSize * 2;	// (hidden state and cell state)
			}

			// Dense layer weights
			auto& denseLayer = model->get<1>();

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
		RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, numLayers, hiddenSize>, RTNeural::DenseT<float, hiddenSize, 1>>* model = nullptr;
	};

	class RTNeuralModelDyn : public NeuralModel
	{
	public:
		RTNeuralModelDyn()
			: model(nullptr)
		{
		}

		~RTNeuralModelDyn()
		{
			if (model)
				model.reset();
		}

		bool LoadFromKerasJson(nlohmann::json& modelJson)
		{
			model = RTNeural::json_parser::parseJson<float>(modelJson, true);
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
		std::unique_ptr<RTNeural::Model<float>> model = nullptr;
	};

	class RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
		{
			return nullptr;
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
	};
}
