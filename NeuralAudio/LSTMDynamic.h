#pragma once

#include <cassert>
#include <Eigen/Dense>
#include "Activation.h"
#include "LSTM.h"

namespace NeuralAudio
{
	class LSTMLayer
	{
	private:
		size_t inputSize;
		size_t hiddenSize;
		size_t inputHiddenSize;
		size_t gateSize;
		Eigen::MatrixXf inputHiddenWeights;
		Eigen::VectorXf bias;
		Eigen::VectorXf state;
		Eigen::VectorXf gates;
		Eigen::VectorXf cellState;

		size_t iOffset;
		size_t fOffset;
		size_t gOffset;
		size_t oOffset;
		size_t hOffset;

	public:
		LSTMLayer(size_t inputSize, size_t hiddenSize) :
			inputSize(inputSize),
			hiddenSize(hiddenSize),
			inputHiddenSize(inputSize + hiddenSize),
			gateSize(4 * hiddenSize),
			inputHiddenWeights(gateSize, inputHiddenSize),
			bias(gateSize),
			state(inputHiddenSize),
			gates(gateSize),
			cellState(hiddenSize),
			iOffset(0),
			fOffset(hiddenSize),
			gOffset(2 * hiddenSize),
			oOffset(3 * hiddenSize),
			hOffset(inputSize)
		{
		}

		auto GetHiddenState() const { return state(Eigen::placeholders::lastN(hiddenSize)); };

		void SetNAMWeights(std::vector<float>::iterator& weights)
		{
			for (size_t i = 0; i < gateSize; i++)
				for (size_t j = 0; j < inputHiddenSize; j++)
					inputHiddenWeights(i, j) = *(weights++);

			for (size_t i = 0; i < gateSize; i++)
				bias[i] = *(weights++);

			for (size_t i = 0; i < hiddenSize; i++)
				state[i + inputSize] = *(weights++);

			for (size_t i = 0; i < hiddenSize; i++)
				cellState[i] = *(weights++);
		}

		void SetWeights(LSTMLayerDef& def)
		{
			std::vector<float>::iterator it = def.InputWeights.begin();

			for (size_t j = 0; j < inputSize; j++)
				for (size_t i = 0; i < gateSize; i++)
				{
					inputHiddenWeights(i, j) = *(it++);
				}

			assert(std::distance(def.InputWeights.begin(), it) == (long)def.InputWeights.size());

			it = def.HiddenWeights.begin();

			for (size_t j = 0; j < hiddenSize; j++)
				for (size_t i = 0; i < gateSize; i++)
				{
					inputHiddenWeights(i, j + inputSize) = *(it++);
				}

			assert(std::distance(def.HiddenWeights.begin(), it) == (long)def.HiddenWeights.size());

			for (size_t i = 0; i < gateSize; i++)
				bias[i] = def.BiasWeights[i];

			state.setZero();
			cellState.setZero();
		}

		inline void Process(const float* input)
		{
			for (size_t i = 0; i < inputSize; i++)
				state(i) = input[i];

			gates = (inputHiddenWeights * state) + bias;

			for (size_t i = 0; i < hiddenSize; i++)
				cellState[i] = (LSTM_MATH::Sigmoid(gates[i + fOffset]) * cellState[i]) + (LSTM_MATH::Sigmoid(gates[i + iOffset]) *
					LSTM_MATH::Tanh(gates[i + gOffset]));

			for (size_t i = 0; i < hiddenSize; i++)
				state[i + hOffset] = LSTM_MATH::Sigmoid(gates[i + oOffset]) * LSTM_MATH::Tanh(cellState[i]);
		}
	};

	class LSTMModel
	{
	private:
		size_t numLayers;
		size_t lastLayer;
		size_t hiddenSize;
		std::vector<LSTMLayer> layers;
		Eigen::VectorXf headWeights;
		float headBias;

	public:
		LSTMModel(size_t numLayers, size_t hiddenSize) :
			numLayers(numLayers),
			lastLayer(numLayers - 1),
			hiddenSize(hiddenSize),
			headWeights(hiddenSize)
		{
			layers.push_back(LSTMLayer(1, hiddenSize));
				
			for (size_t i = 0; i < numLayers - 1; i++)
			{
				layers.push_back(LSTMLayer(hiddenSize, hiddenSize));
			}
		}

		void SetNAMWeights(std::vector<float> weights)
		{
			std::vector<float>::iterator it = weights.begin();

			for (auto& layer : layers)
			{
				layer.SetNAMWeights(it);
			}

			for (int i = 0; i < hiddenSize; i++)
				headWeights[i] = *(it++);

			headBias = *(it++);

			assert(std::distance(weights.begin(), it) == (long)weights.size());
		}

		void SetWeights(LSTMDef& def)
		{
			for (size_t i = 0; i < hiddenSize; i++)
				headWeights[i] = def.HeadWeights[i];

			headBias = def.HeadBias;

			for (size_t i = 0; i < numLayers; i++)
			{
				layers[i].SetWeights(def.Layers[i]);
			}
		}

		void Process(const float* input, float* output, const size_t numSamples)
		{
			for (size_t i = 0; i < numSamples; i++)
			{
				layers[0].Process(input + i);

				for (size_t layer = 1; layer < numLayers; layer++)
				{
					layers[layer].Process(layers[layer - 1].GetHiddenState().data());
				}

				output[i] = headWeights.dot(layers[lastLayer].GetHiddenState()) + headBias;
			}
		}
	};
}