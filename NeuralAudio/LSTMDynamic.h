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
		int inputSize;
		int hiddenSize;
		Eigen::MatrixXf inputHiddenWeights;
		Eigen::VectorXf bias;
		Eigen::VectorXf state;
		Eigen::VectorXf gates;
		Eigen::VectorXf cellState;

		long iOffset;
		long fOffset;
		long gOffset;
		long oOffset;
		long hOffset;

	public:
		LSTMLayer(int inputSize, int hiddenSize) :
			inputSize(inputSize),
			hiddenSize(hiddenSize),
			inputHiddenWeights(4 * hiddenSize, inputSize + hiddenSize),
			bias(4 * hiddenSize),
			state(inputSize + hiddenSize),
			gates(4 * hiddenSize),
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
			for (int i = 0; i < inputHiddenWeights.rows(); i++)
				for (int j = 0; j < inputHiddenWeights.cols(); j++)
					inputHiddenWeights(i, j) = *(weights++);

			for (int i = 0; i < bias.size(); i++)
				bias[i] = *(weights++);

			for (int i = 0; i < hiddenSize; i++)
				state[i + inputSize] = *(weights++);

			for (int i = 0; i < hiddenSize; i++)
				cellState[i] = *(weights++);
		}

		void SetWeights(LSTMLayerDef& def)
		{
			std::vector<float>::iterator it = def.InputWeights.begin();

			for (int j = 0; j < inputSize; j++)
				for (int i = 0; i < inputHiddenWeights.rows(); i++)
				{
					inputHiddenWeights(i, j) = *(it++);
				}

			assert(std::distance(def.InputWeights.begin(), it) == (long)def.InputWeights.size());

			it = def.HiddenWeights.begin();

			for (int j = 0; j < hiddenSize; j++)
				for (int i = 0; i < inputHiddenWeights.rows(); i++)
				{
					inputHiddenWeights(i, j + inputSize) = *(it++);
				}

			assert(std::distance(def.HiddenWeights.begin(), it) == (long)def.HiddenWeights.size());

			for (int i = 0; i < bias.rows(); i++)
				bias[i] = def.BiasWeights[i];

			state.setZero();
			cellState.setZero();
		}

		inline void Process(const float* input)
		{
			for (int i = 0; i < inputSize; i++)
				state(i) = input[i];

			gates = (inputHiddenWeights * state) + bias;

			for (auto i = 0; i < hiddenSize; i++)
				cellState[i] = (FastSigmoid(gates[i + fOffset]) * cellState[i]) + (FastSigmoid(gates[i + iOffset]) * FastTanh(gates[i + gOffset]));

			for (int i = 0; i < hiddenSize; i++)
				state[i + hOffset] = FastSigmoid(gates[i + oOffset]) * FastTanh(cellState[i]);
		}
	};

	class LSTMModel
	{
	private:
		int numLayers;
		int lastLayer;
		int hiddenSize;
		std::vector<LSTMLayer> layers;
		Eigen::VectorXf headWeights;
		float headBias;

	public:
		LSTMModel(int numLayers, int hiddenSize) :
			numLayers(numLayers),
			lastLayer(numLayers - 1),
			hiddenSize(hiddenSize),
			headWeights(hiddenSize)
		{
			layers.push_back(LSTMLayer(1, hiddenSize));
				
			for (auto i = 0; i < numLayers - 1; i++)
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
			for (int i = 0; i < hiddenSize; i++)
				headWeights[i] = def.HeadWeights[i];

			headBias = def.HeadBias;

			for (int i = 0; i < numLayers; i++)
			{
				layers[i].SetWeights(def.Layers[i]);
			}
		}

		void Process(const float* input, float* output, const int numSamples)
		{
			for (auto i = 0; i < numSamples; i++)
			{
				layers[0].Process(input + i);

				for (int i = 1; i < numLayers; i++)
				{
					layers[i].Process(layers[i - 1].GetHiddenState().data());
				}

				output[i] = headWeights.dot(layers[lastLayer].GetHiddenState()) + headBias;
			}
		}
	};
}