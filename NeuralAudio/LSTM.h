#pragma once

#include <Eigen/Dense>
#include "Activation.h"

namespace NeuralAudio
{
	template<int InputSize, int HiddenSize>
	class LSTMLayer
	{
	private:
		Eigen::Matrix<float, 4 * HiddenSize, InputSize + HiddenSize>  inputHiddenWeights;
		Eigen::Vector<float, 4 * HiddenSize> bias;
		Eigen::Vector<float, InputSize + HiddenSize> state;
		Eigen::Vector<float, 4 * HiddenSize> gates;
		Eigen::Vector<float, HiddenSize> cellState;

		constexpr static long iOffset = 0;
		constexpr static long fOffset = HiddenSize;
		constexpr static long gOffset = 2 * HiddenSize;
		constexpr static long oOffset = 3 * HiddenSize;
		constexpr static long hOffset = InputSize;

	public:
		auto GetHiddenState() const { return this->state(Eigen::placeholders::lastN(HiddenSize)); };

		void SetWeights(std::vector<float>::iterator& weights)
		{
			for (int i = 0; i < inputHiddenWeights.rows(); i++)
				for (int j = 0; j < inputHiddenWeights.cols(); j++)
					inputHiddenWeights(i, j) = *(weights++);

			for (int i = 0; i < bias.size(); i++)
				bias[i] = *(weights++);

			for (int i = 0; i < HiddenSize; i++)
				this->state[i + InputSize] = *(weights++);

			for (int i = 0; i < HiddenSize; i++)
				this->cellState[i] = *(weights++);
		}

		inline void Process(const float* input)
		{
			for (int i = 0; i < InputSize; i++)
				state(i) = input[i];

			gates = (inputHiddenWeights * state) + bias;

			for (auto i = 0; i < HiddenSize; i++)
				cellState[i] = (FastSigmoid(gates[i + fOffset]) * cellState[i]) + (FastSigmoid(gates[i + iOffset]) * FastTanh(this->gates[i + gOffset]));

			for (int i = 0; i < HiddenSize; i++)
				this->state[i + hOffset] = FastSigmoid(gates[i + oOffset]) * FastTanh(cellState[i]);
		}
	};

	template<int HiddenSize>
	class LSTMModel
	{
	private:
		LSTMLayer<1, HiddenSize> layer;
		Eigen::Vector<float, HiddenSize> headWeights;
		float headBias;

	public:
		void SetWeights(std::vector<float> weights)
		{
			std::vector<float>::iterator it = weights.begin();

			layer.SetWeights(it);

			for (int i = 0; i < HiddenSize; i++)
				headWeights[i] = *(it++);

			headBias = *(it++);

			assert(std::distance(weights.begin(), it) == weights.size());
		}

		void Process(const float* input, float* output, const int numSamples)
		{
			for (auto i = 0; i < numSamples; i++)
			{
				layer.Process(input + i);

				output[i] = headWeights.dot(layer.GetHiddenState()) + headBias;
			}
		}
	};
}