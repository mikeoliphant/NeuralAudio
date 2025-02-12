#pragma once

#include <Eigen/Dense>
#include "Activation.h"
#include "TemplateHelper.h"

namespace NeuralAudio
{
	struct LSTMLayerDef
	{
		std::vector<float> InputWeights;
		std::vector<float> HiddenWeights;
		std::vector<float> BiasWeights;
	};

	struct LSTMDef
	{
		std::vector<LSTMLayerDef> Layers;
		std::vector<float> HeadWeights;
		float HeadBias;
	};

	template<int InputSize, int HiddenSize>
	class LSTMLayerT
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
		auto GetHiddenState() const { return state(Eigen::placeholders::lastN(HiddenSize)); };

		void SetNAMWeights(std::vector<float>::iterator& weights)
		{
			for (int i = 0; i < inputHiddenWeights.rows(); i++)
				for (int j = 0; j < inputHiddenWeights.cols(); j++)
					inputHiddenWeights(i, j) = *(weights++);

			for (int i = 0; i < bias.size(); i++)
				bias[i] = *(weights++);

			for (int i = 0; i < HiddenSize; i++)
				state[i + InputSize] = *(weights++);

			for (int i = 0; i < HiddenSize; i++)
				cellState[i] = *(weights++);
		}

		void SetWeights(LSTMLayerDef& def)
		{
			std::vector<float>::iterator it = def.InputWeights.begin();

			for (int j = 0; j < InputSize; j++)
				for (int i = 0; i < inputHiddenWeights.rows(); i++)
				{
					inputHiddenWeights(i, j) = *(it++);
				}

			assert(std::distance(def.InputWeights.begin(), it) == (long)def.InputWeights.size());

			it = def.HiddenWeights.begin();

			for (int j = 0; j < HiddenSize; j++)
				for (int i = 0; i < inputHiddenWeights.rows(); i++)
				{
					inputHiddenWeights(i, j + InputSize) = *(it++);
				}

			assert(std::distance(def.HiddenWeights.begin(), it) == (long)def.HiddenWeights.size());

			for (int i = 0; i < bias.rows(); i++)
				bias[i] = def.BiasWeights[i];

			state.setZero();
			cellState.setZero();
		}

		inline void Process(const float* input)
		{
			for (int i = 0; i < InputSize; i++)
				state(i) = input[i];

			gates = (inputHiddenWeights * state) + bias;

			for (auto i = 0; i < HiddenSize; i++)
				cellState[i] = (FastSigmoid(gates[i + fOffset]) * cellState[i]) + (FastSigmoid(gates[i + iOffset]) * FastTanh(gates[i + gOffset]));

			for (int i = 0; i < HiddenSize; i++)
				state[i + hOffset] = FastSigmoid(gates[i + oOffset]) * FastTanh(cellState[i]);
		}
	};

	template<int NumLayers, int HiddenSize>
	class LSTMModelT
	{
	private:
		LSTMLayerT<1, HiddenSize> firstLayer;
		std::vector<LSTMLayerT<HiddenSize, HiddenSize>> remainingLayers;
		Eigen::Vector<float, HiddenSize> headWeights;
		float headBias;

	public:
		LSTMModelT()
		{
			if constexpr (NumLayers > 1)
			{
				remainingLayers.resize(NumLayers - 1);

				ForEachIndex<NumLayers - 1>([&](auto layerIndex)
					{
						(void)layerIndex;

						LSTMLayerT<HiddenSize, HiddenSize> layer;

						remainingLayers.push_back(layer);
					});
			}
		}

		void SetNAMWeights(std::vector<float> weights)
		{
			std::vector<float>::iterator it = weights.begin();

			firstLayer.SetNAMWeights(it);

			ForEachIndex<NumLayers - 1>([&](auto layerIndex)
				{
					remainingLayers[layerIndex].SetNAMWeights(it);
				});

			for (int i = 0; i < HiddenSize; i++)
				headWeights[i] = *(it++);

			headBias = *(it++);

			assert(std::distance(weights.begin(), it) == (long)weights.size());
		}

		void SetWeights(LSTMDef& def)
		{
			for (int i = 0; i < HiddenSize; i++)
				headWeights[i] = def.HeadWeights[i];

			headBias = def.HeadBias;

			firstLayer.SetWeights(def.Layers[0]);

			ForEachIndex<NumLayers - 1>([&](auto layerIndex)
				{
					remainingLayers[layerIndex].SetWeights(def.Layers[layerIndex + 1]);
				});
		}

		void Process(const float* input, float* output, const size_t numSamples)
		{
			for (size_t i = 0; i < numSamples; i++)
			{
				firstLayer.Process(input + i);

				ForEachIndex<NumLayers - 1>([&](auto layerIndex)
					{
						if constexpr (layerIndex == 0)
						{
							remainingLayers[layerIndex].Process(firstLayer.GetHiddenState().data());
						}
						else
						{
							remainingLayers[layerIndex].Process(remainingLayers[layerIndex - 1].GetHiddenState().data());
						}
					});

				if constexpr (NumLayers == 1)
				{
					output[i] = headWeights.dot(firstLayer.GetHiddenState()) + headBias;
				}
				else
				{
					output[i] = headWeights.dot(remainingLayers[NumLayers - 2].GetHiddenState()) + headBias;
				}
			}
		}
	};
}