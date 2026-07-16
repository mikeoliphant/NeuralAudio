#pragma once

// Based on WaveNet model structure from https://github.com/sdatkinson/NeuralAmpModelerCore
// with some template ideas from https://github.com/jatinchowdhury18/RTNeural-NAM

#include <array>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "TemplateHelper.h"
#include "Activation.h"
#include "ChannelBuffer.h"
#include "MatMul.h"

#ifndef WAVENET_MAX_NUM_FRAMES
#define WAVENET_MAX_NUM_FRAMES 64
#endif

#ifndef LAYER_ARRAY_BUFFER_PADDING
#define LAYER_ARRAY_BUFFER_PADDING 24
#endif

enum EActivationType
{
	Tanh,
	LeakyReLU
};

namespace NeuralAudio
{
	template <int Channels, int ReceptiveFieldSize>
	class ChannelHistoryBuffer
	{
	public:
		static constexpr auto BufferSize = ReceptiveFieldSize + ((LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES);

		ChannelBuffer<float, Channels, BufferSize> buffer;
		size_t bufferStart;

		void AllocBuffer(int allocNum)
		{
			buffer.SetZero();

			//if (offset > (size - (ReceptiveFieldSize + WAVENET_MAX_NUM_FRAMES)))
			//{
			//	bufferStart = ReceptiveFieldSize;
			//}
			//else
			//{
			//	bufferStart = size - offset;
			//}

#if (LAYER_ARRAY_BUFFER_PADDING == 0)
			bufferStart = ReceptiveFieldSize;
#else
			bufferStart = BufferSize - (WAVENET_MAX_NUM_FRAMES * ((allocNum % LAYER_ARRAY_BUFFER_PADDING) + 1));	// Do the modulo to handle cases where LAYER_ARRAY_BUFFER_PADDING is not big enough to handle offset
#endif
		}

		void AdvanceFrames(const size_t numFrames)
		{
			bufferStart += numFrames;

			if ((bufferStart + WAVENET_MAX_NUM_FRAMES) > (size_t)buffer.GetNumCols())
				RewindBuffer();
		}

		void RewindBuffer()
		{
			buffer.Slice(0, ReceptiveFieldSize).CopyData(buffer.Slice(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize));

			bufferStart = ReceptiveFieldSize;
		}

		void CopyBuffer()
		{
			auto slice = buffer.Slice(bufferStart, 1);

			for (size_t offset = 1; offset < ReceptiveFieldSize + 1; offset++)
			{
				buffer.Slice(bufferStart - offset, 1).CopyData(slice);
			}
		}
	};

	struct Empty {};

	template <int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DT
	{
	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;
		ChannelHistoryBuffer<InChannels, ReceptiveFieldSize> channelBuffer;

		size_t GetNumWeights()
		{
			return OutChannels * InChannels * KernelSize + (DoBias ? OutChannels : 0);
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			for (size_t i = 0; i < OutChannels; i++)
				for (size_t j = 0; j < InChannels; j++)
					for (size_t k = 0; k < KernelSize; k++)
						weights[k](i, j) = *(inWeights++);

			if constexpr (DoBias)
			{
				for (size_t i = 0; i < OutChannels; i++)
					bias(i) = *(inWeights++);
			}
		}

		Conv1DT()
		{
			for (int k = 0; k < KernelSize; k++)
				weightPtrs[k] = weights[k].GetData();
		}

		auto GetInputBuffer(size_t numFrames)
		{
			return channelBuffer.buffer.Slice(channelBuffer.bufferStart, numFrames);
		}

		inline void Process(const ChannelRowSpan<float, OutChannels>& output)
		{
			const size_t numFrames = output.GetNumCols();
			float* __restrict outputPtr = output.GetData();

#if ENABLE_MULTIFRAME_8X8_CONVOLUTION
			if constexpr ((InChannels == 8) && (OutChannels == 8) && DoBias)
			{
				// Based on @jfsantos NAM Core implementation - https://github.com/sdatkinson/NeuralAmpModelerCore/pull/277

				constexpr size_t tileSize = 4;
				const size_t nF4 = (numFrames / tileSize) * tileSize;

				for (size_t f = 0; f < nF4; f += tileSize)
				{
					alignas(32) float a[tileSize][InChannels]{};

					for (size_t k = 0; k < KernelSize; k++)
					{
						const float* __restrict W = weightPtrs[k];
						const auto offset = Dilation * (k + 1 - KernelSize);
						const float* __restrict hb = channelBuffer.buffer.GetDataConst(channelBuffer.bufferStart + offset + f);

						for (size_t cp = 0; cp < InChannels; cp++)
						{
							const float* __restrict Wcol = W + cp * InChannels;
							const float h0 = hb[cp], h1 = hb[InChannels + cp], h2 = hb[2 * InChannels + cp], h3 = hb[3 * InChannels + cp];

							for (size_t o = 0; o < InChannels; o++)
							{
								const float wo = Wcol[o];
								a[0][o] += wo * h0;
								a[1][o] += wo * h1;
								a[2][o] += wo * h2;
								a[3][o] += wo * h3;
							}
						}
					}

					for (size_t ti = 0; ti < tileSize; ti++)
						std::memcpy(outputPtr + static_cast<size_t>(f + ti) * InChannels, a[ti], InChannels * sizeof(float));
				}

				// Scalar tail for any frames past the tile-aligned boundary.
				for (size_t f = nF4; f < numFrames; f++)
				{
					float* zf = outputPtr + static_cast<size_t>(f) * InChannels;

					for (size_t o = 0; o < InChannels; o++)
						zf[o] = 0.0f;

					for (size_t k = 0; k < KernelSize; k++)
					{
						const float* W = weightPtrs[k];
						const auto offset = Dilation * (k + 1 - KernelSize);
						const float* h = channelBuffer.buffer.GetDataConst(channelBuffer.bufferStart + offset + f);

						for (int cp = 0; cp < InChannels; cp++)
						{
							const float hv = h[cp];
							const float* Wcol = W + cp * InChannels;

							for (size_t o = 0; o < InChannels; o++)
								zf[o] += Wcol[o] * hv;
						}
					}
				}
			}
			else
#endif
			{
				const float* biasPtr = nullptr;
				
				if constexpr (DoBias)
				{
					biasPtr = bias.data();
				}

				for (size_t k = 0; k < KernelSize; k++)
				{
					const float* weightPtr = this->weights[k].GetDataConst();

					const auto offset = Dilation * ((int)k + 1 - KernelSize);

					if constexpr (DoBias && MatMul<InChannels, OutChannels>::HasKernel())
					{
						const float* inputPtr = channelBuffer.buffer.GetDataConst(channelBuffer.bufferStart + offset);

						if (k == 0)	// Maybe move this out of loop?
						{
							if constexpr (DoBias)
							{
								MatMul<InChannels, OutChannels>::MultiplyInitColwise(inputPtr, outputPtr, weightPtr, biasPtr, numFrames);
							}
							else
							{
								MatMul<InChannels, OutChannels>::MultiplyInitZero(inputPtr, outputPtr, weightPtr, numFrames);
							}
						}
						else
						{
							MatMul<InChannels, OutChannels>::MultiplyAccumlulate(inputPtr, outputPtr, weightPtr, numFrames);
						}
					}
					else
					{
						const auto inBlock = channelBuffer.buffer.Slice(channelBuffer.bufferStart + offset, numFrames);

						if (k == 0)
							output.GetEigenMap().noalias() = weights[k].GetEigenMapConst() * inBlock.GetEigenMapConst();
						else
							output.GetEigenMap().noalias() += weights[k].GetEigenMapConst() * inBlock.GetEigenMapConst();
					}
				}
			}

			if constexpr (DoBias && !MatMul<InChannels, OutChannels>::HasKernel())
				output.GetEigenMap().colwise() += bias;
		}

	private:
		alignas(32) std::array<ChannelBuffer<float, OutChannels, InChannels>, KernelSize> weights;	// consider making this a contiguous block of data instead of block of ChannelBuffers
		std::array<float *, KernelSize> weightPtrs;

		// Avoid allocation for unused bias
		using BiasType = typename std::conditional<DoBias,
			Eigen::Vector<float, OutChannels>,
			Empty>::type;

		BiasType bias;
	};

	template <int InSize, int OutSize, bool DoBias>
	class DenseLayerT
	{
	public:
		size_t GetNumWeights()
		{
			return OutSize * InSize + (DoBias ? OutSize : 0);
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			for (size_t i = 0; i < OutSize; i++)
				for (size_t j = 0; j < InSize; j++)
					weights(i, j) = *(inWeights++);

			if constexpr (DoBias)
			{
				for (size_t i = 0; i < OutSize; i++)
					bias(i) = *(inWeights++);
			}
		}

		void Process(const ChannelRowSpan<float, InSize>& input, const ChannelRowSpan<float, OutSize>& output) const
		{
			size_t numFrames = output.GetNumCols();

			if constexpr (MatMul<InSize, OutSize>::HasKernel())
			{
				if constexpr (DoBias)
				{
					MatMul<InSize, OutSize>::MultiplyInitColwise(input.GetDataConst(), output.GetData(), weights.GetDataConst(), bias.data(), numFrames);
				}
				else
				{
					MatMul<InSize, OutSize>::MultiplyInitZero(input.GetDataConst(), output.GetData(), weights.GetDataConst(), numFrames);
				}
			}
			else
			{
				if constexpr (DoBias)
				{
					output.GetEigenMap().noalias() = (weights.GetEigenMapConst() * input.GetEigenMapConst()).colwise() + bias;
				}
				else
				{
					output.GetEigenMap().noalias() = weights.GetEigenMapConst() * input.GetEigenMapConst();
				}
			}
		}

		void ProcessAcc(const ChannelRowSpan<float, InSize>& input, const ChannelRowSpan<float, OutSize>& output) const
		{
			size_t numFrames = output.GetNumCols();

			if constexpr (!DoBias && MatMul<InSize, OutSize>::HasKernel())
			{
				MatMul<InSize, OutSize>::MultiplyAccumlulate(input.GetDataConst(), output.GetData(), weights.GetDataConst(), numFrames);
			}
			else
			{
				if constexpr (DoBias)
				{
					output.GetEigenMap().noalias() += (weights.GetEigenMapConst() * input.GetEigenMapConst()) + bias;
				}
				else
				{
					output.GetEigenMap().noalias() += weights.GetEigenMapConst() * input.GetEigenMapConst();
				}
			}
		}

	private:
		ChannelBuffer<float, OutSize, InSize> weights;

		// Avoid allocation for unused bias
		using BiasType = typename std::conditional<DoBias,
			Eigen::Vector<float, OutSize>,
			Empty>::type;
		
		BiasType bias;
	};

	template <int ConditionSize, int Channels, int KernelSize, int Dilation, EActivationType Activation>
	class WaveNetLayerT
	{
	private:
		Conv1DT<Channels, Channels, KernelSize, true, Dilation> conv1D;
		DenseLayerT<ConditionSize, Channels, false> inputMixin;
		DenseLayerT<Channels, Channels, true> oneByOne;
		ChannelBuffer<float, Channels, WAVENET_MAX_NUM_FRAMES> state;

	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;

		WaveNetLayerT()
		{
			state.SetZero();
		}

		void AllocBuffer(int allocNum)
		{
			conv1D.channelBuffer.AllocBuffer(allocNum);
		}

		void AdvanceFrames(const size_t numFrames)
		{
			conv1D.channelBuffer.AdvanceFrames(numFrames);
		}

		size_t GetNumWeights()
		{
			return conv1D.GetNumWeights() + inputMixin.GetNumWeights() + oneByOne.GetNumWeights();
		}

		auto GetInputBuffer(size_t numFrames)
		{
			return conv1D.channelBuffer.buffer.Slice(conv1D.channelBuffer.bufferStart, numFrames);
		}

		void CopyBuffer()
		{
			conv1D.channelBuffer.CopyBuffer();
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			conv1D.SetWeights(weights);
			inputMixin.SetWeights(weights);
			oneByOne.SetWeights(weights);
		}

		void Process(const ChannelRowSpan<float, ConditionSize>& condition, const ChannelRowSpan<float, Channels>& headInput, const ChannelRowSpan<float, Channels>& output)
		{
			size_t numFrames = output.GetNumCols();

			auto block = state.Slice(numFrames);

			conv1D.Process(block);

			inputMixin.ProcessAcc(condition, block);

			if constexpr (Activation == EActivationType::Tanh)
			{
				WAVENET_MATH::Tanh(block);
			}
			else if constexpr (Activation == EActivationType::LeakyReLU)
			{
				WAVENET_MATH::LeakyReLU(block);
			}

			headInput.GetEigenMap().noalias() += block.GetEigenMapConst();

			//headInput.AddData(block);

			oneByOne.Process(block, output);

			output.GetEigenMap().noalias() += conv1D.GetInputBuffer(numFrames).GetEigenMapConst();

			//output.AddData(conv1D.GetInputBuffer(numFrames));
		}
	};

	template <int... values>
		using Dilations = std::integer_sequence<int, values...>;

	template <int... values>
		using KernelSizes = std::integer_sequence<int, values...>;

	template <int InputSize, int ConditionSize, int HeadSize, int HeadKernelSize, int HeadDilation, int Channels, typename KernelSizeSequence, typename DilationsSequence, bool HasHeadBias, EActivationType Activation>
	class WaveNetLayerArrayT
	{
		template <typename, typename>
		struct LayersHelper
		{
		};

		template <int... dilationVals, int... kernelSizeVals>
		struct LayersHelper<KernelSizes<kernelSizeVals...>, Dilations<dilationVals...>>
		{
			using type = std::tuple<WaveNetLayerT<ConditionSize, Channels, kernelSizeVals, dilationVals, Activation>...>;
		};

		using Layers = typename LayersHelper<KernelSizeSequence, DilationsSequence>::type;

	private:
		Layers layers;
		DenseLayerT<InputSize, Channels, false> rechannel;
		Conv1DT<Channels, HeadSize, HeadKernelSize, HasHeadBias, HeadDilation> headRechannel;

		static constexpr auto numLayers = std::tuple_size_v<decltype (layers)>;
		static constexpr auto lastLayer = numLayers - 1;

	public:
		static constexpr auto NumChannelsP = Channels;
		static constexpr auto HeadSizeP = HeadSize;

		ChannelBuffer<float, Channels, WAVENET_MAX_NUM_FRAMES> arrayOutputs;
		ChannelBuffer<float, HeadSize, WAVENET_MAX_NUM_FRAMES> headOutputs;
		int ReceptiveFieldSize = 0;	// This should be a static constexpr, but I haven't sorted out the right template magic

		WaveNetLayerArrayT()
		{
			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					ReceptiveFieldSize += std::get<layerIndex>(layers).ReceptiveFieldSize;
				});

			ReceptiveFieldSize += headRechannel.ReceptiveFieldSize;
		}

		int AllocBuffers(int allocNum)
		{
			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).AllocBuffer(allocNum++);
				});

			headRechannel.channelBuffer.AllocBuffer(allocNum++);

			return allocNum;
		}

		size_t GetNumWeights()
		{
			size_t numWeights = rechannel.GetNumWeights();

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					numWeights += std::get<layerIndex>(layers).GetNumWeights();
				});

			numWeights += headRechannel.GetNumWeights();

			return numWeights;
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			rechannel.SetWeights(weights);

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).SetWeights(weights);
				});

			headRechannel.SetWeights(weights);
		}

		void Prewarm(const ChannelRowSpan<float, InputSize>& layerInputs, const ChannelRowSpan<float, ConditionSize>& condition, const ChannelRowSpan<float, Channels>& headInputs)
		{
			rechannel.Process(layerInputs, std::get<0>(layers).GetInputBuffer(1));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).CopyBuffer();

					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs.Slice(1));
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, std::get<layerIndex + 1>(layers).GetInputBuffer(1));
					}
				});

			//headRechannel.channelBuffer.buffer.middleCols(headRechannel.channelBuffer.bufferStart, 1).noalias() = headInputs.leftCols(1);	// Should be able to avoid this copy

			headRechannel.GetInputBuffer(1).CopyData(headInputs.Slice(1));
			headRechannel.channelBuffer.CopyBuffer();
			headRechannel.Process(headOutputs.Slice(1));
		}

		void Process(const ChannelRowSpan<float, InputSize>& layerInputs, const ChannelRowSpan<float, ConditionSize>& condition, const ChannelRowSpan<float, Channels>& headInputs)
		{
			size_t numFrames = condition.GetNumCols();

			rechannel.Process(layerInputs, std::get<0>(layers).GetInputBuffer(numFrames));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs.Slice(numFrames));
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs,	std::get<layerIndex + 1>(layers).GetInputBuffer(numFrames));
					}

					std::get<layerIndex>(layers).AdvanceFrames(numFrames);
				});


			//headRechannel.channelBuffer.buffer.middleCols(headRechannel.channelBuffer.bufferStart, numFrames).noalias() = headInputs.leftCols(numFrames);	// Should be able to avoid this copy

			//headRechannel.GetInputBuffer(numFrames).CopyData(headInputs);

			headRechannel.GetInputBuffer(numFrames).GetEigenMap().noalias() = headInputs.GetEigenMapConst();
			headRechannel.Process(headOutputs.Slice(numFrames));
			headRechannel.channelBuffer.AdvanceFrames(numFrames);
		}
	};

	template <typename... LayerArrays>
	class WaveNetModelT
	{
	public:
		int ReceptiveFieldSize = 0;	// This should be a static constexpr, but I haven't sorted out the right template magic

		WaveNetModelT()
		{
			int allocNum = 0;

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					ReceptiveFieldSize += std::get<layerIndex>(layerArrays).ReceptiveFieldSize;

					allocNum = std::get<layerIndex>(layerArrays).AllocBuffers(allocNum);
				});
		}

		size_t GetNumWeights()
		{
			size_t numWeights = 0;

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					numWeights += std::get<layerIndex>(layerArrays).GetNumWeights();
				});

			numWeights++; // headScale;

			return numWeights;
		}

		void SetWeights(std::vector<float> weights)
		{
			size_t numWeights = GetNumWeights();

			if (numWeights != weights.size())
			{
				std::stringstream str;
				str << "Wrong number of weights. Expected " << numWeights << " but got " << weights.size();
				throw std::runtime_error(str.str());
			}

			std::vector<float>::iterator it = weights.begin();

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					std::get<layerIndex>(layerArrays).SetWeights(it);
				});

			headScale = *(it++);
		}

		size_t GetMaxFrames()
		{
			return WAVENET_MAX_NUM_FRAMES;
		}

		void Prewarm()
		{
			condition.GetData()[0] = 0;

			headArray.SetZero();

			auto conditionSpan = condition.Slice(1);
			auto headArraySpan = headArray.Slice(1);

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					if constexpr (layerIndex == 0)
					{
						std::get<layerIndex>(layerArrays).Prewarm(conditionSpan, conditionSpan, headArraySpan);
					}
					else
					{
						std::get<layerIndex>(layerArrays).Prewarm(std::get<layerIndex - 1>(layerArrays).arrayOutputs.Slice(1), conditionSpan, std::get<layerIndex - 1>(layerArrays).headOutputs.Slice(1));
					}
				});
		}

		void Process(const float* input, float* output, const size_t numFrames)
		{
			std::memcpy(condition.GetData(), input, numFrames * sizeof(float));

			headArray.SetZero();

			auto conditionSpan = condition.Slice(numFrames);
			auto headArraySpan = headArray.Slice(numFrames);

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					if constexpr (layerIndex == 0)
					{
						std::get<layerIndex>(layerArrays).Process(conditionSpan, conditionSpan, headArraySpan);
					}
					else
					{
						std::get<layerIndex>(layerArrays).Process(std::get<layerIndex - 1>(layerArrays).arrayOutputs.Slice(numFrames), conditionSpan, std::get<layerIndex - 1>(layerArrays).headOutputs.Slice(numFrames));
					}
				});

			const float* finalHeadArray = std::get<sizeof...(LayerArrays) - 1>(layerArrays).headOutputs.GetData();

			for (size_t i = 0; i < numFrames; i++)
			{
				output[i] = headScale * finalHeadArray[i];
			}
		}

	private:
		static constexpr auto headLayerChannels = std::tuple_element_t<0, std::tuple<LayerArrays...>>::NumChannelsP;

		std::tuple<LayerArrays...> layerArrays;
		ChannelBuffer<float, 1, WAVENET_MAX_NUM_FRAMES> condition;
		ChannelBuffer<float, headLayerChannels, WAVENET_MAX_NUM_FRAMES> headArray;
		float headScale;
	};
}
