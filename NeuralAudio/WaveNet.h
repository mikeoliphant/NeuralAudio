#pragma once

// Based on WaveNet model structure from https://github.com/sdatkinson/NeuralAmpModelerCore
// with some template ideas from https://github.com/jatinchowdhury18/RTNeural-NAM

#include <Eigen/Dense>
#include <Eigen/Core>
#include "TemplateHelper.h"
#include "Activation.h"

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
	class ChannelBuffer
	{
	public:
		static constexpr auto BufferSize = ReceptiveFieldSize + ((LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES);
		static constexpr bool TooBigForStatic = ((Channels * BufferSize) * 4) > EIGEN_STACK_ALLOCATION_LIMIT;

		using ChannelBufferType = typename std::conditional<TooBigForStatic,
			Eigen::Matrix<float, Channels, -1>,
			Eigen::Matrix<float, Channels, BufferSize>>::type;

		ChannelBufferType buffer;
		//Eigen::Matrix<float, Channels, -1> layerBuffer;
		size_t bufferStart;

		void AllocBuffer(int allocNum)
		{
			long size = BufferSize;

			if constexpr (TooBigForStatic)
			{
				buffer.resize(Channels, size);
			}

			buffer.setZero();

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
			bufferStart = size - (WAVENET_MAX_NUM_FRAMES * ((allocNum % LAYER_ARRAY_BUFFER_PADDING) + 1));	// Do the modulo to handle cases where LAYER_ARRAY_BUFFER_PADDING is not big enough to handle offset
#endif
		}

		void AdvanceFrames(const size_t numFrames)
		{
			bufferStart += numFrames;

			if ((bufferStart + WAVENET_MAX_NUM_FRAMES) > (size_t)buffer.cols())
				RewindBuffer();
		}

		void RewindBuffer()
		{
			size_t start = ReceptiveFieldSize;

			buffer.middleCols(start - ReceptiveFieldSize, ReceptiveFieldSize) = buffer.middleCols(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize);

			bufferStart = start;
		}

		void CopyBuffer()
		{
			for (size_t offset = 1; offset < ReceptiveFieldSize + 1; offset++)
			{
				buffer.col(bufferStart - offset) = buffer.col(bufferStart);
			}
		}
	};

	template <int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DT
	{
	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;
		ChannelBuffer<InChannels, ReceptiveFieldSize> channelBuffer;

		size_t GetNumWeights()
		{
			return OutChannels * InChannels * KernelSize + (DoBias ? OutChannels : 0);
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			weights.resize(KernelSize);

			for (size_t i = 0; i < OutChannels; i++)
				for (size_t j = 0; j < InChannels; j++)
					for (size_t k = 0; k < KernelSize; k++)
						weights[k](i, j) = *(inWeights++);

			if (DoBias)
			{
				for (size_t i = 0; i < OutChannels; i++)
					bias(i) = *(inWeights++);
			}
		}

		template<typename Derived>
		inline void Process(Eigen::MatrixBase<Derived> const & output, const size_t numFrames) const
		{
			for (size_t k = 0; k < KernelSize; k++)
			{
				auto offset = Dilation * ((int)k + 1 - KernelSize);

				auto inBlock = channelBuffer.buffer.middleCols(channelBuffer.bufferStart + offset, numFrames);

				if (k == 0)
					const_cast<Eigen::MatrixBase<Derived>&>(output).noalias() = weights[k] * inBlock;
				else
					const_cast<Eigen::MatrixBase<Derived>&>(output).noalias() += weights[k] * inBlock;
			}

			if constexpr (DoBias)
				const_cast<Eigen::MatrixBase<Derived>&>(output).colwise() += bias;
		}

	private:
		std::vector<Eigen::Matrix<float, OutChannels, InChannels>> weights;
		Eigen::Vector<float, OutChannels> bias;
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

		template<typename Derived, typename Derived2>
		void Process(const Eigen::MatrixBase<Derived>& input, Eigen::MatrixBase<Derived2> const& output) const
		{
			if constexpr (DoBias)
			{
				const_cast<Eigen::MatrixBase<Derived2>&>(output).noalias() = (weights * input).colwise() + bias;
			}
			else
			{
				const_cast<Eigen::MatrixBase<Derived2>&>(output).noalias() = weights * input;
			}
		}

		template<typename Derived, typename Derived2>
		void ProcessAcc(const Eigen::MatrixBase<Derived>& input, Eigen::MatrixBase<Derived2> const& output) const
		{
			if constexpr (DoBias)
			{
				const_cast<Eigen::MatrixBase<Derived2>&>(output).noalias() += (weights * input).colwise() + bias;
			}
			else
			{
				const_cast<Eigen::MatrixBase<Derived2>&>(output).noalias() += weights * input;
			}
		}

	private:
		Eigen::Matrix<float, OutSize, InSize> weights;
		Eigen::Vector<float, OutSize> bias;
	};

	template <int ConditionSize, int Channels, int KernelSize, int Dilation, EActivationType Activation>
	class WaveNetLayerT
	{
	private:
		Conv1DT<Channels, Channels, KernelSize, true, Dilation> conv1D;
		DenseLayerT<ConditionSize, Channels, false> inputMixin;
		DenseLayerT<Channels, Channels, true> oneByOne;
		Eigen::Matrix<float, Channels, WAVENET_MAX_NUM_FRAMES> state;

	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;

		WaveNetLayerT()
		{
			state.setZero();
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

		auto& GetInputBuffer()
		{
			return conv1D.channelBuffer;
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			conv1D.SetWeights(weights);
			inputMixin.SetWeights(weights);
			oneByOne.SetWeights(weights);
		}

		template<typename Derived, typename Derived2, typename Derived3>
		void Process(const Eigen::MatrixBase<Derived>& condition, Eigen::MatrixBase<Derived2> const& headInput, Eigen::MatrixBase<Derived3> const& output, const size_t outputStart, const size_t numFrames)
		{
			auto block = state.leftCols(numFrames);

			conv1D.Process(block, numFrames);

			inputMixin.ProcessAcc(condition, block);

			if constexpr (Activation == EActivationType::Tanh)
			{
				WAVENET_MATH::Tanh(&block);
			}
			else if constexpr (Activation == EActivationType::LeakyReLU)
			{
				WAVENET_MATH::LeakyReLU(&block);
			}

			const_cast<Eigen::MatrixBase<Derived2>&>(headInput).leftCols(numFrames).noalias() += block;

			oneByOne.Process(block, const_cast<Eigen::MatrixBase<Derived3>&>(output).middleCols(outputStart, numFrames));

			const_cast<Eigen::MatrixBase<Derived3>&>(output).middleCols(outputStart, numFrames).noalias() += conv1D.channelBuffer.buffer.middleCols(conv1D.channelBuffer.bufferStart, numFrames);
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

		Eigen::Matrix<float, Channels, WAVENET_MAX_NUM_FRAMES> arrayOutputs;
		Eigen::Matrix<float, HeadSize, WAVENET_MAX_NUM_FRAMES> headOutputs;
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

		template<typename Derived, typename Derived2, typename Derived3>
		void Prewarm(const Eigen::MatrixBase<Derived>& layerInputs, const Eigen::MatrixBase<Derived2>& condition, Eigen::MatrixBase<Derived3> const& headInputs)
		{
			rechannel.Process(layerInputs.leftCols(1), std::get<0>(layers).GetInputBuffer().buffer.middleCols(std::get<0>(layers).GetInputBuffer().bufferStart, 1));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).GetInputBuffer().CopyBuffer();

					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs, 0, 1);
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, std::get<layerIndex + 1>(layers).GetInputBuffer().buffer, std::get<layerIndex + 1>(layers).GetInputBuffer().bufferStart, 1);
					}
				});

			headRechannel.channelBuffer.buffer.middleCols(headRechannel.channelBuffer.bufferStart, 1).noalias() = headInputs.leftCols(1);	// Should be able to avoid this copy
			headRechannel.channelBuffer.CopyBuffer();
			headRechannel.Process(headOutputs.leftCols(1), 1);
		}

		template<typename Derived, typename Derived2, typename Derived3>
		void Process(const Eigen::MatrixBase<Derived>& layerInputs, const Eigen::MatrixBase<Derived2>& condition, Eigen::MatrixBase<Derived3> const& headInputs, const size_t numFrames)
		{
			rechannel.Process(layerInputs.leftCols(numFrames), std::get<0>(layers).GetInputBuffer().buffer.middleCols(std::get<0>(layers).GetInputBuffer().bufferStart, numFrames));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs, 0, numFrames);
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs,	std::get<layerIndex + 1>(layers).GetInputBuffer().buffer, std::get<layerIndex + 1>(layers).GetInputBuffer().bufferStart, numFrames);
					}

					std::get<layerIndex>(layers).AdvanceFrames(numFrames);
				});

			headRechannel.channelBuffer.buffer.middleCols(headRechannel.channelBuffer.bufferStart, numFrames).noalias() = headInputs.leftCols(numFrames);	// Should be able to avoid this copy
			headRechannel.Process(headOutputs.leftCols(numFrames), numFrames);
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
			float input = 0;

			auto condition = Eigen::Map<const Eigen::Matrix<float, 1, -1>>(&input, 1, 1);

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					if constexpr (layerIndex == 0)
					{
						std::get<layerIndex>(layerArrays).Prewarm(condition, condition, headArray);
					}
					else
					{
						std::get<layerIndex>(layerArrays).Prewarm(std::get<layerIndex - 1>(layerArrays).arrayOutputs, condition, std::get<layerIndex - 1>(layerArrays).headOutputs);
					}
				});
		}

		void Process(const float* input, float* output, const size_t numFrames)
		{
			//numRewinds = 0;

			auto condition = Eigen::Map<const Eigen::Matrix<float, 1, -1>>(input, 1, numFrames);

			headArray.setZero();

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					if constexpr (layerIndex == 0)
					{
						std::get<layerIndex>(layerArrays).Process(condition, condition, headArray, numFrames);
					}
					else
					{
						std::get<layerIndex>(layerArrays).Process(std::get<layerIndex - 1>(layerArrays).arrayOutputs, condition, std::get<layerIndex - 1>(layerArrays).headOutputs, numFrames);
					}
				});

			const auto finalHeadArray = std::get<sizeof...(LayerArrays) - 1>(layerArrays).headOutputs;

			auto out = Eigen::Map<Eigen::Matrix<float, 1, -1>>(output, 1, numFrames);

			out.noalias() = headScale * finalHeadArray.leftCols(numFrames);

			//if (numRewinds > maxRewinds)
			//{
			//	maxRewinds = numRewinds;

			//	std::cout << "New Max Rewinds: " << maxRewinds << std::endl;
			//}
		}

	private:
		static constexpr auto headLayerChannels = std::tuple_element_t<0, std::tuple<LayerArrays...>>::NumChannelsP;

		std::tuple<LayerArrays...> layerArrays;
		Eigen::Matrix<float, headLayerChannels, WAVENET_MAX_NUM_FRAMES> headArray;
		float headScale;
	};
}
