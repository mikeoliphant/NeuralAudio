#pragma once

// Based on WaveNet model structure from https://github.com/sdatkinson/NeuralAmpModelerCore
// with some template ideas from https://github.com/jatinchowdhury18/RTNeural-NAM

#include <Eigen/Dense>
#include <Eigen/Core>
#include "TemplateHelper.h"
#include "NeuralModel.h"
#include "Activation.h"

#ifndef WAVENET_MAX_NUM_FRAMES
#define WAVENET_MAX_NUM_FRAMES 64
#endif

namespace NeuralAudio
{
	//int numRewinds = 0;
	//int maxRewinds = 0;

	template <int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DT
	{
	public:
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

		template<typename Derived, typename Derived2>
		inline void Process(const Eigen::MatrixBase<Derived>& input, Eigen::MatrixBase<Derived2> const & output, const size_t iStart, const size_t nCols) const
		{
			for (size_t k = 0; k < KernelSize; k++)
			{
				auto offset = Dilation * ((int)k + 1 - KernelSize);

				auto inBlock = input.middleCols(iStart + offset, nCols);

				if (k == 0)
					const_cast<Eigen::MatrixBase<Derived2>&>(output).noalias() = weights[k] * inBlock;
				else
					const_cast<Eigen::MatrixBase<Derived2>&>(output).noalias() += weights[k] * inBlock;
			}

			if constexpr (DoBias)
				const_cast<Eigen::MatrixBase<Derived2>&>(output).colwise() += bias;
		}

	private:
		std::vector<Eigen::Matrix<float, OutChannels, InChannels>> weights;
		Eigen::Vector<float, OutChannels> bias;
	};

	template <int InSize, int OutSize, bool DoBias>
	class DenseLayerT
	{
	public:
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

	template <int ConditionSize, int Channels, int KernelSize, int Dilation>
	class WaveNetLayerT
	{
	private:
		Conv1DT<Channels, Channels, KernelSize, true, Dilation> conv1D;
		DenseLayerT<ConditionSize, Channels, false> inputMixin;
		DenseLayerT<Channels, Channels, true> oneByOne;
		Eigen::Matrix<float, Channels, WAVENET_MAX_NUM_FRAMES> state;

	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;
		static constexpr auto BufferSize = (ReceptiveFieldSize * 2) + WAVENET_MAX_NUM_FRAMES;
		static constexpr bool TooBigForStatic = ((Channels * BufferSize) * 4) > EIGEN_STACK_ALLOCATION_LIMIT;

		using LayerBufferType = typename std::conditional<TooBigForStatic,
			Eigen::Matrix<float, Channels, -1>,
			Eigen::Matrix<float, Channels, BufferSize>>::type;

		LayerBufferType layerBuffer;
		//Eigen::Matrix<float, Channels, -1> layerBuffer;
		size_t bufferStart;

		WaveNetLayerT()
		{
			state.setZero();
		}

		void AllocBuffer(int allocNum)
		{
			long size = BufferSize;

			if constexpr(TooBigForStatic)
			{
				layerBuffer.resize(Channels, size);
			}

			layerBuffer.setZero();

			//if (offset > (size - (ReceptiveFieldSize + WAVENET_MAX_NUM_FRAMES)))
			//{
			//	bufferStart = ReceptiveFieldSize;
			//}
			//else
			//{
			//	bufferStart = size - offset;
			//}

			bufferStart = ReceptiveFieldSize;

//#if (LAYER_ARRAY_BUFFER_PADDING == 0)
//				bufferStart = ReceptiveFieldSize;
//#else
//				bufferStart = size - (WAVENET_MAX_NUM_FRAMES * ((allocNum % LAYER_ARRAY_BUFFER_PADDING) + 1));	// Do the modulo to handle cases where LAYER_ARRAY_BUFFER_PADDING is not big enough to handle offset
//#endif
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			conv1D.SetWeights(weights);
			inputMixin.SetWeights(weights);
			oneByOne.SetWeights(weights);
		}

		void AdvanceFrames(const size_t numFrames)
		{
			if constexpr (ReceptiveFieldSize <= WAVENET_MAX_NUM_FRAMES)
			{
				layerBuffer.middleCols(0, ReceptiveFieldSize).noalias() = layerBuffer.middleCols(numFrames, ReceptiveFieldSize);
			}
			else
			{
				layerBuffer.middleCols(bufferStart - ReceptiveFieldSize, numFrames).noalias() = layerBuffer.middleCols(bufferStart, numFrames);

				bufferStart += numFrames;

				if (bufferStart > (BufferSize - WAVENET_MAX_NUM_FRAMES))
					bufferStart -= ReceptiveFieldSize;
			}
		}

		//void RewindBuffer()
		//{
		//	//numRewinds++;

		//	size_t start = ReceptiveFieldSize;

		//	layerBuffer.middleCols(start - ReceptiveFieldSize, ReceptiveFieldSize) = layerBuffer.middleCols(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize);

		//	bufferStart = start;
		//}

		void CopyBuffer()
		{
			for (size_t offset = 1; offset < ReceptiveFieldSize + 1; offset++)
			{
				layerBuffer.col(bufferStart - offset).noalias() = layerBuffer.col(bufferStart);
			}
		}

		template<typename Derived, typename Derived2, typename Derived3>
		void Process(const Eigen::MatrixBase<Derived>& condition, Eigen::MatrixBase<Derived2> const& headInput, Eigen::MatrixBase<Derived3> const& output, const size_t outputStart, const size_t numFrames)
		{
			auto block = state.leftCols(numFrames);

			conv1D.Process(layerBuffer, block, bufferStart, numFrames);

			inputMixin.ProcessAcc(condition, block);

			block = WAVENET_MATH::Tanh(block);

			//block = block.array().tanh();

			//float* data = block.data();
			//size_t size = block.rows() * block.cols();

			//for (size_t pos = 0; pos < size; pos++)
			//{
			//	data[pos] = WAVENET_MATH::Tanh(data[pos]);
			//}

			const_cast<Eigen::MatrixBase<Derived2>&>(headInput).noalias() += block.topRows(Channels);

			oneByOne.Process(block.topRows(Channels), const_cast<Eigen::MatrixBase<Derived3>&>(output).middleCols(outputStart, numFrames));

			const_cast<Eigen::MatrixBase<Derived3>&>(output).middleCols(outputStart, numFrames).noalias() += layerBuffer.middleCols(bufferStart, numFrames);
		}
	};

	template <int... values>
		using Dilations = std::integer_sequence<int, values...>;

	template <int InputSize, int ConditionSize, int HeadSize, int Channels, int KernelSize, typename DilationsSequence, bool HasHeadBias>
	class WaveNetLayerArrayT
	{
		template <typename>
		struct LayersHelper
		{
		};

		template <int... dilationVals>
		struct LayersHelper<Dilations<dilationVals...>>
		{
			using type = std::tuple<WaveNetLayerT<ConditionSize, Channels, KernelSize, dilationVals>...>;
		};

		using Layers = typename LayersHelper<DilationsSequence>::type;

	private:
		Layers layers;
		DenseLayerT<InputSize, Channels, false> rechannel;
		DenseLayerT<Channels, HeadSize, HasHeadBias> headRechannel;

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
		}

		int AllocBuffers(int allocNum)
		{
			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).AllocBuffer(allocNum++);
				});

			return allocNum;
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
			rechannel.Process(layerInputs, std::get<0>(layers).layerBuffer.middleCols(std::get<0>(layers).bufferStart, 1));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).CopyBuffer();

					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs, 0, 1);
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, std::get<layerIndex + 1>(layers).layerBuffer, std::get<layerIndex + 1>(layers).bufferStart, 1);
					}
				});

			headRechannel.Process(headInputs, headOutputs.leftCols(1));
		}

		template<typename Derived, typename Derived2, typename Derived3>
		void Process(const Eigen::MatrixBase<Derived>& layerInputs, const Eigen::MatrixBase<Derived2>& condition, Eigen::MatrixBase<Derived3> const& headInputs, const size_t numFrames)
		{
			rechannel.Process(layerInputs, std::get<0>(layers).layerBuffer.middleCols(std::get<0>(layers).bufferStart, numFrames));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs, 0, numFrames);
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs,	std::get<layerIndex + 1>(layers).layerBuffer, std::get<layerIndex + 1>(layers).bufferStart, numFrames);
					}

					std::get<layerIndex>(layers).AdvanceFrames(numFrames);
				});

			headRechannel.Process(headInputs, headOutputs.leftCols(numFrames));
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

		void SetWeights(std::vector<float> weights)
		{
			std::vector<float>::iterator it = weights.begin();

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					std::get<layerIndex>(layerArrays).SetWeights(it);
				});

			headScale = *(it++);

			assert(std::distance(weights.begin(), it) == (long)weights.size());
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
						std::get<layerIndex>(layerArrays).Prewarm(condition, condition, headArray.leftCols(1));
					}
					else
					{
						std::get<layerIndex>(layerArrays).Prewarm(std::get<layerIndex - 1>(layerArrays).arrayOutputs.leftCols(1), condition, std::get<layerIndex - 1>(layerArrays).headOutputs.leftCols(1));
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
						std::get<layerIndex>(layerArrays).Process(condition, condition, headArray.leftCols(numFrames), numFrames);
					}
					else
					{
						std::get<layerIndex>(layerArrays).Process(std::get<layerIndex - 1>(layerArrays).arrayOutputs.leftCols(numFrames), condition, std::get<layerIndex - 1>(layerArrays).headOutputs.leftCols(numFrames), numFrames);
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