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
#define LAYER_ARRAY_BUFFER_SIZE 4096

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
	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;

		Eigen::Matrix<float, Channels, -1> layerBuffer;
		size_t bufferStart;

		WaveNetLayerT()
		{
			state.setZero();
		}

		void AllocBuffer(int allocNum)
		{
			long size = ReceptiveFieldSize + LAYER_ARRAY_BUFFER_SIZE;

			layerBuffer.resize(Channels, size);
			layerBuffer.setZero();

			// offset prevents buffer rewinds of various layers from happening at the same time
			bufferStart = size - (WAVENET_MAX_NUM_FRAMES * allocNum);
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			conv1D.SetWeights(weights);
			inputMixin.SetWeights(weights);
			oneByOne.SetWeights(weights);
		}

		void SetMaxFrames(const size_t maxFrames)
		{
			(void)maxFrames;
		}

		void AdvanceFrames(const size_t numFrames)
		{
			bufferStart += numFrames;

			if ((bufferStart + WAVENET_MAX_NUM_FRAMES) > (size_t)layerBuffer.cols())
				RewindBuffer();
		}

		void RewindBuffer()
		{
			//numRewinds++;

			size_t start = ReceptiveFieldSize;

			layerBuffer.middleCols(start - ReceptiveFieldSize, ReceptiveFieldSize) = layerBuffer.middleCols(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize);

			bufferStart = start;
		}

		template<typename Derived, typename Derived2, typename Derived3>
		void Process(const Eigen::MatrixBase<Derived>& condition, Eigen::MatrixBase<Derived2> const& headInput, Eigen::MatrixBase<Derived3> const& output, const size_t iStart, const size_t jStart, const size_t numFrames)
		{
			auto block = state.leftCols(numFrames);

			conv1D.Process(layerBuffer, block, iStart, numFrames);

			inputMixin.ProcessAcc(condition, block);

			//block = block.array().tanh();

			float* data = block.data();
			size_t size = block.rows() * block.cols();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = FastTanh(data[pos]);
			}

			const_cast<Eigen::MatrixBase<Derived2>&>(headInput).noalias() += block.topRows(Channels);

			oneByOne.Process(block.topRows(Channels), const_cast<Eigen::MatrixBase<Derived3>&>(output).middleCols(jStart, numFrames));

			const_cast<Eigen::MatrixBase<Derived3>&>(output).middleCols(jStart, numFrames).noalias() += layerBuffer.middleCols(iStart, numFrames);

			AdvanceFrames(numFrames);
		}

	private:
		Conv1DT<Channels, Channels, KernelSize, true, Dilation> conv1D;
		DenseLayerT<ConditionSize, Channels, false> inputMixin;
		DenseLayerT<Channels, Channels, true> oneByOne;
		Eigen::Matrix<float, Channels, WAVENET_MAX_NUM_FRAMES> state;
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
		static constexpr auto ReceptiveFieldSize = std::tuple_element_t<numLayers - 1, Layers>::ReceptiveFieldSize;
		static constexpr auto NumChannelsP = Channels;
		static constexpr auto HeadSizeP = HeadSize;

		Eigen::Matrix<float, Channels, WAVENET_MAX_NUM_FRAMES> arrayOutputs;
		Eigen::Matrix<float, HeadSize, WAVENET_MAX_NUM_FRAMES> headOutputs;

		WaveNetLayerArrayT()
		{
		}

		int AllocBuffers(int allocNum)
		{
			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).AllocBuffer(allocNum++);
				});

			return allocNum;
		}

		void SetMaxFrames(const size_t maxFrames)
		{
			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).SetMaxFrames(maxFrames);
				});
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
		void Process(const Eigen::MatrixBase<Derived>& layerInputs, const Eigen::MatrixBase<Derived2>& condition, Eigen::MatrixBase<Derived3> const& headInputs, const size_t numFrames)
		{
			rechannel.Process(layerInputs, std::get<0>(layers).layerBuffer.middleCols(std::get<0>(layers).bufferStart, numFrames));

			ForEachIndex<numLayers>([&](auto layerIndex)
				{
					if constexpr (layerIndex == lastLayer)
					{
						std::get<layerIndex>(layers).Process(condition, headInputs, arrayOutputs, std::get<layerIndex>(layers).bufferStart, 0, numFrames);
					}
					else
					{
						std::get<layerIndex>(layers).Process(condition, headInputs,	std::get<layerIndex + 1>(layers).layerBuffer, std::get<layerIndex>(layers).bufferStart, std::get<layerIndex + 1>(layers).bufferStart, numFrames);
					}
				});

			headRechannel.Process(headInputs, headOutputs.leftCols(numFrames));
		}
	};

	template <typename... LayerArrays>
	class WaveNetModelT
	{
	public:
		WaveNetModelT()
		{
			int allocNum = 1;

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
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
			return maxFrames;
		}

		void SetMaxFrames(const size_t maxFrames)
		{
			this->maxFrames = maxFrames;

			if (this->maxFrames > WAVENET_MAX_NUM_FRAMES)
				this->maxFrames = WAVENET_MAX_NUM_FRAMES;

			ForEachIndex<sizeof...(LayerArrays)>([&](auto layerIndex)
				{
					std::get<layerIndex>(layerArrays).SetMaxFrames(this->maxFrames);
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
		size_t maxFrames;
	};
}