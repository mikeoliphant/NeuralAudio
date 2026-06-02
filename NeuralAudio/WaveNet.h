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

#ifndef LAYER_ARRAY_BUFFER_PADDING
#define LAYER_ARRAY_BUFFER_PADDING 24
#endif

namespace NeuralAudio
{
	//int numRewinds = 0;
	//int maxRewinds = 0;

	template <int NumRows, int NumCols>
	class NAMatCols;

	template <int NumRows, int NumCols>
	class NAMat
	{
		public:
			NAMat()
			{
				if constexpr (TooBigForStatic)
				{
					Resize(NumRows, NumCols);
				}
			}

			void Resize(int numRows, int numCols)
			{
				matrix.resize(numRows, numCols);
			}

			int GetNumRows() const
			{
				return matrix.rows();
			}

			int GetNumCols() const
			{
				return matrix.cols();
			}

			NAMatCols<NumRows, NumCols> Slice(int startCol, int numCols)
			{
				NAMatCols<NumRows, NumCols> slice(this, startCol, numCols);

				return slice;
			}

			void SetZero()
			{
				matrix.setZero();
			}

			auto GetMatrix()
			{
				return &matrix;
			}

			auto GetCols(int startCol, int numCols)
			{
				return matrix.middleCols(startCol, numCols);
			}

			auto GetColsConst(int startCol, int numCols) const
			{
				return matrix.middleCols(startCol, numCols);
			}

		private:
			static constexpr bool TooBigForStatic = ((NumRows * NumCols) * 4) > EIGEN_STACK_ALLOCATION_LIMIT;

			using MatrixType = typename std::conditional<TooBigForStatic,
				Eigen::Matrix<float, NumRows, -1>,
				Eigen::Matrix<float, NumRows, NumCols>>::type;
			
			MatrixType matrix;
	};

	template <int NumRows, int NumCols>
	class NAMatCols
	{
		public:
			NAMatCols(NAMat<NumRows, NumCols>* ref, int startCol, int numCols)
			{
				matrixRef = ref;
				this->startCol = startCol;
				this->numCols = numCols;
			}

			NAMatCols<NumRows, NumCols> Slice(int startCol, int numCols)
			{
				NAMatCols<NumRows, NumCols> slice(matrixRef, this->startCol + startCol, numCols);

				return slice;
			}

		private:
			NAMat<NumRows, NumCols>* matrixRef;
			int startCol;
			int numCols;
	};

	template <int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DT
	{
	public:
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

		template<int InputCols, int OutputCols>
		inline void Process(const NAMat<InChannels, InputCols>& input, NAMat<OutChannels, OutputCols>& output, const size_t iStart, const size_t nCols) const
		{
			auto outBlock = output.GetCols(0, nCols);

			for (size_t k = 0; k < KernelSize; k++)
			{
				auto offset = Dilation * ((int)k + 1 - KernelSize);

				auto inBlock = input.GetColsConst(iStart + offset, nCols);

				if (k == 0)
					outBlock.noalias() = weights[k] * inBlock;
				else
					outBlock.noalias() += weights[k] * inBlock;
			}

			if constexpr (DoBias)
				outBlock.colwise() += bias;
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

	template <int ConditionSize, int Channels, int KernelSize, int Dilation>
	class WaveNetLayerT
	{
	private:
		Conv1DT<Channels, Channels, KernelSize, true, Dilation> conv1D;
		DenseLayerT<ConditionSize, Channels, false> inputMixin;
		DenseLayerT<Channels, Channels, true> oneByOne;
		NAMat<Channels, WAVENET_MAX_NUM_FRAMES> state;

	public:
		static constexpr auto ReceptiveFieldSize = (KernelSize - 1) * Dilation;
		static constexpr auto BufferSize = ReceptiveFieldSize + ((LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES);

		NAMat<Channels, BufferSize> layerBuffer;
		size_t bufferStart;

		WaveNetLayerT()
		{
			state.SetZero();
		}

		void AllocBuffer(int allocNum)
		{
			long size = BufferSize;

			layerBuffer.SetZero();

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

		size_t GetNumWeights()
		{
			return conv1D.GetNumWeights() + inputMixin.GetNumWeights() + oneByOne.GetNumWeights();
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			conv1D.SetWeights(weights);
			inputMixin.SetWeights(weights);
			oneByOne.SetWeights(weights);
		}

		void AdvanceFrames(const size_t numFrames)
		{
			bufferStart += numFrames;

			if ((bufferStart + WAVENET_MAX_NUM_FRAMES) > (size_t)layerBuffer.GetNumCols())
				RewindBuffer();
		}

		void RewindBuffer()
		{
			//numRewinds++;

			size_t start = ReceptiveFieldSize;

			layerBuffer.GetCols(start - ReceptiveFieldSize, ReceptiveFieldSize) = layerBuffer.GetCols(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize);

			bufferStart = start;
		}

		void CopyBuffer()
		{
			for (size_t offset = 1; offset < ReceptiveFieldSize + 1; offset++)
			{
				layerBuffer.GetCols(bufferStart - offset, 1) = layerBuffer.GetCols(bufferStart, 1);
			}
		}

		template<int InputCols, int OutputCols>
		void Process(const NAMat<1, WAVENET_MAX_NUM_FRAMES>& condition, NAMat<Channels, InputCols>& headInputs, NAMat<Channels, OutputCols>& output, const size_t outputStart, const size_t numFrames)
		{
			auto block = state.GetCols(0, numFrames);

			conv1D.Process(layerBuffer, state, bufferStart, numFrames);

			inputMixin.ProcessAcc(condition.GetColsConst(0, numFrames), block);

			WAVENET_MATH::Tanh(&block);

			headInputs.GetCols(0, numFrames).noalias() += block;

			oneByOne.Process(block, output.GetCols(outputStart, numFrames));

			output.GetCols(outputStart, numFrames).noalias() += layerBuffer.GetCols(bufferStart, numFrames);
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

		NAMat<Channels, WAVENET_MAX_NUM_FRAMES> arrayOutputs;
		NAMat<HeadSize, WAVENET_MAX_NUM_FRAMES> headOutputs;
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

		template<int LayerInputCols, int HeadInputCols>
		void Prewarm(const NAMat<InputSize, LayerInputCols>& layerInputs, const NAMat<1, WAVENET_MAX_NUM_FRAMES>& condition, NAMat<Channels, HeadInputCols>& headInputs)
		{
			rechannel.Process(layerInputs.GetColsConst(0, 1), std::get<0>(layers).layerBuffer.GetCols(std::get<0>(layers).bufferStart, 1));

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

			headRechannel.Process(headInputs.GetCols(0, 1), headOutputs.GetCols(0, 1));
		}

		template<int LayerInputCols, int HeadInputCols>
		void Process(const NAMat<InputSize, LayerInputCols>& layerInputs, const NAMat<1, WAVENET_MAX_NUM_FRAMES>& condition, NAMat<Channels, HeadInputCols>& headInputs, const size_t numFrames)
		{
			rechannel.Process(layerInputs.GetColsConst(0, numFrames), std::get<0>(layers).layerBuffer.GetCols(std::get<0>(layers).bufferStart, numFrames));

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

			headRechannel.Process(headInputs.GetCols(0, numFrames), headOutputs.GetCols(0, numFrames));
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
			condition.SetZero();

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

			//auto condition = Eigen::Map<const Eigen::Matrix<float, 1, -1>>(input, 1, numFrames);

			float* data = condition.GetCols(0, condition.GetNumCols()).data();

			memcpy(data, input, numFrames);

			headArray.SetZero();

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

			const float* finalHeadData = std::get<sizeof...(LayerArrays) - 1>(layerArrays).headOutputs.GetCols(0, numFrames).data();

			for (size_t i = 0; i < numFrames; i++)
			{
				output[i] = headScale * finalHeadData[i];
			}

			//if (numRewinds > maxRewinds)
			//{
			//	maxRewinds = numRewinds;

			//	std::cout << "New Max Rewinds: " << maxRewinds << std::endl;
			//}
		}

	private:
		static constexpr auto headLayerChannels = std::tuple_element_t<0, std::tuple<LayerArrays...>>::NumChannelsP;

		std::tuple<LayerArrays...> layerArrays;
		NAMat<1, WAVENET_MAX_NUM_FRAMES> condition;
		NAMat<headLayerChannels, WAVENET_MAX_NUM_FRAMES> headArray;
		float headScale;
	};
}
