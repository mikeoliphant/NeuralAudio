#pragma once

// Based on WaveNet model structure from https://github.com/sdatkinson/NeuralAmpModelerCore
// with some template ideas from https://github.com/jatinchowdhury18/RTNeural-NAM

#include <Eigen/Dense>
#include <Eigen/Core>
#include "NeuralModel.h"
#include "Activation.h"

#ifndef WAVENET_MAX_NUM_FRAMES
#define WAVENET_MAX_NUM_FRAMES 64
#endif
#define LAYER_ARRAY_BUFFER_SIZE 4096

namespace NeuralAudio
{
	class Conv1D
	{
	private:
		size_t inChannels;
		size_t outChannels;
		size_t kernelSize;
		bool doBias;
		size_t dilation;
		std::vector<Eigen::MatrixXf> weights;
		Eigen::VectorXf bias;

	public:
		Conv1D(size_t inChannels, size_t outChannels, size_t kernelSize, bool doBias, size_t dilation) :
			inChannels(inChannels),
			outChannels(outChannels),
			kernelSize(kernelSize),
			doBias(doBias),
			dilation(dilation)
		{
			for (size_t k = 0; k < kernelSize; k++)
			{
				auto kernelWeights = Eigen::MatrixXf(outChannels, inChannels);
				weights.push_back(kernelWeights);
			}

			if (doBias)
			{
				bias.resize(outChannels);
			}
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			weights.resize(kernelSize);

			for (size_t i = 0; i < outChannels; i++)
				for (size_t j = 0; j < inChannels; j++)
					for (size_t k = 0; k < kernelSize; k++)
						weights[k](i, j) = *(inWeights++);

			if (doBias)
			{
				for (size_t i = 0; i < outChannels; i++)
					bias(i) = *(inWeights++);
			}
		}

		inline void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output, const size_t iStart, const size_t nCols) const
		{
			for (size_t k = 0; k < kernelSize; k++)
			{
				size_t offset = dilation * (k + 1 - kernelSize);

				auto& inBlock = input.middleCols(iStart + offset, nCols);

				if (k == 0)
					output.noalias() = weights[k] * inBlock;
				else
					output.noalias() += weights[k] * inBlock;
			}

			if (doBias)
				output.colwise() += bias;
		}
	};

	class DenseLayer
	{
	private:
		size_t inSize;
		size_t outSize;
		bool doBias;
		Eigen::MatrixXf weights;
		Eigen::VectorXf bias;

	public:
		DenseLayer(size_t inSize, size_t outSize, bool doBias) :
			inSize(inSize),
			outSize(outSize),
			doBias(doBias),
			weights(outSize, inSize)
		{
			if (doBias)
			{
				bias.resize(outSize);
			}
		}

		void SetWeights(std::vector<float>::iterator& inWeights)
		{
			for (auto i = 0; i < outSize; i++)
				for (auto j = 0; j < inSize; j++)
					weights(i, j) = *(inWeights++);

			if (doBias)
			{
				for (auto i = 0; i < outSize; i++)
					bias(i) = *(inWeights++);
			}
		}

		void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output) const
		{
			if (doBias)
			{
				output.noalias() = (weights * input).colwise() + bias;
			}
			else
			{
				output.noalias() = weights * input;
			}
		}

		void ProcessAcc(const Eigen::Ref<const Eigen::MatrixXf>& input, Eigen::Ref<Eigen::MatrixXf> output) const
		{
			if (doBias)
			{
				output.noalias() += (weights * input).colwise() + bias;
			}
			else
			{
				output.noalias() += weights * input;
			}
		}
	};

	class WaveNetLayer
	{
	private:
		size_t conditionSize;
		size_t channels;
		size_t kernelSize;
		size_t dilation;
		Conv1D conv1D;
		DenseLayer inputMixin;
		DenseLayer oneByOne;
		Eigen::MatrixXf state;
		Eigen::MatrixXf layerBuffer;

	public:
		size_t ReceptiveFieldSize;
		size_t bufferStart;

		WaveNetLayer(size_t conditionSize, size_t channels, size_t kernelSize, size_t dilation) :
			conditionSize(conditionSize),
			channels(channels),
			kernelSize(kernelSize),
			dilation(dilation),
			conv1D(channels, channels, kernelSize, true, dilation),
			inputMixin(conditionSize, channels, false),
			oneByOne(channels, channels, true),
			state(channels, WAVENET_MAX_NUM_FRAMES),
			ReceptiveFieldSize((kernelSize - 1) * dilation)
		{
			state.setZero();
		}

		Eigen::MatrixXf& GetLayerBuffer()
		{
			return layerBuffer;
		}

		void AllocBuffer(size_t allocNum)
		{
			size_t size = ReceptiveFieldSize + LAYER_ARRAY_BUFFER_SIZE;

			layerBuffer.resize(channels, size);
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

			if ((int)(bufferStart + WAVENET_MAX_NUM_FRAMES) > layerBuffer.cols())
				RewindBuffer();
		}

		void RewindBuffer()
		{
			size_t start = ReceptiveFieldSize;

			layerBuffer.middleCols(start - ReceptiveFieldSize, ReceptiveFieldSize) = layerBuffer.middleCols(bufferStart - ReceptiveFieldSize, ReceptiveFieldSize);

			bufferStart = start;
		}

		void Process(const Eigen::Ref<const Eigen::MatrixXf>& condition, Eigen::Ref<Eigen::MatrixXf> headInput, Eigen::Ref<Eigen::MatrixXf> output, const size_t iStart, const size_t jStart, const size_t numFrames)
		{
			auto block = state.leftCols(numFrames);

			conv1D.Process(layerBuffer, block, iStart, numFrames);

			inputMixin.ProcessAcc(condition, state);

			//block = block.array().tanh();

			float* data = block.data();
			auto size = block.rows() * block.cols();

			for (auto pos = 0; pos < size; pos++)
			{
				data[pos] = FastTanh(data[pos]);
			}

			headInput.noalias() += block.topRows(channels);

			oneByOne.Process(block.topRows(channels), output.middleCols(jStart, numFrames));

			output.middleCols(jStart, numFrames).noalias() += layerBuffer.middleCols(iStart, numFrames);

			AdvanceFrames(numFrames);
		}
	};

	class WaveNetLayerArray
	{
	private:
		size_t inputSize;
		size_t conditionSize;
		size_t headSize;
		size_t channels;
		size_t kernelSize;
		std::vector<WaveNetLayer> layers;
		DenseLayer rechannel;
		DenseLayer headRechannel;
		size_t lastLayer;
		Eigen::MatrixXf arrayOutputs;
		Eigen::MatrixXf headOutputs;


	public:
		WaveNetLayerArray(size_t inputSize, size_t conditionSize, size_t headSize, size_t channels, size_t kernelSize, bool hasHeadBias, std::vector<size_t> dilations) :
			inputSize(inputSize),
			conditionSize(conditionSize),
			headSize(headSize),
			channels(channels),
			kernelSize(kernelSize),
			rechannel(inputSize, channels, false),
			headRechannel(channels, headSize, hasHeadBias),
			arrayOutputs(channels, WAVENET_MAX_NUM_FRAMES),
			headOutputs(headSize, WAVENET_MAX_NUM_FRAMES)
		{
			for (auto dilation : dilations)
			{
				layers.push_back(WaveNetLayer(conditionSize, channels, kernelSize, dilation));
			}

			lastLayer = layers.size() - 1;
		}

		Eigen::MatrixXf& GetArrayOutputs()
		{
			return arrayOutputs;
		}

		Eigen::MatrixXf& GetHeadOutputs()
		{
			return headOutputs;
		}

		size_t GetNumChannels()
		{
			return channels;
		}

		size_t AllocBuffers(size_t allocNum)
		{
			for (auto& layer : layers)
			{
				layer.AllocBuffer(allocNum++);
			}

			return allocNum;
		}

		void SetMaxFrames(const long maxFrames)
		{
			for (auto& layer : layers)
			{
				layer.SetMaxFrames(maxFrames);
			}
		}

		void SetWeights(std::vector<float>::iterator& weights)
		{
			rechannel.SetWeights(weights);

			for (auto& layer : layers)
			{
				layer.SetWeights(weights);
			}

			headRechannel.SetWeights(weights);
		}

		void Process(const Eigen::MatrixXf& layerInputs, const Eigen::MatrixXf& condition, Eigen::Ref<Eigen::MatrixXf> headInputs, const size_t numFrames)
		{
			rechannel.Process(layerInputs,layers[0].GetLayerBuffer().middleCols(layers[0].bufferStart, numFrames));

			for (size_t layerIndex = 0; layerIndex < layers.size(); layerIndex++)
			{
				if (layerIndex == lastLayer)
				{
					layers[layerIndex].Process(condition, headInputs, arrayOutputs, layers[layerIndex].bufferStart, 0, numFrames);
				}
				else
				{
					layers[layerIndex].Process(condition, headInputs, layers[layerIndex + 1].GetLayerBuffer(), layers[layerIndex].bufferStart, layers[layerIndex + 1].bufferStart, numFrames);
				}
			}

			headRechannel.Process(headInputs, headOutputs.leftCols(numFrames));
		}
	};

	class WaveNetModel
	{
	private:
		std::vector<WaveNetLayerArray> layerArrays;
		size_t lastLayerArray;
		Eigen::MatrixXf headArray;
		float headScale;
		int maxFrames;

	public:
		WaveNetModel(std::vector<WaveNetLayerArray>& layerArrays) :
			layerArrays(layerArrays),									// ****** this is making a copy, which is gross
			lastLayerArray(layerArrays.size() - 1),
			headArray(layerArrays[0].GetNumChannels(), WAVENET_MAX_NUM_FRAMES)
		{
			size_t allocNum = 1;

			for (auto& layerArray : this->layerArrays)
			{
				allocNum = layerArray.AllocBuffers(allocNum);
			}
		}

		void SetWeights(std::vector<float> weights)
		{
			std::vector<float>::iterator it = weights.begin();

			for (auto& layerArray : layerArrays)
			{
				layerArray.SetWeights(it);
			}

			headScale = *(it++);

			assert(std::distance(weights.begin(), it) == (long)weights.size());
		}

		int GetMaxFrames()
		{
			return maxFrames;
		}

		void SetMaxFrames(const long maxFrames)
		{
			this->maxFrames = maxFrames;

			if (this->maxFrames > WAVENET_MAX_NUM_FRAMES)
				this->maxFrames = WAVENET_MAX_NUM_FRAMES;

			for (auto& layerArray : layerArrays)
			{
				layerArray.SetMaxFrames(this->maxFrames);
			}
		}

		void Process(const float* input, float* output, const size_t numFrames)
		{
			auto condition = Eigen::Map<const Eigen::MatrixXf>(input, 1, numFrames);

			headArray.setZero();

			for (size_t layerArrayIndex = 0; layerArrayIndex < layerArrays.size(); layerArrayIndex++)
			{
				if (layerArrayIndex == 0)
				{
					layerArrays[layerArrayIndex].Process(condition, condition, headArray.leftCols(numFrames), numFrames);
				}
				else
				{
					layerArrays[layerArrayIndex].Process(layerArrays[layerArrayIndex - 1].GetArrayOutputs().leftCols(numFrames), condition, layerArrays[layerArrayIndex - 1].GetHeadOutputs().leftCols(numFrames), numFrames);
				}
			}

			const auto& finalHeadArray = layerArrays[lastLayerArray].GetHeadOutputs();

			auto out = Eigen::Map<Eigen::Matrix<float, 1, -1>>(output, 1, numFrames);

			out.noalias() = headScale * finalHeadArray.leftCols(numFrames);
		}
	};
}