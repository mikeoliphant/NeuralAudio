#pragma once

#include "NeuralModel.h"
#include <RTNeural/RTNeural.h>
#include "wavenet_model.hpp"

namespace NeuralAudio
{
	inline static float fast_tanh(const float x)
	{
		const float ax = fabsf(x);
		const float x2 = x * x;

		return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
			/ (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
	}

	struct FastMathsProvider
	{
		template <typename Matrix>
		static auto tanh(Matrix& x)
		{
			using T = typename Matrix::Scalar;

			long size = x.rows() * x.cols();
			float* data = x.data();

			for (long i = 0; i < size; i++)
			{
				data[i] = 0.5f * (fast_tanh(data[i] * 0.5f) + 1.0f);
			}

			return x;
		}

		template <typename Matrix>
		static auto sigmoid(Matrix& x)
		{
			using T = typename Matrix::Scalar;

			long size = x.rows() * x.cols();
			float* data = x.data();

			for (long i = 0; i < size; i++)
			{
				data[i] = 0.5f * (fast_tanh(data[i] * 0.5f) + 1.0f);
			}

			return x;

			//return ((x.array() / (T)2).array().tanh() + (T)1) / (T)2;
		}

		template <typename Matrix>
		static auto exp(const Matrix& x)
		{
			return x.array().exp();
		}
	};

	template <typename T, int size, typename MathsProvider = FastMathsProvider>
	class FastTanhActivationT
	{
		using v_type = Eigen::Matrix<T, size, 1>;

	public:
		static constexpr auto in_size = size;
		static constexpr auto out_size = size;

		FastTanhActivationT()
			: outs(outs_internal)
		{
		}

		/** Returns the name of this layer. */
		std::string getName() const noexcept { return "tanh"; }

		/** Returns true if this layer is an activation layer. */
		constexpr bool isActivation() const noexcept { return true; }

		RTNEURAL_REALTIME void reset() {}

		/** Performs forward propagation for tanh activation. */
		RTNEURAL_REALTIME inline void forward(v_type& ins) noexcept
		{
			long size = ins.rows() * ins.cols();
			T* data = ins.data();

			for (long i = 0; i < size; i++)
			{
				data[i] = fast_tanh(data[i]);
			}

			outs = ins;
		}

		v_type& outs;

	private:
		v_type outs_internal;
	};

	class RTNeuralModel : public NeuralModel
	{
	public:
		virtual float GetRecommendedInputDBAdjustment()
		{
			return inputGain;
		}

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return outputGain;
		}
		
		float GetSampleRate()
		{
			return sampleRate;
		}

		bool LoadFromKerasJson(nlohmann::json& modelJson)
		{
			if (modelJson.contains("samplerate"))
			{
				sampleRate = modelJson["samplerate"];
			}

			if (modelJson.contains("in_gain"))
			{
				inputGain = modelJson["in_gain"];
			}

			if (modelJson.contains("out_gain"))
			{
				outputGain = modelJson["out_gain"];
			}

			return CreateModelFromKerasJson(modelJson);

			return true;
		}

		virtual bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			return false;
		}

		virtual bool LoadFromNAMJson(nlohmann::json& modelJson)
		{
			if (modelJson.contains("samplerate"))
			{
				sampleRate = modelJson["samplerate"];
			}

			if (modelJson.contains("sample_rate"))
			{
				sampleRate = modelJson["sample_rate"];
			}

			if (modelJson.contains("metadata"))
			{
				nlohmann::json metaData = modelJson["metadata"];

				if (metaData.contains("loudness"))
				{
					outputGain = -18 - (float)metaData["loudness"];
				}
			}

			return CreateModelFromNAMJson(modelJson);
		}

		virtual bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			return false;
		}

	protected:
		float sampleRate = 48000;
		float inputGain = 0;
		float outputGain = 0;
	};

	template <std::size_t ... Is, typename F>
	void ForEachIndex(std::index_sequence<Is...>, F&& f)
	{
		int dummy[] = { 0, /* Handles empty Is. following cast handle evil operator comma */
					   (static_cast<void>(f(std::integral_constant<std::size_t, Is>())), 0)... };
		static_cast<void>(dummy); // avoid warning for unused variable
	}

	template <std::size_t N, typename F>
	void ForEachIndex(F&& f)
	{
		ForEachIndex(std::make_index_sequence<N>(), std::forward<F>(f));
	}

	template <int numLayers, int hiddenSize>
	class RTNeuralLSTMModelT : public RTNeuralModel
	{
		using ModelType = typename std::conditional<numLayers == 1,
			RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, hiddenSize, RTNeural::SampleRateCorrectionMode::None, FastMathsProvider>, RTNeural::DenseT<float, hiddenSize, 1>>,
			RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, hiddenSize, RTNeural::SampleRateCorrectionMode::None, FastMathsProvider>,
				RTNeural::LSTMLayerT<float, hiddenSize, hiddenSize, RTNeural::SampleRateCorrectionMode::None, FastMathsProvider>, RTNeural::DenseT<float, hiddenSize, 1>>
		>::type;

	public:
		RTNeuralLSTMModelT()
			: model(nullptr)
		{
		}

		~RTNeuralLSTMModelT()
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}
		}

		bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			model->parseJson(modelJson, true);
			model->reset();

			return true;
		}

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			nlohmann::json config = modelJson["config"];

			std::vector<float> weights = modelJson["weights"];

			const int networkInputSize = 1;
			const int networkOutputSize = 1;
			const int gateSize = 4 * hiddenSize;

			auto iter = weights.begin();

			ForEachIndex<numLayers>([&](auto layer)
				{
					const int layerInputSize = (layer == 0) ? networkInputSize : hiddenSize;

					Eigen::MatrixXf inputPlusHidden = Eigen::Map<Eigen::MatrixXf>(&(*iter), layerInputSize + hiddenSize, gateSize);

					auto& lstmLayer = model->template get<layer>();

					// Input weights
					std::vector<std::vector<float>> inputWeights;

					inputWeights.resize(layerInputSize);

					for (size_t col = 0; col < layerInputSize; col++)
					{
						inputWeights[col].resize(gateSize);

						for (size_t row = 0; row < gateSize; row++)
						{
							inputWeights[col][row] = inputPlusHidden(col, row);
						}
					}

					lstmLayer.setWVals(inputWeights);

					// Recurrent weights
					std::vector<std::vector<float>> hiddenWeights;

					hiddenWeights.resize(hiddenSize);

					for (size_t col = 0; col < hiddenSize; col++)
					{
						hiddenWeights[col].resize(gateSize);

						for (size_t row = 0; row < gateSize; row++)
						{
							hiddenWeights[col][row] = inputPlusHidden(col + layerInputSize, row);
						}
					}

					lstmLayer.setUVals(hiddenWeights);

					iter += (gateSize * (layerInputSize + hiddenSize));

					// Bias weights
					std::vector<float> biasWeights = std::vector<float>(iter, iter + gateSize);

					lstmLayer.setBVals(biasWeights);

					iter += gateSize;

					// initial internal state values follow here in NAM, but aren't supported by RTNeural

					iter += hiddenSize * 2;	// (hidden state and cell state)
				});

			// Dense layer weights
			auto& denseLayer = model->template get<numLayers>();

			std::vector<std::vector<float>> denseWeights;
			denseWeights.resize(1);
			denseWeights[0] = std::vector<float>(iter, iter + hiddenSize);

			denseLayer.setWeights(denseWeights);

			iter += hiddenSize;

			// Dense layer bias
			auto denseBias = std::vector<float>(iter, iter + networkOutputSize);
			denseLayer.setBias(&(*iter));

			iter += networkOutputSize;

			model->reset();

			return true;
		}

		void Process(float* input, float* output, int numSamples)
		{
			for (int i = 0; i < numSamples; i++)
				output[i] = model->forward(input + i);
		}

		void Prewarm()
		{
			float sample = 0;

			for (int i = 0; i < 2048; i++)
				model->forward(&sample);
		}

	private:
		ModelType* model = nullptr;
	};

	using StdDilations = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
	using LiteDilations1 = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64>;
	using LiteDilations2 = wavenet::Dilations<128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;

	template <int numChannels, int headSize>
	class RTNeuralWaveNetModelT : public RTNeuralModel
	{
		using ModelType = typename std::conditional<numChannels == 16,
			wavenet::Wavenet_Model<float, 1,
				wavenet::Layer_Array<float, 1, 1, headSize, numChannels, 3, StdDilations, false, FastMathsProvider, FastTanhActivationT<float, numChannels, FastMathsProvider>>,
				wavenet::Layer_Array<float, numChannels, 1, 1, headSize, 3, StdDilations, true, FastMathsProvider, FastTanhActivationT<float, headSize, FastMathsProvider>>>,
			wavenet::Wavenet_Model<float, 1,
				wavenet::Layer_Array<float, 1, 1, headSize, numChannels, 3, LiteDilations1, false, FastMathsProvider, FastTanhActivationT<float, numChannels, FastMathsProvider>>,
				wavenet::Layer_Array<float, numChannels, 1, 1, headSize, 3, LiteDilations2, true, FastMathsProvider, FastTanhActivationT<float, headSize, FastMathsProvider>>>
			>::type;

	public:
		RTNeuralWaveNetModelT()
			: model(nullptr)
		{
		}

		~RTNeuralWaveNetModelT()
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}
		}

		//bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		//{
		//	if (model != nullptr)
		//	{
		//		delete model;
		//		model = nullptr;
		//	}

		//	model = new ModelType;

		//	model->parseJson(modelJson, true);
		//	model->reset();

		//	return true;
		//}

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			if (model != nullptr)
			{
				delete model;
				model = nullptr;
			}

			model = new ModelType;

			nlohmann::json config = modelJson["config"];

			model->load_weights(modelJson);

			return true;
		}

		void Process(float* input, float* output, int numSamples)
		{
			for (int i = 0; i < numSamples; i++)
				output[i] = model->forward(input[i]);
		}

		void Prewarm()
		{
			float sample = 0;

			for (int i = 0; i < 2048; i++)
				model->forward(sample);
		}

	private:
		ModelType* model = nullptr;
	};


	class RTNeuralModelDyn : public RTNeuralModel
	{
	public:
		RTNeuralModelDyn()
		{
		}

		~RTNeuralModelDyn()
		{
			if (model)
				model.reset();
		}

		bool CreateModelFromKerasJson(nlohmann::json& modelJson)
		{
			model = RTNeural::json_parser::parseJson<float, FastMathsProvider>(modelJson, true);
			model->reset();

			return true;
		}

		bool CreateModelFromNAMJson(nlohmann::json& modelJson)
		{
			model = std::make_unique<RTNeural::Model<float>>(1);

			nlohmann::json config = modelJson["config"];

			const int numLayers = config["num_layers"];
			const int inputSize = config["input_size"];
			const int hiddenSize = config["hidden_size"];

			std::vector<float> weights = modelJson["weights"];

			const int networkInputSize = inputSize;
			const int networkOutputSize = inputSize;
			const int gateSize = 4 * hiddenSize;

			auto iter = weights.begin();

			for (int layer = 0; layer < numLayers; layer++)
			{
				const int layerInputSize = (layer == 0) ? networkInputSize : hiddenSize;

				Eigen::MatrixXf inputPlusHidden = Eigen::Map<Eigen::MatrixXf>(&(*iter), layerInputSize + hiddenSize, gateSize);

				auto lstmLayer = new RTNeural::LSTMLayer<float>(layerInputSize, hiddenSize);

				model->addLayer(lstmLayer);

				// Input weights
				std::vector<std::vector<float>> inputWeights;

				inputWeights.resize(layerInputSize);

				for (size_t col = 0; col < layerInputSize; col++)
				{
					inputWeights[col].resize(gateSize);

					for (size_t row = 0; row < gateSize; row++)
					{
						inputWeights[col][row] = inputPlusHidden(col, row);
					}
				}

				lstmLayer->setWVals(inputWeights);

				// Recurrent weights
				std::vector<std::vector<float>> hiddenWeights;

				hiddenWeights.resize(hiddenSize);

				for (size_t col = 0; col < hiddenSize; col++)
				{
					hiddenWeights[col].resize(gateSize);

					for (size_t row = 0; row < gateSize; row++)
					{
						hiddenWeights[col][row] = inputPlusHidden(col + layerInputSize, row);
					}
				}

				lstmLayer->setUVals(hiddenWeights);

				iter += (gateSize * (layerInputSize + hiddenSize));

				// Bias weights
				std::vector<float> biasWeights = std::vector<float>(iter, iter + gateSize);

				lstmLayer->setBVals(biasWeights);

				iter += gateSize;

				// initial internal state values follow here in NAM, but aren't supported by RTNeural

				iter += hiddenSize * 2;	// (hidden state and cell state)

				//// LSTM hidden state
				//auto hiddenState = std::vector<float>(iter, iter + hiddenSize);

				//iter += hiddenSize;

				//// LSTM cell state
				//auto cellState = std::vector<float>(iter, iter + hiddenSize);

				//lstmLayer->setHCVals(hiddenState, cellState);

				//iter += hiddenSize;
			}

			// Dense layer weights
			auto denseLayer = new RTNeural::Dense<float>(hiddenSize, networkOutputSize);
			model->addLayer(denseLayer);

			std::vector<std::vector<float>> denseWeights;
			denseWeights.resize(1);
			denseWeights[0] = std::vector<float>(iter, iter + hiddenSize);

			denseLayer->setWeights(denseWeights);

			iter += hiddenSize;

			// Dense layer bias
			auto denseBias = std::vector<float>(iter, iter + networkOutputSize);
			denseLayer->setBias(&(*iter));

			iter += networkOutputSize;

			model->reset();

			return true;
		}

		void Process(float* input, float* output, int numSamples)
		{
			for (int i = 0; i < numSamples; i++)
				output[i] = model->forward(input + i);
		}

		void Prewarm()
		{
			float sample = 0;

			for (int i = 0; i < 2048; i++)
				model->forward(&sample);
		}

	private:
		std::unique_ptr<RTNeural::Model<float>> model;
	};

	class RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
		{
			return nullptr;
		}
	};


	class RTNeuralLSTMDefinitionBase : public RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
		{
			return nullptr;
		}

		virtual int GetNumLayers()
		{
			return 0;
		}

		virtual int GetHiddenSize()
		{
			return 0;
		}
	};

	template <int numLayers, int hiddenSize>
	class RTNeuralLSTMDefinitionT : public RTNeuralLSTMDefinitionBase
	{
	public:
		RTNeuralModel* CreateModel()
		{
			return new RTNeuralLSTMModelT<numLayers, hiddenSize>;
		}

		int GetNumLayers()
		{
			return numLayers;
		}

		int GetHiddenSize()
		{
			return hiddenSize;
		}
	};

	class RTNeuralWaveNetDefinitionBase : public RTNeuralModelDefinitionBase
	{
	public:
		virtual RTNeuralModel* CreateModel()
		{
			return nullptr;
		}

		virtual int GetNumChannels()
		{
			return 0;
		}

		virtual int GetHeadSize()
		{
			return 0;
		}
	};

	template <int numChannels, int headSize>
	class RTNeuralWaveNetDefinitionT : public RTNeuralWaveNetDefinitionBase
	{
	public:
		RTNeuralModel* CreateModel()
		{
			return new RTNeuralWaveNetModelT<numChannels, headSize>;
		}

		virtual int GetNumChannels()
		{
			return numChannels;
		}

		virtual int GetHeadSize()
		{
			return headSize;
		}
	};
}
