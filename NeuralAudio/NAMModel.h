#pragma once

#include "NeuralModel.h"
#include <NAM/activations.h>
#include <NAM/dsp.h>
#include <NAM/lstm.h>
#include <NAM/wavenet.h>

namespace NeuralAudio
{
	class NAMModel : public NeuralModel
	{
	public:
		NAMModel()
		{
			nam::activations::Activation::enable_fast_tanh();
		}

		~NAMModel()
		{
			if (namModel)
				namModel.reset();
		}

		EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::NAMCore;
		}

		bool LoadFromJson(const nlohmann::json& modelJson)
		{
			if (namModel)
				namModel.reset();

			ReadNAMConfig(modelJson);

			std::string arch = modelJson.at("architecture");

			nlohmann::json config = modelJson.at("config");

			std::vector<float> weights = modelJson.at("weights");

			if (arch == "WaveNet")
			{
				std::vector<nam::wavenet::LayerArrayParams> layer_array_params;

				for (size_t i = 0; i < config.at("layers").size(); i++)
				{
					nlohmann::json layerConfig = config.at("layers").at(i);

					layer_array_params.push_back(
						nam::wavenet::LayerArrayParams(layerConfig.at("input_size"), layerConfig.at("condition_size"), layerConfig.at("head_size"),
							layerConfig.at("channels"), layerConfig.at("kernel_size"), layerConfig.at("dilations"),
							layerConfig.at("activation"), layerConfig.at("gated"), layerConfig.at("head_bias")));
				}

				const bool with_head = !config.at("head").is_null();
				const float head_scale = config.at("head_scale");

				namModel = std::make_unique<nam::wavenet::WaveNet>(layer_array_params, head_scale, with_head, weights, sampleRate);
			}
			else if (arch == "LSTM")
			{
				const int num_layers = config.at("num_layers");
				const int input_size = config.at("input_size");
				const int hidden_size = config.at("hidden_size");

				namModel = std::make_unique<nam::lstm::LSTM>(num_layers, input_size, hidden_size, weights, sampleRate);
			}

			return true;
		}

		void Process(float* input, float* output, size_t numSamples)
		{
			namModel->process(input, output, (int)numSamples);
		}

		void Prewarm()
		{
			namModel->prewarm();
		}

	private:
		std::unique_ptr<nam::DSP> namModel = nullptr;
	};
}