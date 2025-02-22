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

			std::string arch = modelJson["architecture"];

			nlohmann::json config = modelJson["config"];

			std::vector<float> weights = modelJson["weights"];

			if (arch == "WaveNet")
			{
				std::vector<nam::wavenet::LayerArrayParams> layer_array_params;

				for (size_t i = 0; i < config["layers"].size(); i++)
				{
					nlohmann::json layer_config = config["layers"][i];

					layer_array_params.push_back(
						nam::wavenet::LayerArrayParams(layer_config["input_size"], layer_config["condition_size"], layer_config["head_size"],
							layer_config["channels"], layer_config["kernel_size"], layer_config["dilations"],
							layer_config["activation"], layer_config["gated"], layer_config["head_bias"]));
				}

				const bool with_head = !config["head"].is_null();
				const float head_scale = config["head_scale"];

				namModel = std::make_unique<nam::wavenet::WaveNet>(layer_array_params, head_scale, with_head, weights, sampleRate);
			}
			else if (arch == "LSTM")
			{
				const int num_layers = config["num_layers"];
				const int input_size = config["input_size"];
				const int hidden_size = config["hidden_size"];

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