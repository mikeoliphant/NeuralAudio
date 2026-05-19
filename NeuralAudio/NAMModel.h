#pragma once

#include "NeuralModel.h"
#include "NeuralModelImpl.h"
#include <NAM/activations.h>
#include <NAM/get_dsp.h>
#include <NAM/dsp.h>
#include <NAM/registry.h>
#include <NAM/slimmable.h>

namespace NeuralAudio
{
	class NAMModel : public NeuralModelImpl
	{
	public:
		NAMModel()
		{
			nam::activations::Activation::enable_fast_tanh();

			slimmableSize = defaultQualityScaleFactor;
		}

		~NAMModel()
		{
			if (namModel)
				namModel.reset();
		}

		EModelLoadMode GetLoadMode() override
		{
			return EModelLoadMode::NAMCore;
		}

		bool LoadFromJson(const nlohmann::json& modelJson)
		{
			if (namModel)
				namModel.reset();

			ReadNAMConfig(modelJson);

			namModel = nam::get_dsp(modelJson);

			auto* slim = dynamic_cast<nam::SlimmableModel*>(namModel.get());

			if (slim != nullptr)
			{
				isSlimmable = true;

				slim->SetSlimmableSize(slimmableSize);
			}

			SetMaxAudioBufferSize(defaultMaxAudioBufferSize);

			return true;
		}

		bool HasQualityScaling() override
		{
			return isSlimmable;
		}

		float GetQualityScaleFactor() override
		{
			return slimmableSize;
		}

		void SetQualityScaleFactor(float scaleFactor) override
		{
			if (HasQualityScaling())
			{
				if (slimmableSize != scaleFactor)
				{
					slimmableSize = scaleFactor;

					if (namModel != nullptr)
					{
						auto* slim = dynamic_cast<nam::SlimmableModel*>(namModel.get());

						slim->SetSlimmableSize(slimmableSize);
					}
				}
			}
		}

		void SetMaxAudioBufferSize(const int maxSize) override
		{
			namModel->Reset(namModel->GetExpectedSampleRate(), maxSize);
		}

		void Process(float* input, float* output, size_t numSamples) override
		{
			namModel->process(&input, &output, (int)numSamples);
		}

		void Prewarm() override
		{
			namModel->prewarm();
		}

	private:
		std::unique_ptr<nam::DSP> namModel = nullptr;
		float slimmableSize = 1.0f;
		bool isSlimmable = false;
	};
}