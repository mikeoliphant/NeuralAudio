#pragma once

#include <atomic>
#include <vector>
#include "NeuralModel.h"
#include "NeuralModelImpl.h"

namespace NeuralAudio
{
	class CompositeModel : public NeuralModelImpl
	{
		public:
			CompositeModel()
			{
			}

			~CompositeModel()
			{
				for (auto& model : models)
				{
					delete model;
				}
			}

			void AddModel(NeuralModel* model)
			{
				if (currentModelIndex.load() == -1)
					currentModelIndex.store(0);

				models.push_back(model);
			}

			int GetModelCount()
			{
				return (int)models.size();
			}

			int GetCurrentModelIndex()
			{
				return currentModelIndex.load();
			}

			void SetCurrentModelIndex(int index)
			{
				currentModelIndex.store(index);
			}

			void SetMaxAudioBufferSize(const int maxSize) override
			{
				for (auto& model : models)
				{
					model->SetMaxAudioBufferSize(maxSize);
				}
			}

			void Process(float* input, float* output, size_t numSamples) override
			{
				if (currentModelIndex == -1)
					return;

				models[currentModelIndex.load()]->Process(input, output, numSamples);
			}

			void Prewarm() override
			{
				for (auto& model : models)
				{
					model->Prewarm();
				}
			}

		protected:
			std::vector<NeuralModel*> models;
			std::atomic<int> currentModelIndex = -1;
	};


	class ScalableCompositeModel : public CompositeModel
	{
		public:			
			EModelLoadMode GetLoadMode() override
			{
				return EModelLoadMode::NAMCore;
			}

			bool LoadFromJson(const nlohmann::json& modelJson)
			{
				ReadNAMConfig(modelJson);

				return CreateModelFromNAMJson(modelJson);
			}

			virtual bool CreateModelFromNAMJson(const nlohmann::json& modelJson)
			{
				nlohmann::json config = modelJson.at("config");

				nlohmann::json subModels = config.at("submodels");

				for (auto& submodelJson : subModels)
				{
					NeuralModel* submodel = loader->CreateFromJson(submodelJson.at("model"), ".nam", false);

					AddModel(submodelJson.at("max_value"), submodel);
				}

				SetMaxAudioBufferSize(loader->GetDefaultMaxAudioBufferSize());
				SetQualityScaleFactor(loader->GetDefaultQualityScaleFactor());

				return true;
			}

			bool HasQualityScaling() override
			{
				return true;
			}

			float GetQualityScaleFactor() override
			{
				return currentQualityLevel.load();
			}

			void SetQualityScaleFactor(float scaleFactor) override
			{
				currentQualityLevel.store(scaleFactor);

				int modelIndex = 0;

				for (auto& level : qualityLevels)
				{
					if (std::get<0>(level) > scaleFactor)
						break;

					modelIndex = std::get<1>(level);
				}

				SetCurrentModelIndex(modelIndex);
			}

			void AddModel(float scaleFactor, NeuralModel* model)
			{
				CompositeModel::AddModel(model);

				qualityLevels.emplace_back(scaleFactor, GetModelCount() - 1);

				// Sort by quality level
				std::sort(qualityLevels.begin(), qualityLevels.end(), [](const auto& a, const auto& b)
				{
					return std::get<0>(a) < std::get<0>(b);
				});
			}

		protected:
			std::atomic<float> currentQualityLevel = 1.0f;
			std::vector<std::tuple<float, int>> qualityLevels;
	};
}