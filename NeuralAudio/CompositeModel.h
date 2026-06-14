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

			void AddModel(NeuralModelImpl* model)
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

			bool IsModelChangeRealtimeSafe(int newIndex)
			{
				if (newIndex == currentModelIndex.load())
					return true;

				return models[newIndex]->HadInitialPrewarm();
			}

			void SetCurrentModelIndex(int index)
			{
				if (index != currentModelIndex.load())
				{
					currentModelIndex.store(index);

					if (compositeLoadMode == ECompositeModelLoadMode::OnDemand)
					{
						// For on-demand loading we need to check if the model we are switching to has been prewarmed
						if (!models[currentModelIndex.load()]->HadInitialPrewarm())
						{
							Prewarm();
						}
					}
				}
			}

			int GetReceptiveFieldSize() override
			{
				if (currentModelIndex == -1)
					return -1;

				return models[currentModelIndex.load()]->GetReceptiveFieldSize();
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
				if (compositeLoadMode == ECompositeModelLoadMode::OnDemand)
				{
					auto& model = models[currentModelIndex.load()];
					model->Prewarm();
					model->SetHadInitialPrewarm();
				}
				else
				{
					for (auto& model : models)
					{
						model->Prewarm();
						model->SetHadInitialPrewarm();
					}
				}
			}

		protected:
			std::vector<NeuralModelImpl*> models;
			std::atomic<int> currentModelIndex = -1;
			ECompositeModelLoadMode compositeLoadMode = ECompositeModelLoadMode::LoadAll;
	};


	class ScalableCompositeModel : public CompositeModel
	{
		public:			
			EModelLoadMode GetLoadMode() override
			{
				return EModelLoadMode::NAMCore;
			}

			bool LoadFromJson(nlohmann::json& modelJson)
			{
				ReadNAMConfig(modelJson);

				return CreateModelFromNAMJson(modelJson);
			}

			virtual bool CreateModelFromNAMJson(nlohmann::json& modelJson)
			{
				compositeLoadMode = loader->GetCompositeModelLoadMode();

				auto& config = modelJson.at("config");

				auto& subModels = config.at("submodels");

				for (auto& submodelJson : subModels)
				{
					NeuralModelImpl* submodel = dynamic_cast<NeuralModelImpl*>(loader->CreateFromJson(submodelJson.at("model"), ".nam", false));

					AddModel(submodelJson.at("max_value"), submodel);
				}

				// Don't call SetMaxAudioBufferSize because it has already been done by individual models
				// (this can be re-added for completeness if NAM Core GetMaxBufferSize() is made public
				//SetMaxAudioBufferSize(loader->GetDefaultMaxAudioBufferSize());

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

			bool IsQualityChangeRealtimeSafe(float newScaleFactor) override
			{
				return IsModelChangeRealtimeSafe(GetModelIndexFromQualityScale(newScaleFactor));
			}

			void SetQualityScaleFactor(float scaleFactor) override
			{
				currentQualityLevel.store(scaleFactor);

				SetCurrentModelIndex(GetModelIndexFromQualityScale(scaleFactor));
			}

			void AddModel(float scaleFactor, NeuralModelImpl* model)
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

			int GetModelIndexFromQualityScale(float qualityScale)
			{
				int modelIndex = 0;

				for (auto& level : qualityLevels)
				{
					modelIndex = std::get<1>(level);

					if (qualityScale <= std::get<0>(level))
						break;
				}

				return modelIndex;
			}
	};
}