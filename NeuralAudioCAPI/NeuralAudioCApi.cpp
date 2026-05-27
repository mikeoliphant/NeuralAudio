#include "NeuralAudioCApi.h"
#include "NeuralModel.h"

struct NeuralModel
{
    NeuralAudio::NeuralModel* model;
};

struct NeuralModelLoader
{
	NeuralAudio::NeuralModelLoader* loader;
};

NeuralModelLoader* CreateLoader()
{
	NeuralModelLoader* loader = new NeuralModelLoader();

	loader->loader = new NeuralAudio::NeuralModelLoader();

	return loader;
}

void DeleteLoader(NeuralModelLoader* loader)
{
	delete loader->loader;
	delete loader;
}

NeuralModel* CreateModelFromFile(NeuralModelLoader* loader, const wchar_t* modelPath)
{
    NeuralModel* model = new NeuralModel();

    model->model = loader->loader->CreateFromFile(modelPath);

    return model;
}

void DeleteModel(NeuralModel* model)
{
    delete model->model;
    delete model;
}

void SetLSTMLoadMode(NeuralModelLoader* loader, int loadMode)
{
	loader->loader->SetLSTMLoadMode((NeuralAudio::EModelLoadMode)loadMode);
}

void SetWaveNetLoadMode(NeuralModelLoader* loader, int loadMode)
{
	loader->loader->SetWaveNetLoadMode((NeuralAudio::EModelLoadMode)loadMode);
}

void SetAudioInputLevelDBu(NeuralModelLoader* loader, float audioDBu)
{
	loader->loader->SetAudioInputLevelDBu(audioDBu);
}

void SetDefaultMaxAudioBufferSize(NeuralModelLoader* loader, int maxSize)
{
	loader->loader->SetDefaultMaxAudioBufferSize(maxSize);
}

int GetLoadMode(NeuralModel* model)
{
	return model->model->GetLoadMode();
}

bool IsStatic(NeuralModel* model)
{
	return model->model->IsStatic();
}

void SetMaxAudioBufferSize(NeuralModel* model, int maxSize)
{
	model->model->SetMaxAudioBufferSize(maxSize);
}

float GetRecommendedInputDBAdjustment(NeuralModel* model)
{
	return model->model->GetRecommendedInputDBAdjustment();
}

float GetRecommendedOutputDBAdjustment(NeuralModel* model)
{
	return model->model->GetRecommendedOutputDBAdjustment();
}

float GetSampleRate(NeuralModel* model)
{
	return model->model->GetSampleRate();
}

void Process(NeuralModel* model, float* input, float* output, size_t numSamples)
{
    model->model->Process(input, output, numSamples);
}


