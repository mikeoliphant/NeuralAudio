#include "NeuralAudioCApi.h"
#include "NeuralModel.h"

struct NeuralModel
{
    NeuralAudio::NeuralModel* model;
};

NeuralModel* CreateModelFromFile(const wchar_t* modelPath)
{
    NeuralModel* model = new NeuralModel();

    model->model = NeuralAudio::NeuralModel::CreateFromFile(modelPath);

    return model;
}

void DeleteModel(NeuralModel* model)
{
    delete model->model;
    delete model;
}

void SetLSTMLoadMode(int loadMode)
{
	NeuralAudio::NeuralModel::SetLSTMLoadMode((NeuralAudio::EModelLoadMode)loadMode);
}

void SetWaveNetLoadMode(int loadMode)
{
	NeuralAudio::NeuralModel::SetWaveNetLoadMode((NeuralAudio::EModelLoadMode)loadMode);
}

void SetAudioInputLevelDBu(float audioDBu)
{
	NeuralAudio::NeuralModel::SetAudioInputLevelDBu(audioDBu);
}

void SetDefaultMaxAudioBufferSize(int maxSize)
{
	NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(maxSize);
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


