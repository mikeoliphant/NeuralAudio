#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define NA_EXTERN extern __declspec(dllexport)
#else
#define NA_EXTERN extern
#endif

struct NeuralModel;


NA_EXTERN NeuralModel* CreateModelFromFile(const wchar_t* modelPath);

NA_EXTERN void DeleteModel(NeuralModel* model);

NA_EXTERN void SetLSTMLoadMode(int loadMode);

NA_EXTERN void SetWaveNetLoadMode(int loadMode);

NA_EXTERN void SetAudioInputLevelDBu(float audioDBu);

NA_EXTERN void SetDefaultMaxAudioBufferSize(int maxSize);

NA_EXTERN int GetLoadMode(NeuralModel* model);

NA_EXTERN bool IsStatic(NeuralModel* model);

NA_EXTERN void SetMaxAudioBufferSize(NeuralModel* model, int maxSize);

NA_EXTERN float GetRecommendedInputDBAdjustment(NeuralModel* model);

NA_EXTERN float GetRecommendedOutputDBAdjustment(NeuralModel* model);

NA_EXTERN float GetSampleRate(NeuralModel* model);

NA_EXTERN void Process(NeuralModel* model, float* input, float* output, size_t numSamples);

#ifdef __cplusplus
}
#endif