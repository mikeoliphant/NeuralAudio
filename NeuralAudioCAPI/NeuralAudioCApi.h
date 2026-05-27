#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define NA_EXTERN extern __declspec(dllexport)
#else
#define NA_EXTERN extern
#endif

struct NeuralModel;
struct NeuralModelLoader;

NA_EXTERN NeuralModelLoader* CreateLoader();

NA_EXTERN void DeleteLoader(NeuralModelLoader* loader);

NA_EXTERN NeuralModel* CreateModelFromFile(NeuralModelLoader *loader, const wchar_t* modelPath);

NA_EXTERN void DeleteModel(NeuralModel* model);

NA_EXTERN void SetLSTMLoadMode(NeuralModelLoader* loader, int loadMode);

NA_EXTERN void SetWaveNetLoadMode(NeuralModelLoader* loader, int loadMode);

NA_EXTERN void SetAudioInputLevelDBu(NeuralModelLoader* loader, float audioDBu);

NA_EXTERN void SetDefaultMaxAudioBufferSize(NeuralModelLoader* loader, int maxSize);

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