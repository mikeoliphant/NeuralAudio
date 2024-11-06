# NeuralAudio

NeuralAudio is a C++ library designed to make it easy to use neural network machine learning models in real-time audio applications.

# Supported Models

NeuralAudio currently supports the following model types:

- Neural Amp Modeler (NAM) WaveNet and LSTM models
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) keras models (LSTM, GRU)

# Underlying Libraries and Performance

NeuralAudio uses the [NAM core](https://github.com/sdatkinson/NeuralAmpModelerCore) codebase for handling WaveNet models.

[RTNeural](https://github.com/jatinchowdhury18/RTNeural) is used for NAM LSTM models and RTNeural keras models.

A subset of RTNeural LSTM models are processed using pre-compiled static architectures (increasing performance). Currently the following architectures are accelerated:

- LSTM 1x8
- LSTM 1x12
- LSTM 1x16
- LSTM 1x24
- LSTM 2x8
- LSTM 2x12
- LSTM 2x16

Other architectures will work fine, but will have somewhat reduced performance.

# API overview

To load a model:
```
NeuralModel* model = NeuralAudio::NeuralModel::CreateFromFile("<path to model file");
```

To process a model:

```
model->Process(pointerToFloatInputData, pointerToFloatOutputData, int numSamples);
```

Use **model->GetRecommendedInputDBAdjustment()** and **model->GetRecommendedOutputDBAdjustment()** to obtain the ideal input and output volume level adjustments in dB.
