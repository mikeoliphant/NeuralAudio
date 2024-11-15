# NeuralAudio

NeuralAudio is a C++ library designed to make it easy to use neural network machine learning models (ie: guitar amplifier captures/profiles) in real-time audio applications.

# Supported Models

NeuralAudio currently supports the following model types:

- Neural Amp Modeler (NAM) WaveNet and LSTM models
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) keras models (LSTM, GRU)

# Underlying Libraries and Performance

[RTNeural](https://github.com/jatinchowdhury18/RTNeural) is used for RTNeural keras models and most NAM models.

The official NAM WaveNet model architectures ("Standard", "Lite", "Feather", "Nano") are loaded using RTNeural by default, and use pre-compiled static architectures. Other NAM WaveNet model architectures will fall back on using the [NAM Core implementation](https://github.com/sdatkinson/NeuralAmpModelerCore).

A subset of LSTM models are processed using pre-compiled static architectures (increasing performance). Currently the following architectures are accelerated:

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
NeuralModel* model = NeuralAudio::NeuralModel::CreateFromFile("<path to model file>");
```

To process a model:

```
model->Process(pointerToFloatInputData, pointerToFloatOutputData, int numSamples);
```

Use **model->GetRecommendedInputDBAdjustment()** and **model->GetRecommendedOutputDBAdjustment()** to obtain the ideal input and output volume level adjustments in dB.

If you would like to force the use of the NAM Core implementation for NAM models, you can.

For LSTM:
```
NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::PreferNAMCore);
```

For WaveNet:
```
NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::PreferNAMCore)
```

# Software Using NeuralAudio

The following applications are using the NeuralAudio library for model processing:

- [neural-amp-modeler-lv2](https://github.com/mikeoliphant/neural-amp-modeler-lv2): LV2 plugin for using neural network machine learning amp models.
- [stompbox](https://github.com/mikeoliphant/stompbox): Guitar amplification and effects pedalboard simulation.
