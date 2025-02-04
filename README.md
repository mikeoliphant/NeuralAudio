# NeuralAudio

NeuralAudio is a C++ library designed to make it easy to use neural network machine learning models (ie: guitar amplifier captures/profiles) in real-time audio applications.

# Supported Models

NeuralAudio currently supports the following model types:

- Neural Amp Modeler (NAM) WaveNet and LSTM models
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) keras models (LSTM, GRU)

# Underlying Libraries and Performance

[RTNeural](https://github.com/jatinchowdhury18/RTNeural) is used for RTNeural keras models.

The [NAM Core implementation](https://github.com/sdatkinson/NeuralAmpModelerCore) is used by default for NAM models, both LSTM and WaveNet.

You can also instruct the library to load NAM models using RTNeural (see the API overview below).

When using RTNeural, the official NAM WaveNet model architectures ("Standard", "Lite", "Feather", "Nano") are loaded using RTNeural using pre-compiled static architectures. Other NAM WaveNet model architectures will fall back on using the [NAM Core implementation](https://github.com/sdatkinson/NeuralAmpModelerCore).

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

## Setting maximum buffer size

Some models need to allocate memory based on the size of the audio buffers being used. You need to make sure that processing does not exceed the specified maximum buffer size.

The default maximum size is 512 samples. To change it, do:

```
NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(maxSize);
```

if you want to change the maximum buffer size of an already created model, do:

```
model->SetMaxAudioBufferSize(int maxSize);
```

***Note: this is not real-time safe, and should not be done on a real-time audio thread.***

## Input/Output calibration

Use ```model->GetRecommendedInputDBAdjustment()``` and ```model->GetRecommendedOutputDBAdjustment()``` to obtain the ideal input and output volume level adjustments in dB.

To set a known audio input level (ie: from an audio interface), use ```model->SetAudioInputLevelDBu(float audioDBu)```. This is set at 12DBu by default.

## Model load behavior

By default, NAM models are loaded using the NAM Core codebase. If you would like to force the use of RTNeural for NAM models, you can.

For LSTM:
```
NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::PreferRTNeural);
```

For WaveNet:
```
NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::PreferRTNeural)
```

# Software Using NeuralAudio

The following applications are using the NeuralAudio library for model processing:

- [neural-amp-modeler-lv2](https://github.com/mikeoliphant/neural-amp-modeler-lv2): LV2 plugin for using neural network machine learning amp models.
- [stompbox](https://github.com/mikeoliphant/stompbox): Guitar amplification and effects pedalboard simulation.
