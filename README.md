# NeuralAudio

NeuralAudio is a C++ library designed to make it easy to use neural network machine learning models (ie: guitar amplifier captures/profiles) in real-time audio applications.

# Supported Models

NeuralAudio currently supports the following model types:

- [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) (NAM) WaveNet and LSTM models
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) keras models (LSTM, GRU)

# Underlying Libraries and Performance

By default, NeuralAudio uses its own implementation of WaveNet and LSTM network models.

It can also load models using the [NAM Core implementation](https://github.com/sdatkinson/NeuralAmpModelerCore) and [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

The internal NeuralAudio implmentation currently outperforms the other implementations on all tested platforms (Windows x64, Linux x64/Arm64).

For WaveNet, the internal implmeentation supports optimized static models of the offical NAM network architectures:  "Standard", "Lite", "Feather", "Nano".

For LSTM, the internal implementation supports optimized static models architectures:

- LSTM 1x8
- LSTM 1x12
- LSTM 1x16
- LSTM 1x24
- LSTM 2x8
- LSTM 2x12
- LSTM 2x16

All NAM files with WaveNet and LSTM architectures not supported internally will fall back on a less performant dynamic implementation (although still faster than NAM Core).

All keras models not supported internally will fall back to the RTNeural implmentation.

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

The default maximum size is 128 samples. To change it, do:

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

By default, models are loaded using the internal NeuralAudio implementation. If you would like to force the use of the NAM Core or RTNeural implementations, you can use:

```
NeuralAudio::NeuralModel::SetWaveNetLoadMode(loadMode)
```

and

```
NeuralAudio::NeuralModel::SetLSTMLoadMode(loadMode)
```

where "loadMode" is one of:

```
NeuralAudio::EModelLoadMode::Internal
NeuralAudio::EModelLoadMode::NAMCore
NeuralAudio::EModelLoadMode::RTNeural
```

You can check which implementation was actually used to load the model with ```model->GetLoadMode()```.

**NOTE:** Because of compile time and executable size considerations, only the dynamic RTNeural implementation is built by default. If you want to use RTNeural, it is recommended that you add ```-DBUILD_STATIC_RTNEURAL=ON``` to your cmake commandline. This will create static model implmentations for the same sets of WaveNet and LSTM models as the internal implmentation, and results in increased performance.

# Building

First clone the repository:
```bash
git clone --recurse-submodules https://github.com/mikeoliphant/NeuralAudio
cd NeuralAudio/build
```

Then compile the plugin using:

**Linux/MacOS**
```bash
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j4
```

**Windows**
```bash
cmake.exe -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config=release -j4
```

Note - you'll have to change the Visual Studio version if you are using a different one.

## CMake Options

```-DBUILD_STATIC_RTNEURAL=ON```: Build static RTNeural model architectures (slower compile, larger size - only use if you plan on forcing RTNeural model loading)

```-DWAVENET_FRAMES=XXX```: Sample buffer size for the internal WaveNet implementation. Defaults to 64. If you know you are going to be using a fixed sample buffer smaller or larger than this, use that instead. Note that the model will still be able to process any buffer size - it is just optimized for this size.

```-DBUILD_UTILS=ON```: Build performance/accuracy testing tools (located in the "Utils" folder).

# Software Using NeuralAudio

The following applications are using the NeuralAudio library for model processing:

- [neural-amp-modeler-lv2](https://github.com/mikeoliphant/neural-amp-modeler-lv2): LV2 plugin for using neural network machine learning amp models.
- [stompbox](https://github.com/mikeoliphant/stompbox): Guitar amplification and effects pedalboard simulation.
