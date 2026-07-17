# NeuralAudio

NeuralAudio is a C++ library designed to make it easy to use neural network machine learning models (ie: guitar amplifier captures/profiles) in real-time audio applications.

# License

This repository is licensed under the [MIT license](https://github.com/mikeoliphant/NeuralAudio/blob/main/LICENSE). It is a liberal license, but please make sure that you comply with the terms - as well as the terms of [this project's dependencies](https://github.com/mikeoliphant/NeuralAudio/blob/main/CREDITS.md). I would also appreciate it if you would let me know if you are using this library.

# Supported Models and Underlying Libraries

NeuralAudio currently supports the following model types:

- [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) (NAM) WaveNet and LSTM models, A1 and A2 support
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) keras models (LSTM, GRU)

For WaveNet, the internal implmeentation supports optimized static implemenationas the offical NAM A1 and A2 network architectures:  A1 "Standard", "Lite", "Feather", "Nano" and A2 "Lite" and "Full".

For LSTM, the internal implementation supports optimized static models architectures for 1x8, 1x12, 1x16, 1x24, 2x8, 2x12, and 2x16 models.

All A1 NAM files with WaveNet and LSTM architectures not supported statically will fall back on a less performant dynamic implementation.

All non-standard A2 models currently use the NAM Core implementation (and consequently require building with NAM Core enabled).

All keras models not supported internally will fall back to the RTNeural implmentation.

# API overview

Models are loaded with a model loader:

```
NeuralModelLoader loader;

NeuralModel* model = loader.CreateFromFile("<path to model file>");
```

To process a model:

```
model->Process(pointerToFloatInputData, pointerToFloatOutputData, int numSamples);
```

## Setting maximum buffer size

Some models need to allocate memory based on the size of the audio buffers being used. You need to make sure that processing does not exceed the specified maximum buffer size.

The default maximum size is 128 samples. To change it, change the default size on the model loader:

```
loader.SetDefaultMaxAudioBufferSize(maxSize);
```

if you want to change the maximum buffer size of an already created model, do:

```
model->SetMaxAudioBufferSize(int maxSize);
```

***Note: this is not real-time safe, and should not be done on a real-time audio thread.***

## Input/Output calibration

Use ```model->GetRecommendedInputDBAdjustment()``` and ```model->GetRecommendedOutputDBAdjustment()``` to obtain the ideal input and output volume level adjustments in dB.

To set a known audio input level (ie: from an audio interface), use ```loader.SetAudioInputLevelDBu(float audioDBu)```. This is set at 12DBu by default.

## Model load behavior

By default, models are loaded using the internal NeuralAudio implementation (if possible). If you would like to force the use of the NAM Core or RTNeural implementations, you can use:

```
loader.SetWaveNetLoadMode(loadMode);
```

and

```
loader.SetLSTMLoadMode(loadMode);
```

where "loadMode" is one of:

```
NeuralAudio::EModelLoadMode::Internal
NeuralAudio::EModelLoadMode::NAMCore
NeuralAudio::EModelLoadMode::RTNeural  (only supported for LSTM)
```

You can check which implementation was actually used to load the model with ```model->GetLoadMode()```.

**NOTE:** Because of compile time and executable size considerations, only the internal, NAM Core and dynamic RTNeural implementations are built by default. If you want to use RTNeural for LSTM models, it is recommended that you add ```-DBUILD_STATIC_RTNEURAL=ON``` to your cmake commandline. This will create static model implmentations for the same set of LSTM models as the internal implmentation, and results in increased performance. Interal static LSTM model support is also off by default - to turn it on use ```-DBUILD_INTERNAL_STATIC_LSTM=ON```.

### Composite model load behavior

Some models (notably NAM A2 models) are comprised of multiple sub-models. By default, all sub-models will be fully loaded and initialized when the model is loaded.

If you wish to avoid the overhead of initializing unused models and only initialize the active model on load, you can do:

```
loader->SetCompositeModelLoadMode(ECompositeModelLoadMode::OnDemand);
```

Note that this means that switching to a different model for the first time via quality scaling will ***not be realtime safe***.

## Setting model quality scaling factor

Some models (notably, slimmable NAM A2 models) support quality scaling - trading off quality for performance.

Quality scaling is a floating point range from 0.0 (highest performance) to 1.0 (highest quality).

To set the default quality scaling factor, set it on the loader:

```
loader.SetDefaultQualityScaleFactor(scaleFactor);
```

To check if a model supports quality scaling, do:

```
if (model->HasQualityScaling()) ...
```

To set the quality scaling factor for a loaded model, do:

```
model->SetQualityScaleFactor(scaleFactor);
```

***Note: This operation is not real-time safe if the quality scale factor results in switching to an uninitialized model.*** If you are using the default composite model loading behavior, setting the quality scale factor is always real-time safe. If you are using "OnDemand" composite model loading, you can check whether a quality scale change is real-time safe by doing:

```
if (!model->IsQualityChangeRealtimeSafe(newScaleFactor)
{
  (call SetQualityScaleFactor(), but ensure it is not done in a real-time context)
}
```

To get the current quality scaling factor for a model, do:

```
float scaleFactor = model->GetQualityScaleFactor();
```

## Model oversampling

WaveNet models are altered (via scaling the convolution dilation sizes) to produce the correct output when the external sample rate is an even multiple of the model sample rate.

By default, the external sample rate is set to 48kHz. To change it, do:

```
loader.SetExternalSampleRate(sampleRate);
```

Note that ***models are only altered on load***. If you change the external sample rate you will need to reload any existing models where you want the change to take effect.

## Getting model metadata

To retrieve arbitrary metadata fields from models that contain them, do:

```
std::string fieldName = "this_is_a_field_name";

std::string metadataValue = model->GetMetadata(fieldName);

if (!metadataValue.empty())
{
  // do something
}
```
Results are always returned a strings. Field names are case sensitive.

To get the model version string, do:

```
std::string version = model->GetModelVersion()
```

The string will be empty if no version information exists.

## Getting the model receptive field size

WaveNet models have a fixed receptive field size (ie: size of the input that the output depends on).

To get this value, do:

```
int receptiveFieldSamples = model->GetReceptiveFieldSize();
```

Note that this can return -1, which means that the receptive field size is unknown, or not fixed (ie: LSTM models technically have an infinite tail because of their internal feedback loop).

This method is only supported for "internal" and NAM Core models. For RTNeural it will always return -1.

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

## Performance considerations

Optimization for this library is highly dependent on the details of the compiler you are using and the architecture of the system you intend to run on. In particular, getting the best performance relies on critical paths in the code being vectorized efficiently by the compiler for the target CPU.

The CMake setup for the library *does not* specify any optimization compiler flags - you are responsible for doing that yourself. Specifically, that means doing the highest possible compiler optimization (ie: "-O3")
and making sure that the compiler is targeting your CPU properly (ie: "-march=native" for GCC/Clang, "/arch:AVX2" for MSVC).

One more specific note - the ```MULTIFRAME_8X8_CONVOLUTION``` option described in the next section enables very impactful performance increases when set to "4" or "8" ("0" is the default) on appropriate hardware and compilers.
It also significantly *slows* performance if the optimizations are not properly supported. The best way to know which is the case in your scenario is to test it. The CMake config for this library does its best to default to the correct setting
based on architecture and compiler.

The "ModelTest" application binaries provided in the [Releases section](https://github.com/mikeoliphant/NeuralAudio/releases) have been optimized for various specific platforms and can be used as a basis for comparison.

## CMake Options

```-DBUILD_NAMCORE=ON|OFF```: Support loading models using the NAM Core implemenations.

```-DNAM_USE_INLINE_GEMM=ON|OFF```: Enable use of inline matrix multiplication in NAM Core.

```-DNAM_ENABLE_A2_FAST=ON|OFF```: Enable use of A2 fast path wavenet in NAM Core.

```-DBUILD_STATIC_RTNEURAL=ON|OFF```: Build static RTNeural model architectures (slower compile, larger size - only use if you plan on forcing RTNeural model loading).

```-DBUILD_INTERNAL_STATIC_WAVENET=ON|OFF```: Build internal static WaveNet model architectures (faster internal WaveNet, but slower compile, larger size).

```-DBUILD_INTERNAL_STATIC_LSTM=ON|OFF```: Build internal static LSTM model architectures (faster internal LSTM, but slower compile, larger size).

```-DBUILD_STATIC_INTERNAL_NAMA2=ON|OFF```: Build internal static A2 implementation.

```-DMULTIFRAME_8X8_CONVOLUTION="0"|"4"||"8"```: Use optimized multiframe 8x8 convolution. Much faster on very modern compilers. Much slower on older compilers. Be sure to use quotes around value. Defaults to "0" (disabled).

```-DDEFAULT_QUALITY_SCALE="X.X"```: Default model quality scale factor (0.0 to 1.0). Be sure to use quotes around value. Defaults to "1.0".

```-DDEFAULT_INPUT_DBU="XX"```: Default dBu level for model input calibration.

```-DWAVENET_FRAMES=XXX```: Sample buffer size for the internal WaveNet implementation. Defaults to **64**. If you know you are going to be using a fixed sample buffer smaller or larger than this, use that instead. Note that the model will still be able to process any buffer size - it is just optimized for this size.

```-DBUFFER_PADDING=XXX```: Amount of padding to convolution layer buffers. This allows ring buffer resets to be staggered accross layers to improve performance. It also uses a significant amount of memory. It is set to **24** by default. It can be set all the way down to 0 to reduce memory usage.

```-DWAVENET_MATH=XXX```
```-DLSTM_MATH=XXX```: Which math approximations (tanh and sigmoid) to use for WaveNet and LSTM models. Options are:

  - ```FastMath``` (the default): Use the same approximations as NAM Core.
  - ```EigenMath```: Use Eigen's builtin tanh approximation. Somewhat slower, but more accurate.
  - ```StdMath```: Use standard math functions. No approxmation used - much slower.

```-DBUILD_UTILS=ON|OFF```: Build performance/accuracy testing tools (located in the "Utils" folder).

# Software/Hardware Using NeuralAudio

The following applications and devices are using the NeuralAudio library for model processing:

- [neural-amp-modeler-lv2](https://github.com/mikeoliphant/neural-amp-modeler-lv2): LV2 plugin for using neural network machine learning amp models.
- [stompbox](https://github.com/mikeoliphant/stompbox): Guitar amplification and effects pedalboard simulation.
- [NeuralRack](https://github.com/brummer10/NeuralRack): Neural Model and Impulse Response File loader for Linux/Windows.
- [Darkglass Anagram](https://www.darkglass.com/creation/anagram): Bass guitar effects unit.
- [neural_tilde](https://github.com/apresta/neural_tilde): Max/MSP external for running neural amplifier captures.
- [Pi Pedal](https://github.com/rerdavies/pipedal): Guitar Effect Pedal for Raspberry Pi.
- [Casette](https://github.com/pawelKapl/Casette): Raspberry Pie guitar processor.
