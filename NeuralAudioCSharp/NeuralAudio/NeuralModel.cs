using System;
using System.Runtime.CompilerServices;

namespace NeuralAudio
{
    public enum EModelLoadMode
    {
        Internal,
        RTNeural,
        NAMCore
    };

    public class NeuralModelLoader
    {
        IntPtr nativeLoader;

        public NeuralModelLoader()
        {
            nativeLoader = NativeApi.CreateLoader();
        }

        ~NeuralModelLoader()
        {
            NativeApi.DeleteLoader(nativeLoader);
        }

        public void SetLSTMModelLoadMode(EModelLoadMode mode)
        {
            NativeApi.SetLSTMLoadMode(nativeLoader, (int)mode);
        }

        public void SetWaveNetModelLoadMode(EModelLoadMode mode)
        {
            NativeApi.SetWaveNetLoadMode(nativeLoader, (int)mode);
        }

        public void SetDefaultMaxAudioBufferSize(int bufferSize)
        {
            NativeApi.SetDefaultMaxAudioBufferSize(nativeLoader, bufferSize);
        }

        public NeuralModel CreateModelFromFile(string modelPath)
        {
            NeuralModel model = new NeuralModel();

            model.nativeModel = NativeApi.CreateModelFromFile(nativeLoader, modelPath);

            if (model.nativeModel == IntPtr.Zero)
                return null;

            return model;
        }
    }

    public class NeuralModel
    {
        internal IntPtr nativeModel;

        public bool IsStatic { get { return NativeApi.IsStatic(nativeModel);  } }
        public EModelLoadMode LoadMode { get { return (EModelLoadMode)NativeApi.GetLoadMode(nativeModel); } }
        public float SampleRate { get { return NativeApi.GetSampleRate(nativeModel); } }
        public float RecommendedInputDBAdjustment { get { return NativeApi.GetRecommendedInputDBAdjustment(nativeModel); } }
        public float RecommendedOutputDBAdjustment { get { return NativeApi.GetRecommendedOutputDBAdjustment(nativeModel); } }

        ~NeuralModel()
        {
            NativeApi.DeleteModel(nativeModel);
        }

        public void SetMaxAudioBufferSize(int bufferSize)
        {
            NativeApi.SetMaxAudioBufferSize(nativeModel, bufferSize);
        }

        public unsafe void Process(ReadOnlySpan<float> input, Span<float> output, uint numSamples)
        {
            fixed (float* inputPtr = input)
            {
                fixed (float* outputPtr = output)
                {
                    NativeApi.Process(nativeModel, inputPtr, outputPtr, numSamples);
                }
            }
        }
    }
}
