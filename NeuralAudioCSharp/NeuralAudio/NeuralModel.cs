using System;
using System.Runtime.CompilerServices;

namespace NeuralAudio
{
    public class NeuralModel
    {
        public enum EModelLoadMode
        {
            Internal,
            RTNeural,
            NAMCore
        };

        IntPtr nativeModel;

        static void SetLSTMModelLoadMode(EModelLoadMode mode)
        {
            NativeApi.SetLSTMLoadMode((int)mode);
        }

        static void SetWaveNetModelLoadMode(EModelLoadMode mode)
        {
            NativeApi.SetWaveNetLoadMode((int)mode);
        }
        
        static void SetDefaultMaxAudioBufferSize(int bufferSize)
        {
            NativeApi.SetDefaultMaxAudioBufferSize(bufferSize);
        }

        public bool IsStatic { get { return NativeApi.IsStatic(nativeModel);  } }
        public EModelLoadMode LoadMode { get { return (EModelLoadMode)NativeApi.GetLoadMode(nativeModel); } }
        public float SampleRate { get { return NativeApi.GetSampleRate(nativeModel); } }
        public float RecommendedInputDBAdjustment { get { return NativeApi.GetRecommendedInputDBAdjustment(nativeModel); } }
        public float RecommendedOutputDBAdjustment { get { return NativeApi.GetRecommendedOutputDBAdjustment(nativeModel); } }

        public static NeuralModel FromFile(string modelPath)
        {
            NeuralModel model = new NeuralModel();

            IntPtr nativeModel = NativeApi.CreateModelFromFile(modelPath);

            model.nativeModel = nativeModel;

            return model;
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
