using Microsoft.VisualBasic;
using System;
using System.Runtime.InteropServices;

namespace NeuralAudio
{
    static class NativeApi
    {
        public const string NEURAL_AUDIO_LIB_NAME = "NeuralAudioCAPI";

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern IntPtr CreateModelFromFile([MarshalAs(UnmanagedType.LPWStr)]string modelPath);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern void DeleteModel(IntPtr model);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern void SetLSTMLoadMode(int loadMode);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern void SetWaveNetLoadMode(int loadMode);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern void SetAudioInputLevelDBu(float audioDBu);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern void SetDefaultMaxAudioBufferSize(int maxSize);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern int GetLoadMode(IntPtr model);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern bool IsStatic(IntPtr model);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern void SetMaxAudioBufferSize(IntPtr model, int maxSize);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern float GetRecommendedInputDBAdjustment(IntPtr model);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern float GetRecommendedOutputDBAdjustment(IntPtr model);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern float GetSampleRate(IntPtr model);

        [DllImport(NEURAL_AUDIO_LIB_NAME)]
        public static extern unsafe void Process(IntPtr model, float* input, float* output, uint numSamples);
    }
}
