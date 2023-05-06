using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace NeuralNet
{
    public class Activation
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float x)
        {
            if (x >= 4f) return 1f;
            float tmp = 1f - 0.25f * x;
            tmp *= tmp;
            tmp *= tmp;
            tmp *= tmp;
            tmp *= tmp;
            return 1f / (1f + tmp);
            //return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tanh(float x)
        {
            return (float)Math.Tanh(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float FastTanh(float x)
        {
            float ax = Math.Abs(x);
            float x2 = x * x;

            return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
                    / (2.44506634652299f + (2.44506634652299f + x2) * Math.Abs(x + 0.814642734961073f * x * ax)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float FastSigmoid(float x)
        {
            return 0.5f * (FastTanh(x * 0.5f) + 1);
        }
    }
}
