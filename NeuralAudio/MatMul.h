#pragma once

namespace NeuralAudio
{
	template <int InChannels, int OutChannels>
	struct MatMul
	{
		static constexpr bool HasKernel() { return ((InChannels == 3) && ((OutChannels == 3) || (OutChannels == 1))); }

		// 3x3 implementation inspired by @jfsantos NAM Core a2fast - https://github.com/sdatkinson/NeuralAmpModelerCore/blob/main/NAM/wavenet/a2_fast.cpp

		static inline void MultiplyInitZero(const float* inData, float* outData, const float* weights, int numFrames)
		{
			static_assert(HasKernel(), "Multiplication not implemented for InChannel/OutChannel combination");

			if constexpr ((InChannels == 3) && (OutChannels == 3))
			{
				const float w00 = weights[0], w10 = weights[1], w20 = weights[2];
				const float w01 = weights[3], w11 = weights[4], w21 = weights[5];
				const float w02 = weights[6], w12 = weights[7], w22 = weights[8];

				for (int frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];
					float* out = &outData[offset];

					float a0 = w00 * in[0];
					float a1 = w10 * in[0];
					float a2 = w20 * in[0];
					a0 += w01 * in[1];
					a1 += w11 * in[1];
					a2 += w21 * in[1];
					a0 += w02 * in[2];
					a1 += w12 * in[2];
					a2 += w22 * in[2];
					out[0] = a0;
					out[1] = a1;
					out[2] = a2;
				}
			}
			else if constexpr ((InChannels == 3) && (OutChannels == 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (int frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];

					float a0 = w0 * in[frame * 3];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					outData[frame] = a0;
				}
			}
		}

		static inline void MultiplyInitColwise(const float* inData, float* outData, const float* weights, const float* initData, int numFrames)
		{
			static_assert(HasKernel(), "Multiplication not implemented for InChannel/OutChannel combination");

			if constexpr ((InChannels == 3) && (OutChannels == 3))
			{
				const float w00 = weights[0], w10 = weights[1], w20 = weights[2];
				const float w01 = weights[3], w11 = weights[4], w21 = weights[5];
				const float w02 = weights[6], w12 = weights[7], w22 = weights[8];

				for (int frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];
					float* out = &outData[offset];

					float a0 = initData[0] + w00 * in[0];
					float a1 = initData[1] + w10 * in[0];
					float a2 = initData[2] + w20 * in[0];
					a0 += w01 * in[1];
					a1 += w11 * in[1];
					a2 += w21 * in[1];
					a0 += w02 * in[2];
					a1 += w12 * in[2];
					a2 += w22 * in[2];
					out[0] = a0;
					out[1] = a1;
					out[2] = a2;
				}
			}
			else if constexpr ((InChannels == 3) && (OutChannels == 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (int frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];

					float a0 = initData[0] + w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					outData[frame] = a0;
				}
			}
		}

		static inline void MultiplyAccumlulate(const float* inData, float* outData, const float* weights, int numFrames)
		{
			static_assert(HasKernel(), "Multiplication not implemented for InChannel/OutChannel combination");

			if constexpr ((InChannels == 3) && (OutChannels == 3))
			{
				const float w00 = weights[0], w10 = weights[1], w20 = weights[2];
				const float w01 = weights[3], w11 = weights[4], w21 = weights[5];
				const float w02 = weights[6], w12 = weights[7], w22 = weights[8];

				for (int frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];
					float* out = &outData[offset];

					float a0 = out[0] + w00 * in[0];
					float a1 = out[1] + w10 * in[0];
					float a2 = out[2] + w20 * in[0];
					a0 += w01 * in[1];
					a1 += w11 * in[1];
					a2 += w21 * in[1];
					a0 += w02 * in[2];
					a1 += w12 * in[2];
					a2 += w22 * in[2];
					out[0] = a0;
					out[1] = a1;
					out[2] = a2;
				}		
			}
			else if constexpr ((InChannels == 3) && (OutChannels == 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (int frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];

					float a0 = outData[frame] + w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					outData[frame] = a0;
				}
			}
		}
	};

}