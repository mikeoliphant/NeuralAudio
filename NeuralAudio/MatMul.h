#pragma once

namespace NeuralAudio
{
	template <int InChannels, int OutChannels>
	struct MatMul
	{
		#define IsKernel(In, Out) ((InChannels == In) && (OutChannels == Out))

		static constexpr bool HasKernel() { return (IsKernel(3, 3) || IsKernel(8, 1) || IsKernel(3, 1) || IsKernel(1, 3)); }

		// 3x3 implementation inspired by @jfsantos NAM Core a2fast - https://github.com/sdatkinson/NeuralAmpModelerCore/blob/main/NAM/wavenet/a2_fast.cpp

		static inline void MultiplyInitZero(const float* inData, float* outData, const float* weights, size_t numFrames)
		{
			static_assert(HasKernel(), "Multiplication not implemented for InChannel/OutChannel combination");

			if constexpr (IsKernel(3, 3))
			{
				const float w00 = weights[0], w10 = weights[1], w20 = weights[2];
				const float w01 = weights[3], w11 = weights[4], w21 = weights[5];
				const float w02 = weights[6], w12 = weights[7], w22 = weights[8];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
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
			else if constexpr (IsKernel(8, 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2], w3 = weights[3];
				const float w4 = weights[4], w5 = weights[5], w6 = weights[6], w7 = weights[7];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * InChannels;
					const float* in = &inData[offset];

					float a0 = w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					a0 += w3 * in[3];
					a0 += w4 * in[4];
					a0 += w5 * in[5];
					a0 += w6 * in[6];
					a0 += w7 * in[7];
					outData[frame] = a0;
				}
			}
			else if constexpr (IsKernel(3, 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
					const float* in = &inData[offset];

					float a0 = w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					outData[frame] = a0;
				}
			}
			else if constexpr (IsKernel(1, 3))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
					float* out = &outData[offset];

					const float in = inData[frame];

					out[0] = w0 * in;
					out[1] = w1 * in;
					out[2] = w2 * in;
				}
			}
		}

		static inline void MultiplyInitColwise(const float* inData, float* outData, const float* weights, const float* initData, size_t numFrames)
		{
			static_assert(HasKernel(), "Multiplication not implemented for InChannel/OutChannel combination");

			if constexpr (IsKernel(3, 3))
			{
				const float w00 = weights[0], w10 = weights[1], w20 = weights[2];
				const float w01 = weights[3], w11 = weights[4], w21 = weights[5];
				const float w02 = weights[6], w12 = weights[7], w22 = weights[8];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
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
			else if constexpr (IsKernel(8, 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2], w3 = weights[3];
				const float w4 = weights[4], w5 = weights[5], w6 = weights[6], w7 = weights[7];

				const float init = initData[0];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * InChannels;
					const float* in = &inData[offset];

					float a0 = init + w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					a0 += w3 * in[3];
					a0 += w4 * in[4];
					a0 += w5 * in[5];
					a0 += w6 * in[6];
					a0 += w7 * in[7];
					outData[frame] = a0;
				}
			}
			else if constexpr (IsKernel(3, 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
					const float* in = &inData[offset];

					float a0 = initData[0] + w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					outData[frame] = a0;
				}
			}
			else if constexpr (IsKernel(1, 3))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];
				const float init0 = inData[0], init1 = inData[1], init2 = inData[2];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
					float* out = &outData[offset];

					const float in = inData[frame];

					out[0] = init0 + (w0 * in);
					out[1] = init1 + (w1 * in);
					out[2] = init2 + (w2 * in);
				}
			}
		}

		static inline void MultiplyAccumlulate(const float* inData, float* outData, const float* weights, size_t numFrames)
		{
			static_assert(HasKernel(), "Multiplication not implemented for InChannel/OutChannel combination");

			if constexpr (IsKernel(3, 3))
			{
				const float w00 = weights[0], w10 = weights[1], w20 = weights[2];
				const float w01 = weights[3], w11 = weights[4], w21 = weights[5];
				const float w02 = weights[6], w12 = weights[7], w22 = weights[8];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
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
			else if constexpr (IsKernel(8, 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2], w3 = weights[3];
				const float w4 = weights[4], w5 = weights[5], w6 = weights[6], w7 = weights[7];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * InChannels;
					const float* in = &inData[offset];

					float a0 = outData[frame] + w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					a0 += w3 * in[3];
					a0 += w4 * in[4];
					a0 += w5 * in[5];
					a0 += w6 * in[6];
					a0 += w7 * in[7];
					outData[frame] = a0;
				}
			}
			else if constexpr (IsKernel(3, 1))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const int offset = frame * 3;
					const float* in = &inData[offset];

					float a0 = outData[frame] + w0 * in[0];
					a0 += w1 * in[1];
					a0 += w2 * in[2];
					outData[frame] = a0;
				}
			}
			else if constexpr (IsKernel(1, 3))
			{
				const float w0 = weights[0], w1 = weights[1], w2 = weights[2];

				for (size_t frame = 0; frame < numFrames; frame++)
				{
					const size_t offset = frame * 3;
					float* out = &outData[offset];

					const float in = inData[frame];

					out[0] += w0 * in;
					out[1] += w1 * in;
					out[2] += w2 * in;
				}
			}
		}
	};

}