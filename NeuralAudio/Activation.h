#pragma once

#include <cmath>
#include <Eigen/Core>
#include <math_approx/math_approx.hpp>
#include <NeuralAudio/ChannelBuffer.h>

namespace NeuralAudio
{
#ifndef LSTM_MATH
#define LSTM_MATH FastMath
#endif

#ifndef WAVENET_MATH
#define WAVENET_MATH FastMath
#endif

	struct StdMath
	{
		template<typename Buffer>
		static void Tanh(Buffer& channelBuffer)
		{
			float* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = Tanh(data[pos]);
			}
		}

		static inline float Tanh(const float x)
		{
			return std::tanh(x);
		}

		static inline float Sigmoid(float x)
		{
			return 1.0f / (1.0f + std::exp(-x));
		}

		template<typename Buffer>
		static void LeakyReLU(Buffer& channelBuffer)
		{
			float* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = LeakyReLU(data[pos]);
			}
		}

		static inline float LeakyReLU(float x, float negativeSlope)
		{
			return x > 0.0f ? x : negativeSlope * x;
		}

		static inline float LeakyReLU(float x)
		{
			return LeakyReLU(x, 0.01f);
		}
	};

	struct FastMath
	{
		template<typename Buffer>
		static void Tanh(Buffer& channelBuffer)
		{
			float* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = Tanh(data[pos]);
			}
		}

		static inline float Tanh(const float x)
		{
			const float ax = fabsf(x);

			const float x2 = x * x;

			return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
				/ (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
		}

		static inline float Sigmoid(float x)
		{
			return  0.5f * (Tanh(x * 0.5f) + 1);
		}

		template<typename Buffer>
		static void LeakyReLU(Buffer& channelBuffer)
		{
			float* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = LeakyReLU(data[pos]);
			}
		}

		static inline float LeakyReLU(float x, float negativeSlope)
		{
			return x > 0.0f ? x : negativeSlope * x;
		}

		static inline float LeakyReLU(float x)
		{
			return LeakyReLU(x, 0.01f);
		}
	};

	struct EigenMath
	{
		template<typename Buffer>
		static void Tanh(Buffer& channelBuffer)
		{
			auto map = channelBuffer.GetEigenMap();

			map = map.array().tanh();
		}

		static inline float Tanh(const float x)
		{
			return Eigen::numext::tanh(x);
		}

		template<typename Buffer>
		static void LeakyReLU(Buffer& channelBuffer)
		{
			auto map = channelBuffer.GetEigenMap();

			map = (map.array() < 0.0f).select(map.array() * 0.01f, map.array());
		}

		static inline float Sigmoid(float x)
		{
			return  0.5f * (Tanh(x * 0.5f) + 1);
		}
	};
}