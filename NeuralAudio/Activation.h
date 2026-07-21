#pragma once

#include <cmath>
#include <Eigen/Core>
#include <math_approx/math_approx.hpp>
#include <NeuralAudio/ChannelBuffer.h>

#define TCONST(num) static_cast<T>(num)

namespace NeuralAudio
{
#ifndef LSTM_MATH
#define LSTM_MATH FastMath
#endif

#ifndef WAVENET_MATH
#define WAVENET_MATH FastMath
#endif

	template<typename T>
	struct StdMath
	{
		template<size_t Channels>
		static void Tanh(ChannelRowSpan<T, Channels>& channelBuffer)
		{
			T* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = Tanh(data[pos]);
			}
		}

		static inline T Tanh(const T x)
		{
			return std::tanh(x);
		}

		static inline T Sigmoid(T x)
		{
			return TCONST(1.0) / (TCONST(1.0)+ std::exp(-x));
		}

		template<size_t Channels>
		static void LeakyReLU(ChannelRowSpan<T, Channels> channelBuffer)
		{
			T* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = LeakyReLU(data[pos]);
			}
		}

		static inline T LeakyReLU(T x, T negativeSlope)
		{
			return x > TCONST(0) ? x : negativeSlope * x;
		}

		static inline T LeakyReLU(T x)
		{
			return LeakyReLU(x, TCONST(0.01));
		}
	};

	template<typename T>
	struct FastMath
	{
		template<size_t Channels>
		static void Tanh(ChannelRowSpan<T, Channels>& channelBuffer)
		{
			T* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = Tanh(data[pos]);
			}
		}

		static inline T Tanh(const T x)
		{
			const T ax = std::abs(x);

			const T x2 = x * x;

			return (x * (TCONST(2.45550750702956) + TCONST(2.45550750702956) * ax + (TCONST(0.893229853513558) + TCONST(0.821226666969744) * ax) * x2)
				/ (TCONST(2.44506634652299) + (TCONST(2.44506634652299) + x2) * std::abs(x + TCONST(0.814642734961073) * x * ax)));
		}

		static inline T Sigmoid(T x)
		{
			return  TCONST(0.5) * (Tanh(x * TCONST(0.5) + TCONST(1)));
		}

		template<size_t Channels>
		static void LeakyReLU(ChannelRowSpan<T, Channels> channelBuffer)
		{
			T* data = channelBuffer.GetData();
			size_t size = channelBuffer.GetSize();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = LeakyReLU(data[pos]);
			}
		}

		static inline T LeakyReLU(T x, T negativeSlope)
		{
			return x > TCONST(0.0) ? x : negativeSlope * x;
		}

		static inline T LeakyReLU(T x)
		{
			return LeakyReLU(x, TCONST(0.01));
		}
	};

	template<typename T>
	struct EigenMath
	{
		template<size_t Channels>
		static void Tanh(ChannelRowSpan<T, Channels>& channelBuffer)
		{
			auto map = channelBuffer.GetEigenMap();

			map = map.array().tanh();
		}

		static inline float Tanh(const float x)
		{
			return Eigen::numext::tanh(x);
		}

		template<size_t Channels>
		static void LeakyReLU(ChannelRowSpan<T, Channels> channelBuffer)
		{
			auto map = channelBuffer.GetEigenMap();

			map = (map.array() < TCONST(0.0)).select(map.array() * TCONST(0.01), map.array());
		}

		static inline T Sigmoid(T x)
		{
			return  TCONST(0.5) * (Tanh(x * TCONST(0.5)) + TCONST(1));
		}
	};
}