#pragma once

#include <cmath>
#include <Eigen/Core>
#include <math_approx/math_approx.hpp>

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
		template <typename Matrix>
		static void Tanh(Matrix *x)
		{
			float* data = x->data();
			size_t size = x->rows() * x->cols();

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

		static inline float LeakyRelu(float x, float negativeSlope)
		{
			return x > 0.0f ? x : negativeSlope * x;
		}

		static inline float LeakyRelu(float x)
		{
			return LeakyRelu(x, 0.01f);
		}
	};

	struct FastMath
	{
		template <typename Matrix>
		static void Tanh(Matrix *x)
		{			
			float* data = x->data();
			size_t size = x->rows() * x->cols();

			for (size_t pos = 0; pos < size; pos++)
			{
				data[pos] = Tanh(data[pos]);
			}
		}

		static inline float Tanh(const float x)
		{
			//return std::tanh(x);

			//return math_approx::tanh<5>(x);

			const float ax = fabsf(x);

			const float x2 = x * x;

			return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
				/ (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
		}

		static inline float Sigmoid(float x)
		{
			//return math_approx::sigmoid_exp<5>(x);

			//return 1.0f / (1.0f + std::exp(-x));
			return  0.5f * (Tanh(x * 0.5f) + 1);
		}

		static inline float LeakyRelu(float x, float negativeSlope)
		{
			return x > 0.0f ? x : negativeSlope * x;
		}

		static inline float LeakyRelu(float x)
		{
			return LeakyRelu(x, 0.01f);
		}
	};

	struct EigenMath
	{
		template <typename Matrix>
		static void Tanh(Matrix* x)
		{
			*x = x->array().tanh();
		}

		static inline float Tanh(const float x)
		{
			return Eigen::numext::tanh(x);
		}

		static inline float Sigmoid(float x)
		{
			return  0.5f * (Tanh(x * 0.5f) + 1);
		}
	};
}