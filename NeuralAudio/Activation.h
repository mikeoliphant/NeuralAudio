#pragma once

#include <cmath>

namespace NeuralAudio
{
	inline float FastTanh(const float x)
	{
		//return std::tanh(x);

		const float ax = fabsf(x);
		const float x2 = x * x;

		return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
			/ (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
	}

	inline float FastSigmoid(float x)
	{
		//return 1.0f / (1.0f + std::exp(-x));
		return  0.5f * (FastTanh(x * 0.5f) + 1);
	}
}