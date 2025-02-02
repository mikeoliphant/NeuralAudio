#pragma once

#include <cstddef>

namespace NeuralAudio
{
	template <std::size_t ... Is, typename F>
	void ForEachIndex(std::index_sequence<Is...>, F&& f)
	{
		int dummy[] = { 0, /* Handles empty Is. following cast handle evil operator comma */
					   (static_cast<void>(f(std::integral_constant<std::size_t, Is>())), 0)... };
		static_cast<void>(dummy); // avoid warning for unused variable
	}

	template <std::size_t N, typename F>
	void ForEachIndex(F&& f)
	{
		ForEachIndex(std::make_index_sequence<N>(), std::forward<F>(f));
	}
}