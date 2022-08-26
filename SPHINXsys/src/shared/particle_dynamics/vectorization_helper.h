/**
 * ...
 */

#ifndef VECTORIZATION_HELPER_H
#define VECTORIZATION_HELPER_H


#include "xsimd/xsimd.hpp"

namespace SPH
{
	template <class T>
	inline void EstimateSimdSizes(const std::size_t num_iter_total, std::size_t& num_iter_simd, std::size_t& simd_register_size)
	{
		simd_register_size = xsimd::simd_type<T>::size;
		num_iter_simd = num_iter_total - num_iter_total % simd_register_size;
	}

	template <class T>
	inline void EstimateInitialValue(const std::size_t simd_register_size, const T init_value, xsimd::batch<T>& reg_0)
	{
		// TODO: Time critical, initialize simd_register_size at compile time
		T vec_0[4] = {init_value};
		reg_0 = xsimd::load_aligned(&vec_0[0]);
	}

	template <class T>
	inline T VectorizedSum(std::size_t size, const T init_value, const T* values)
	{
		const auto num_iter_total = size;
		std::size_t num_iter_simd{}, simd_register_size{};
		xsimd::batch<T> vec_sum;

		EstimateSimdSizes<T>(num_iter_total, num_iter_simd, simd_register_size);
		EstimateInitialValue<T>(simd_register_size, init_value, vec_sum);

		// Vectorized loop
		for (auto i = 0; i < num_iter_simd; i += simd_register_size)
		{
			auto value_batch = xsimd::load_aligned(&values[i]);
			vec_sum = vec_sum + value_batch;
		}

		// Scalar loop
		T scalar_sum = 0.0;
		for (auto i = num_iter_simd; i < num_iter_total; ++i)
		{
			scalar_sum += values[i];
		}

		return xsimd::reduce_add(vec_sum) + scalar_sum;
	}

	template <class T>
	inline void EstimateError(T& res_scalar, T& res_vector)
	{
		std::cout << " error = " << res_scalar - res_vector << std::endl;
	}

	template <class T>
	inline void DetectSimdRegisterSize()
	{
		// TODO: Detect max. available simd register size for a particular data type T
	}
}
#endif // VECTORIZATION_HELPER_H