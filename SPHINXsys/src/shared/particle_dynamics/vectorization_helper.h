/**
 * ...
 */

#ifndef VECTORIZATION_HELPER_H
#define VECTORIZATION_HELPER_H


#include "base_data_type.h"
#include "xsimd/xsimd.hpp"


namespace SPH
{
	// Size of a SIMD register of SPH::Real values for default ISA (in bytes)
	static const auto SIMD_REGISTER_SIZE_REAL_BYTES = sizeof(xsimd::types::simd_register<Real, xsimd::default_arch>);

	// Size of a SIMD register of SPH::Real values for default ISA (in elements)
	static const auto SIMD_REGISTER_SIZE_REAL_ELEMENTS = xsimd::simd_type<Real>::size;


	template <class T>
	void EstimateInitialValue(const T init_value, xsimd::batch<T>& reg_0)
	{
		alignas(SIMD_REGISTER_SIZE_REAL_BYTES) T vec_0[SIMD_REGISTER_SIZE_REAL_ELEMENTS] = {init_value};
		reg_0 = xsimd::load_aligned(&vec_0[0]);
	}

	template <class T>
	T VectorizedSum(const std::size_t num_iter_total, const T init_value, const T* values)
	{
		const std::size_t num_iter_simd = num_iter_total - num_iter_total % SIMD_REGISTER_SIZE_REAL_ELEMENTS;
		xsimd::batch<T> vec_sum;

		EstimateInitialValue<T>(init_value, vec_sum);

		// Vectorized loop
		for (auto i = 0; i < num_iter_simd; i += SIMD_REGISTER_SIZE_REAL_ELEMENTS)
		{
			// TODO: alignment ? 
			auto value_batch = xsimd::load_unaligned(&values[i]);
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
	void EstimateError(T& res_scalar, T& res_vector)
	{
		std::cout << " error = " << res_scalar - res_vector << std::endl;
	}

	void WriteTwoValuesToFile(const std::string& file_name, long long value1, long long value2, char delimiter);

}
#endif // VECTORIZATION_HELPER_H