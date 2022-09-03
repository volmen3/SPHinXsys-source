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
	void InitWithDefaultValue(const T default_value, xsimd::batch<T>& reg_0)
	{
		alignas(SIMD_REGISTER_SIZE_REAL_BYTES) T vec_0[SIMD_REGISTER_SIZE_REAL_ELEMENTS] = { default_value };
		reg_0 = xsimd::load_aligned(&vec_0[0]);
	}

	template <class T>
	T DensitySummationInnerInteraction(const std::size_t current_size, const T W0, const T* W_ij)
	{
		const std::size_t num_iter_simd = current_size - current_size % SIMD_REGISTER_SIZE_REAL_ELEMENTS;
		xsimd::batch<T> sigma_v;

		InitWithDefaultValue<T>(W0, sigma_v);

		// Vectorized loop
		for (size_t i = 0; i < num_iter_simd; i += SIMD_REGISTER_SIZE_REAL_ELEMENTS)
		{
			// TODO: alignment ? 
			auto value_batch = xsimd::load_unaligned(&W_ij[i]);
			sigma_v = sigma_v + value_batch;
		}

		// Scalar loop
		T sigma_s = 0.0;
		for (size_t i = num_iter_simd; i < current_size; ++i)
		{
			sigma_s += W_ij[i];
		}

		return xsimd::reduce_add(sigma_v) + sigma_s;
	}

	template <class T>
	void EstimateError(T& res_scalar, T& res_vector)
	{
		std::cout << " error = " << res_scalar - res_vector << std::endl;
	}

	void WriteTwoValuesToFile(const std::string& file_name, long long value1, long long value2, char delimiter);

}
#endif // VECTORIZATION_HELPER_H