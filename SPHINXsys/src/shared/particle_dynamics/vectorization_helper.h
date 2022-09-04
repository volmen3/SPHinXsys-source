/**
 * ...
 */

#ifndef VECTORIZATION_HELPER_H
#define VECTORIZATION_HELPER_H


#include "base_data_type.h"
#include "data_type.h"
#include <large_data_containers.h>
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

	inline Vecd ViscousAccelerationInnerInteraction(
		const StdLargeVec<Vecd>& vel_, const Vecd& vel_i,
		const StdLargeVec<Real>& r_ij_, const StdLargeVec<Real>& dW_ij_, const StdLargeVec<Real>& Vol_,
		Real c_smoothing_length_, Real c_mu_rho_i,
		size_t n_size, const StdLargeVec<size_t>& j_)
	{
		const std::size_t num_iter_simd = n_size - n_size % SIMD_REGISTER_SIZE_REAL_ELEMENTS;

		// Initialization
		xsimd::batch<Real> acceleration_vx, acceleration_vy;
		InitWithDefaultValue<Real>(0.0, acceleration_vx);
		InitWithDefaultValue<Real>(0.0, acceleration_vy);

		// Vectorized loop
		for (size_t n = 0; n < num_iter_simd; n += SIMD_REGISTER_SIZE_REAL_ELEMENTS)
		{
			auto index = xsimd::load_unaligned(&j_[n]);
			auto index_x = index * 2;
			auto index_y = (index * 2) + 1;

			auto Vol_v = xsimd::batch<Real, xsimd::default_arch>::gather(&Vol_[0], index);
			auto vel_xv = xsimd::batch<Real, xsimd::default_arch>::gather(&vel_[0][0], index_x);
			auto vel_yv = xsimd::batch<Real, xsimd::default_arch>::gather(&vel_[0][0], index_y);

			auto dW_ij_v = xsimd::load_unaligned(&dW_ij_[n]);

			auto vel_ixv = xsimd::batch<Real>(vel_i[0]);
			auto vel_iyv = xsimd::batch<Real>(vel_i[1]);

			auto r_ij_v = xsimd::load_unaligned(&r_ij_[n]);
			r_ij_v = r_ij_v + c_smoothing_length_;

			auto vel_derivative_vx = (vel_ixv - vel_xv) / r_ij_v;
			auto vel_derivative_vy = (vel_iyv - vel_yv) / r_ij_v;

			acceleration_vx = acceleration_vx + c_mu_rho_i * vel_derivative_vx * Vol_v * dW_ij_v;
			acceleration_vy = acceleration_vy + c_mu_rho_i * vel_derivative_vy * Vol_v * dW_ij_v;
		}

		// Scalar loop
		Vecd acceleration_s(0);
		for (size_t n = num_iter_simd; n < n_size; ++n)
		{
			auto index_j = j_[n];

			auto vel_derivative_s = (vel_i - vel_[index_j]) / (r_ij_[n] + c_smoothing_length_);
			acceleration_s = acceleration_s + c_mu_rho_i * vel_derivative_s * Vol_[index_j] * dW_ij_[n];
		}

		acceleration_s[0] = acceleration_s[0] + xsimd::reduce_add(acceleration_vx);
		acceleration_s[1] = acceleration_s[1] + xsimd::reduce_add(acceleration_vy);

		return acceleration_s;
	}

	template <class T>
	void EstimateError(T& res_scalar, T& res_vector)
	{
		std::cout << " error = " << res_scalar - res_vector << std::endl;
	}

	void WriteTwoValuesToFile(const std::string& file_name, long long value1, long long value2, char delimiter);

}
#endif // VECTORIZATION_HELPER_H