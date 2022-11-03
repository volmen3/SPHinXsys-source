/**
 * @file 	fluid_dynamics.cpp
 * @author	Chi ZHang and Xiangyu Hu
 */

#include "fluid_dynamics_inner.h"
#include "fluid_dynamics_inner.hpp"
#include "vectorization.h"

namespace SPH
{
	//=================================================================================================//
	namespace fluid_dynamics
	{
		//=================================================================================================//
		FluidInitialCondition::
			FluidInitialCondition(FluidBody &fluid_body)
			: ParticleDynamicsSimple(fluid_body), FluidDataSimple(fluid_body),
			  pos_(particles_->pos_), vel_(particles_->vel_) {}
		//=================================================================================================//
		DensitySummationInner::DensitySummationInner(BaseBodyRelationInner &inner_relation)
			: InteractionDynamicsWithUpdate(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), rho_(particles_->rho_), mass_(particles_->mass_),
			  rho_sum_(particles_->rho_sum_),
			  W0_(sph_adaptation_->getKernel()->W0(Vecd(0))),
			  rho0_(particles_->rho0_), inv_sigma0_(1.0 / particles_->sigma0_) {}
		//=================================================================================================//
		void DensitySummationInner::Interaction(size_t index_i, Real dt)
		{
			/** Inner interaction. */
			Real sigma = W0_;
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
				sigma += inner_neighborhood.W_ij_[n];

			rho_sum_[index_i] = sigma * rho0_ * inv_sigma0_;
		}
		//=================================================================================================//
		void DensitySummationInner::InteractionBatch(size_t index_i, Real dt)
		{
			const Neighborhood& inner_neighborhood = inner_configuration_[index_i];
			const std::size_t num_iter_simd = inner_neighborhood.current_size_ - 
				inner_neighborhood.current_size_ % SIMD_REGISTER_SIZE_REAL_ELEMENTS;

			xsimd::batch<Real> sigma_v;
			InitWithDefaultValue(W0_, sigma_v);

			// Vectorized loop
			for (size_t i = 0; i < num_iter_simd; i += SIMD_REGISTER_SIZE_REAL_ELEMENTS)
			{
				auto value_batch = xsimd::load_unaligned(&inner_configuration_[index_i].W_ij_[i]);
				sigma_v = sigma_v + value_batch;
			}

			// Scalar loop
			Real sigma_s = 0.0;
			for (size_t i = num_iter_simd; i < inner_configuration_[index_i].current_size_; ++i)
			{
				sigma_s += inner_configuration_[index_i].W_ij_[i];
			}

			rho_sum_[index_i] = (xsimd::reduce_add(sigma_v) + sigma_s) * rho0_ * inv_sigma0_;
		}
		//=================================================================================================//
		void DensitySummationInner::Update(size_t index_i, Real dt)
		{
			rho_[index_i] = ReinitializedDensity(rho_sum_[index_i], rho0_, rho_[index_i]);
			Vol_[index_i] = mass_[index_i] / rho_[index_i];
		}
		//=================================================================================================//
		ViscousAccelerationInner::ViscousAccelerationInner(BaseBodyRelationInner &inner_relation)
			: InteractionDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), rho_(particles_->rho_), p_(particles_->p_),
			  vel_(particles_->vel_),
			  acc_prior_(particles_->acc_prior_),
			  mu_(material_->ReferenceViscosity()),
			  smoothing_length_(sph_adaptation_->ReferenceSmoothingLength()) {}
		//=================================================================================================//
		void ViscousAccelerationInner::Interaction(size_t index_i, Real dt)
		{
			Real rho_i = rho_[index_i];
			const Vecd &vel_i = vel_[index_i];

			Vecd acceleration(0), vel_derivative(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];

				//viscous force
				vel_derivative = (vel_i - vel_[index_j]) / (inner_neighborhood.r_ij_[n] + 0.01 * smoothing_length_);
				acceleration += 2.0 * mu_ * vel_derivative * Vol_[index_j] * inner_neighborhood.dW_ij_[n] / rho_i;
			}

			acc_prior_[index_i] += acceleration;
		}
		//=================================================================================================//
		void ViscousAccelerationInner::InteractionBatch(size_t index_i, Real dt)
		{
			const Real rho_i = rho_[index_i];
			const Vecd& vel_i = vel_[index_i];
			const Neighborhood& inner_neighborhood = inner_configuration_[index_i];

			const std::size_t num_iter_simd = inner_neighborhood.current_size_ - 
				inner_neighborhood.current_size_ % SIMD_REGISTER_SIZE_REAL_ELEMENTS;

			VecBatchSph<2> acceleration_v;
			InitWithDefaultValueVecBatch(0.0, acceleration_v);

			// Vectorized loop
			for (size_t n = 0; n < num_iter_simd; n += SIMD_REGISTER_SIZE_REAL_ELEMENTS)
			{
				auto vel_iv = VecBatchSph<2>({ vel_i[0] }, { vel_i[1] });
				auto vel_v = LoadIndirectVecBatchSph<SIMD_REGISTER_SIZE_REAL_ELEMENTS>(&inner_neighborhood.j_[n], vel_);
				auto Vol_v = LoadIndirectSingleBatchSph<SIMD_REGISTER_SIZE_REAL_ELEMENTS>(&inner_neighborhood.j_[n], Vol_);
				auto r_ij_v = xsimd::load_unaligned(&inner_neighborhood.r_ij_[n]);
				auto dW_ij_v = xsimd::load_unaligned(&inner_neighborhood.dW_ij_[n]);

				auto vel_derivative_v = (vel_iv - vel_v);
				vel_derivative_v[0] = vel_derivative_v[0] / (r_ij_v + 0.01 * smoothing_length_);
				vel_derivative_v[1] = vel_derivative_v[1] / (r_ij_v + 0.01 * smoothing_length_);

				acceleration_v[0] = acceleration_v[0] + 2 * mu_ / rho_i * vel_derivative_v[0] * Vol_v * dW_ij_v;
				acceleration_v[1] = acceleration_v[1] + 2 * mu_ / rho_i * vel_derivative_v[1] * Vol_v * dW_ij_v;
			}

			// Scalar loop
			Vecd acceleration_s(0);
			for (size_t n = num_iter_simd; n < inner_neighborhood.current_size_; ++n)
			{
				auto index_j = inner_neighborhood.j_[n];

				auto vel_derivative_s = (vel_i - vel_[index_j]) / (inner_neighborhood.r_ij_[n] + 0.01 * smoothing_length_);
				acceleration_s = acceleration_s + 2 * mu_ / rho_i * vel_derivative_s * Vol_[index_j] * inner_neighborhood.dW_ij_[n];
			}

			acceleration_s[0] = acceleration_s[0] + xsimd::reduce_add(acceleration_v[0]);
			acceleration_s[1] = acceleration_s[1] + xsimd::reduce_add(acceleration_v[1]);

			acc_prior_[index_i] += acceleration_s;
		}
		//=================================================================================================//
		void AngularConservativeViscousAccelerationInner::Interaction(size_t index_i, Real dt)
		{
			Real rho_i = rho_[index_i];
			const Vecd &vel_i = vel_[index_i];

			Vecd acceleration(0);
			Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd &e_ij = inner_neighborhood.e_ij_[n];
				Real r_ij = inner_neighborhood.r_ij_[n];

				/** The following viscous force is given in Monaghan 2005 (Rep. Prog. Phys.), it seems that 
				 * this formulation is more accurate than the previous one for Taylor-Green-Vortex flow. */
				Real v_r_ij = dot(vel_i - vel_[index_j], r_ij * e_ij);
				Real eta_ij = 8.0 * mu_ * v_r_ij / (r_ij * r_ij + 0.01 * smoothing_length_);
				acceleration += eta_ij * Vol_[index_j] / rho_i * inner_neighborhood.dW_ij_[n] * e_ij;
			}

			acc_prior_[index_i] += acceleration;
		}
		//=================================================================================================//
		void AngularConservativeViscousAccelerationInner::InteractionBatch(size_t index_i, Real dt)
		{
			const Real rho_i = rho_[index_i];
			const Vecd& vel_i = vel_[index_i];
			const Neighborhood& inner_neighborhood = inner_configuration_[index_i];
			const auto vel_iv = VecBatchSph<2>({ vel_i[0] }, { vel_i[1] });

			const std::size_t num_iter_simd = inner_neighborhood.current_size_ -
				inner_neighborhood.current_size_ % SIMD_REGISTER_SIZE_REAL_ELEMENTS;

			VecBatchSph<2> acceleration_v;
			InitWithDefaultValueVecBatch(0.0, acceleration_v);

			// Vectorized loop
			for (size_t n = 0; n < num_iter_simd; n += SIMD_REGISTER_SIZE_REAL_ELEMENTS)
			{
				auto vel_v = LoadIndirectVecBatchSph<SIMD_REGISTER_SIZE_REAL_ELEMENTS>(&inner_neighborhood.j_[n], vel_);
				auto Vol_v = LoadIndirectSingleBatchSph<SIMD_REGISTER_SIZE_REAL_ELEMENTS>(&inner_neighborhood.j_[n], Vol_);
				auto e_ij_v = LoadDirectVecBatchSph<SIMD_REGISTER_SIZE_REAL_ELEMENTS>(n, inner_neighborhood.e_ij_);
				auto r_ij_v = xsimd::load_unaligned(&inner_neighborhood.r_ij_[n]);
				auto dW_ij_v = xsimd::load_unaligned(&inner_neighborhood.dW_ij_[n]);

				auto vel_i_sub_vel_v = (vel_iv - vel_v);
				auto r_ij_mul_e_ij_vx = r_ij_v * e_ij_v[0];
				auto r_ij_mul_e_ij_vy = r_ij_v * e_ij_v[1];

				// Dot product: (vel_iv - vel_v) & r_ij_v * e_ij_v
				auto v_r_ij_v = vel_i_sub_vel_v[0] * r_ij_mul_e_ij_vx + vel_i_sub_vel_v[1] * r_ij_mul_e_ij_vy;

				auto eta_ij_v = 8.0 * mu_ * v_r_ij_v / (r_ij_v * r_ij_v + 0.01 * smoothing_length_);

				acceleration_v[0] = acceleration_v[0] + eta_ij_v * Vol_v / rho_i * dW_ij_v * e_ij_v[0];
				acceleration_v[1] = acceleration_v[1] + eta_ij_v * Vol_v / rho_i * dW_ij_v * e_ij_v[1];
			}

			// Scalar loop
			Vecd acceleration_s(0);
			for (size_t n = num_iter_simd; n < inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];

				auto v_r_ij = dot(vel_i - vel_[index_j], inner_neighborhood.r_ij_[n] * inner_neighborhood.e_ij_[n]);
				auto eta_ij = 8.0 * mu_ * v_r_ij / (inner_neighborhood.r_ij_[n] * inner_neighborhood.r_ij_[n] + 0.01 * smoothing_length_);
				acceleration_s = acceleration_s + eta_ij * Vol_[index_j] / rho_i * inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];
			}

			acceleration_s[0] = acceleration_s[0] + xsimd::reduce_add(acceleration_v[0]);
			acceleration_s[1] = acceleration_s[1] + xsimd::reduce_add(acceleration_v[1]);

			acc_prior_[index_i] += acceleration_s;
		}
		//=================================================================================================//
		TransportVelocityCorrectionInner::
			TransportVelocityCorrectionInner(BaseBodyRelationInner &inner_relation)
			: InteractionDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), rho_(particles_->rho_),
			  pos_(particles_->pos_),
			  surface_indicator_(particles_->surface_indicator_), p_background_(0) {}
		//=================================================================================================//
		void TransportVelocityCorrectionInner::setupDynamics(Real dt)
		{
			Real speed_max = particles_->speed_max_;
			Real density = material_->ReferenceDensity();
			p_background_ = 7.0 * density * speed_max * speed_max;
		}
		//=================================================================================================//
		void TransportVelocityCorrectionInner::Interaction(size_t index_i, Real dt)
		{
			Real rho_i = rho_[index_i];

			Vecd acceleration_trans(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd nablaW_ij = inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];

				//acceleration for transport velocity
				acceleration_trans -= 2.0 * p_background_ * Vol_[index_j] * nablaW_ij / rho_i;
			}

			if (surface_indicator_[index_i] == 0)
				pos_[index_i] += acceleration_trans * dt * dt * 0.5;
		}
		//=================================================================================================//
		AcousticTimeStepSize::AcousticTimeStepSize(FluidBody &fluid_body)
			: ParticleDynamicsReduce<Real, ReduceMax>(fluid_body),
			  FluidDataSimple(fluid_body), rho_(particles_->rho_),
			  p_(particles_->p_), vel_(particles_->vel_),
			  smoothing_length_(sph_adaptation_->ReferenceSmoothingLength())
		{
			initial_reference_ = 0.0;
		}
		//=================================================================================================//
		Real AcousticTimeStepSize::ReduceFunction(size_t index_i, Real dt)
		{
			return material_->getSoundSpeed(p_[index_i], rho_[index_i]) + vel_[index_i].norm();
		}
		//=================================================================================================//
		Real AcousticTimeStepSize::OutputResult(Real reduced_value)
		{
			particles_->signal_speed_max_ = reduced_value;
			//since the particle does not change its configuration in pressure relaxation step
			//I chose a time-step size according to Eulerian method
			return 0.6 * smoothing_length_ / (reduced_value + TinyReal);
		}
		//=================================================================================================//
		AdvectionTimeStepSize::AdvectionTimeStepSize(FluidBody &fluid_body, Real U_max)
			: ParticleDynamicsReduce<Real, ReduceMax>(fluid_body),
			  FluidDataSimple(fluid_body), vel_(particles_->vel_),
			  smoothing_length_(sph_adaptation_->ReferenceSmoothingLength())
		{
			Real rho_0 = material_->ReferenceDensity();
			Real mu = material_->ReferenceViscosity();
			Real viscous_speed = mu / rho_0 / smoothing_length_;
			Real u_max = SMAX(viscous_speed, U_max);
			initial_reference_ = u_max * u_max;
		}
		//=================================================================================================//
		Real AdvectionTimeStepSize::ReduceFunction(size_t index_i, Real dt)
		{
			return vel_[index_i].normSqr();
		}
		//=================================================================================================//
		Real AdvectionTimeStepSize::OutputResult(Real reduced_value)
		{
			Real speed_max = sqrt(reduced_value);
			particles_->speed_max_ = speed_max;
			return 0.25 * smoothing_length_ / (speed_max + TinyReal);
		}
		//=================================================================================================//
		AdvectionTimeStepSizeForImplicitViscosity::
			AdvectionTimeStepSizeForImplicitViscosity(FluidBody &fluid_body, Real U_max)
			: AdvectionTimeStepSize(fluid_body, U_max)
		{
			initial_reference_ = U_max * U_max;
		}
		//=================================================================================================//
		VorticityInner::
			VorticityInner(BaseBodyRelationInner &inner_relation)
			: InteractionDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), vel_(particles_->vel_)
		{
			particles_->registerVariable(vorticity_, "VorticityInner");
			particles_->addVariableToWrite<AngularVecd>("VorticityInner");
		}
		//=================================================================================================//
		void VorticityInner::Interaction(size_t index_i, Real dt)
		{
			const Vecd &vel_i = vel_[index_i];

			AngularVecd vorticity(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd r_ij = inner_neighborhood.r_ij_[n] * inner_neighborhood.e_ij_[n];

				Vecd vel_diff = vel_i - vel_[index_j];
				vorticity += SimTK::cross(vel_diff, r_ij) * Vol_[index_j] * inner_neighborhood.dW_ij_[n];
			}

			vorticity_[index_i] = vorticity;
		}
		//=================================================================================================//
		BaseRelaxation::BaseRelaxation(BaseBodyRelationInner &inner_relation)
			: ParticleDynamics1Level(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), mass_(particles_->mass_), rho_(particles_->rho_),
			  p_(particles_->p_), drho_dt_(particles_->drho_dt_),
			  pos_(particles_->pos_), vel_(particles_->vel_),
			  acc_(particles_->acc_),
			  acc_prior_(particles_->acc_prior_) {}
		//=================================================================================================//
		BasePressureRelaxation::
			BasePressureRelaxation(BaseBodyRelationInner &inner_relation) : BaseRelaxation(inner_relation) {}
		//=================================================================================================//
		void BasePressureRelaxation::Initialization(size_t index_i, Real dt)
		{
			rho_[index_i] += drho_dt_[index_i] * dt * 0.5;
			Vol_[index_i] = mass_[index_i] / rho_[index_i];
			p_[index_i] = material_->getPressure(rho_[index_i]);
			pos_[index_i] += vel_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
		void BasePressureRelaxation::Update(size_t index_i, Real dt)
		{
			vel_[index_i] += acc_[index_i] * dt;
		}
		//=================================================================================================//
		Vecd BasePressureRelaxation::computeNonConservativeAcceleration(size_t index_i)
		{
			Real rho_i = rho_[index_i];
			Real p_i = p_[index_i];
			Vecd acceleration = acc_prior_[index_i];
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Real dW_ij = inner_neighborhood.dW_ij_[n];
				const Vecd &e_ij = inner_neighborhood.e_ij_[n];

				Real rho_j = rho_[index_j];
				Real p_j = p_[index_j];

				Real p_star = (rho_i * p_j + rho_j * p_i) / (rho_i + rho_j);
				acceleration += (p_i - p_star) * Vol_[index_j] * dW_ij * e_ij / rho_i;
			}
			return acceleration;
		}
		//=================================================================================================//
		BaseDensityRelaxation::
			BaseDensityRelaxation(BaseBodyRelationInner &inner_relation) : BaseRelaxation(inner_relation) {}
		//=================================================================================================//
		void BaseDensityRelaxation::Initialization(size_t index_i, Real dt)
		{
			pos_[index_i] += vel_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
		void BaseDensityRelaxation::Update(size_t index_i, Real dt)
		{
			rho_[index_i] += drho_dt_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
		PressureRelaxationInnerOldroyd_B ::
			PressureRelaxationInnerOldroyd_B(BaseBodyRelationInner &inner_relation)
			: PressureRelaxationDissipativeRiemannInner(inner_relation),
			  tau_(DynamicCast<ViscoelasticFluidParticles>(this, sph_body_->base_particles_)->tau_),
			  dtau_dt_(DynamicCast<ViscoelasticFluidParticles>(this, sph_body_->base_particles_)->dtau_dt_) {}
		//=================================================================================================//
		void PressureRelaxationInnerOldroyd_B::Initialization(size_t index_i, Real dt)
		{
			PressureRelaxationDissipativeRiemannInner::Initialization(index_i, dt);

			tau_[index_i] += dtau_dt_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
		void PressureRelaxationInnerOldroyd_B::Interaction(size_t index_i, Real dt)
		{
			PressureRelaxationDissipativeRiemannInner::Interaction(index_i, dt);

			Real rho_i = rho_[index_i];
			Matd tau_i = tau_[index_i];

			Vecd acceleration(0);
			Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd nablaW_ij = inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];

				//elastic force
				acceleration += (tau_i + tau_[index_j]) * nablaW_ij * Vol_[index_j] / rho_i;
			}

			acc_[index_i] += acceleration;
		}
		//=================================================================================================//
		DensityRelaxationInnerOldroyd_B::
			DensityRelaxationInnerOldroyd_B(BaseBodyRelationInner &inner_relation)
			: DensityRelaxationDissipativeRiemannInner(inner_relation),
			  tau_(DynamicCast<ViscoelasticFluidParticles>(this, sph_body_->base_particles_)->tau_),
			  dtau_dt_(DynamicCast<ViscoelasticFluidParticles>(this, sph_body_->base_particles_)->dtau_dt_)
		{
			Oldroyd_B_Fluid *oldroy_b_fluid = DynamicCast<Oldroyd_B_Fluid>(this, sph_body_->base_material_);
			mu_p_ = oldroy_b_fluid->ReferencePolymericViscosity();
			lambda_ = oldroy_b_fluid->getReferenceRelaxationTime();
		}
		//=================================================================================================//
		void DensityRelaxationInnerOldroyd_B::Interaction(size_t index_i, Real dt)
		{
			DensityRelaxationDissipativeRiemannInner::Interaction(index_i, dt);

			Vecd vel_i = vel_[index_i];
			Matd tau_i = tau_[index_i];

			Matd stress_rate(0);
			Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd nablaW_ij = inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];

				Matd velocity_gradient = -SimTK::outer((vel_i - vel_[index_j]), nablaW_ij) * Vol_[index_j];
				stress_rate += ~velocity_gradient * tau_i + tau_i * velocity_gradient - tau_i / lambda_ +
							   (~velocity_gradient + velocity_gradient) * mu_p_ / lambda_;
			}

			dtau_dt_[index_i] = stress_rate;
		}
		//=================================================================================================//
		void DensityRelaxationInnerOldroyd_B::Update(size_t index_i, Real dt)
		{
			DensityRelaxationDissipativeRiemannInner::Update(index_i, dt);

			tau_[index_i] += dtau_dt_[index_i] * dt * 0.5;
		}
		//=================================================================================================//
	}
	//=================================================================================================//
}
//=================================================================================================//