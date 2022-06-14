/**
 * @file 	fluid_dynamics.cpp
 * @author	Chi ZHang and Xiangyu Hu
 */

#include "fluid_dynamics_inner.h"
#include "fluid_dynamics_inner.hpp"

namespace SPH
{
	//=================================================================================================//
	namespace fluid_dynamics
	{
		//=================================================================================================//
		FluidInitialCondition::
			FluidInitialCondition(FluidBody &fluid_body)
			: OldParticleDynamicsSimple(fluid_body), FluidDataSimple(fluid_body),
			  pos_n_(particles_->pos_n_), vel_n_(particles_->vel_n_) {}
		//=================================================================================================//
		DensitySummationInner::DensitySummationInner(BaseBodyRelationInner &inner_relation)
			: LocalParticleDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), rho_n_(particles_->rho_n_), mass_(particles_->mass_),
			  rho_sum_(particles_->rho_sum_),
			  W0_(body_->sph_adaptation_->getKernel()->W0(Vecd(0))),
			  rho0_(particles_->rho0_), inv_sigma0_(1.0 / particles_->sigma0_) {}
		//=================================================================================================//
		void DensitySummationInner::interaction(size_t index_i, Real dt)
		{
			/** Inner interaction. */
			Real sigma = W0_;
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
				sigma += inner_neighborhood.W_ij_[n];

			rho_sum_[index_i] = sigma * rho0_ * inv_sigma0_;
		}
		//=================================================================================================//
		void DensitySummationInner::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				rho_n_[index_i] = ReinitializedDensity(rho_sum_[index_i], rho0_, rho_n_[index_i]);
			}

			for (size_t index_i = particle_range.begin(); index_i !=  particle_range.end(); ++index_i)
			{
				Vol_[index_i] = mass_[index_i] / rho_n_[index_i];
			}
		}
		//=================================================================================================//
		ViscousAccelerationInner::ViscousAccelerationInner(BaseBodyRelationInner &inner_relation)
			: OldInteractionDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), rho_n_(particles_->rho_n_), p_(particles_->p_),
			  vel_n_(particles_->vel_n_),
			  dvel_dt_prior_(particles_->dvel_dt_prior_),
			  mu_(material_->ReferenceViscosity()),
			  smoothing_length_(sph_adaptation_->ReferenceSmoothingLength()) {}
		//=================================================================================================//
		void ViscousAccelerationInner::Interaction(size_t index_i, Real dt)
		{
			Real rho_i = rho_n_[index_i];
			const Vecd &vel_i = vel_n_[index_i];

			Vecd acceleration(0), vel_derivative(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];

				// viscous force
				vel_derivative = (vel_i - vel_n_[index_j]) / (inner_neighborhood.r_ij_[n] + 0.01 * smoothing_length_);
				acceleration += 2.0 * mu_ * vel_derivative * Vol_[index_j] * inner_neighborhood.dW_ij_[n] / rho_i;
			}

			dvel_dt_prior_[index_i] += acceleration;
		}
		//=================================================================================================//
		void AngularConservativeViscousAccelerationInner::Interaction(size_t index_i, Real dt)
		{
			Real rho_i = rho_n_[index_i];
			const Vecd &vel_i = vel_n_[index_i];

			Vecd acceleration(0);
			Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd &e_ij = inner_neighborhood.e_ij_[n];
				Real r_ij = inner_neighborhood.r_ij_[n];

				/** The following viscous force is given in Monaghan 2005 (Rep. Prog. Phys.), it seems that
				 * this formulation is more accurate than the previous one for Taylor-Green-Vortex flow. */
				Real v_r_ij = dot(vel_i - vel_n_[index_j], r_ij * e_ij);
				Real eta_ij = 8.0 * mu_ * v_r_ij / (r_ij * r_ij + 0.01 * smoothing_length_);
				acceleration += eta_ij * Vol_[index_j] / rho_i * inner_neighborhood.dW_ij_[n] * e_ij;
			}

			dvel_dt_prior_[index_i] += acceleration;
		}
		//=================================================================================================//
		TransportVelocityCorrectionInner::
			TransportVelocityCorrectionInner(BaseBodyRelationInner &inner_relation)
			: OldInteractionDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), rho_n_(particles_->rho_n_),
			  pos_n_(particles_->pos_n_),
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
			Real rho_i = rho_n_[index_i];

			Vecd acceleration_trans(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd nablaW_ij = inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];

				// acceleration for transport velocity
				acceleration_trans -= 2.0 * p_background_ * Vol_[index_j] * nablaW_ij / rho_i;
			}

			if (surface_indicator_[index_i] == 0)
				pos_n_[index_i] += acceleration_trans * dt * dt * 0.5;
		}
		//=================================================================================================//
		AcousticTimeStepSize::AcousticTimeStepSize(SPHBody &sph_body)
			: LocalParticleDynamicsReduce<Real, ReduceMax>(sph_body, 0.0),
			  FluidDataSimple(sph_body), rho_n_(particles_->rho_n_),
			  p_(particles_->p_), vel_n_(particles_->vel_n_),
			  smoothing_length_(sph_body.sph_adaptation_->ReferenceSmoothingLength()) {}
		//=================================================================================================//
		Real AcousticTimeStepSize::reduceRange(const IndexRange &particle_range, Real dt)
		{
			Real temp = reference_;
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				temp = operation_(temp, material_->getSoundSpeed(p_[index_i], rho_n_[index_i]) + vel_n_[index_i].norm());
			}
			return temp;
		}
		//=================================================================================================//
		Real AcousticTimeStepSize::outputResult(Real reduced_value)
		{
			particles_->signal_speed_max_ = reduced_value;
			// since the particle does not change its configuration in pressure relaxation step
			// I chose a time-step size according to Eulerian method
			return 0.6 * smoothing_length_ / (reduced_value + TinyReal);
		}
		//=================================================================================================//
		AdvectionTimeStepSize::AdvectionTimeStepSize(SPHBody &sph_body, Real U_max)
			: LocalParticleDynamicsReduce<Real, ReduceMax>(sph_body, U_max * U_max),
			  FluidDataSimple(sph_body), vel_n_(particles_->vel_n_),
			  smoothing_length_(sph_body.sph_adaptation_->ReferenceSmoothingLength()),
			  viscous_speed_(material_->ReferenceViscosity() / material_->ReferenceDensity() / smoothing_length_) {}
		//=================================================================================================//
		Real AdvectionTimeStepSize::reduceRange(const IndexRange &particle_range, Real dt)
		{
			Real temp = reference_;
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				temp = operation_(temp, vel_n_[index_i].normSqr());
			}
			return temp;
		}
		//=================================================================================================//
		Real AdvectionTimeStepSize::outputResult(Real reduced_value)
		{
			Real speed_max = SMAX(sqrt(reduced_value), viscous_speed_);
			particles_->speed_max_ = speed_max;
			return 0.25 * smoothing_length_ / (speed_max + TinyReal);
		}
		//=================================================================================================//
		AdvectionTimeStepSizeForImplicitViscosity::
			AdvectionTimeStepSizeForImplicitViscosity(FluidBody &fluid_body, Real U_max)
			: AdvectionTimeStepSize(fluid_body, U_max) {}
		//=================================================================================================//
		Real AdvectionTimeStepSizeForImplicitViscosity::outputResult(Real reduced_value)
		{
			Real speed_max = sqrt(reduced_value);
			particles_->speed_max_ = speed_max;
			return 0.25 * smoothing_length_ / (speed_max + TinyReal);
		}
		//=================================================================================================//
		VorticityInner::
			VorticityInner(BaseBodyRelationInner &inner_relation)
			: OldInteractionDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), vel_n_(particles_->vel_n_)
		{
			particles_->registerAVariable(vorticity_, "VorticityInner");
			particles_->addAVariableToWrite<AngularVecd>("VorticityInner");
		}
		//=================================================================================================//
		void VorticityInner::Interaction(size_t index_i, Real dt)
		{
			const Vecd &vel_i = vel_n_[index_i];

			AngularVecd vorticity(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd r_ij = inner_neighborhood.r_ij_[n] * inner_neighborhood.e_ij_[n];

				Vecd vel_diff = vel_i - vel_n_[index_j];
				vorticity += SimTK::cross(vel_diff, r_ij) * Vol_[index_j] * inner_neighborhood.dW_ij_[n];
			}

			vorticity_[index_i] = vorticity;
		}
		//=================================================================================================//
		BaseRelaxation::BaseRelaxation(BaseBodyRelationInner &inner_relation)
			: LocalParticleDynamics(*inner_relation.sph_body_),
			  FluidDataInner(inner_relation),
			  Vol_(particles_->Vol_), mass_(particles_->mass_), rho_n_(particles_->rho_n_),
			  p_(particles_->p_), drho_dt_(particles_->drho_dt_),
			  pos_n_(particles_->pos_n_), vel_n_(particles_->vel_n_),
			  dvel_dt_(particles_->dvel_dt_),
			  dvel_dt_prior_(particles_->dvel_dt_prior_) {}
		//=================================================================================================//
		BasePressureRelaxation::
			BasePressureRelaxation(BaseBodyRelationInner &inner_relation) : BaseRelaxation(inner_relation) {}
		//=================================================================================================//
		void BasePressureRelaxation::initializeRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				rho_n_[index_i] += drho_dt_[index_i] * dt * 0.5;
			}

			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				Vol_[index_i] = mass_[index_i] / rho_n_[index_i];
			}

			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				p_[index_i] = material_->getPressure(rho_n_[index_i]);
			}

			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				pos_n_[index_i] += vel_n_[index_i] * dt * 0.5;
			}
		}
		//=================================================================================================//
		void BasePressureRelaxation::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				vel_n_[index_i] += dvel_dt_[index_i] * dt;
			}
		}
		//=================================================================================================//
		Vecd BasePressureRelaxation::computeNonConservativeAcceleration(size_t index_i)
		{
			Real rho_i = rho_n_[index_i];
			Real p_i = p_[index_i];
			Vecd acceleration = dvel_dt_prior_[index_i];
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Real dW_ij = inner_neighborhood.dW_ij_[n];
				const Vecd &e_ij = inner_neighborhood.e_ij_[n];

				Real rho_j = rho_n_[index_j];
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
		void BaseDensityRelaxation::initializeRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				pos_n_[index_i] += vel_n_[index_i] * dt * 0.5;
			}
		}
		//=================================================================================================//
		void BaseDensityRelaxation::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				rho_n_[index_i] += drho_dt_[index_i] * dt * 0.5;
			}
		}
		//=================================================================================================//
		PressureRelaxationInnerOldroyd_B ::
			PressureRelaxationInnerOldroyd_B(BaseBodyRelationInner &inner_relation)
			: PressureRelaxationDissipativeRiemannInner(inner_relation),
			  tau_(DynamicCast<ViscoelasticFluidParticles>(this, body_->base_particles_)->tau_),
			  dtau_dt_(DynamicCast<ViscoelasticFluidParticles>(this, body_->base_particles_)->dtau_dt_) {}
		//=================================================================================================//
		void PressureRelaxationInnerOldroyd_B::
			initializeRange(const IndexRange &particle_range, Real dt)
		{
			PressureRelaxationDissipativeRiemannInner::initializeRange(particle_range, dt);

			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				tau_[index_i] += dtau_dt_[index_i] * dt * 0.5;
			}
		}
		//=================================================================================================//
		void PressureRelaxationInnerOldroyd_B::interaction(size_t index_i, Real dt)
		{
			PressureRelaxationDissipativeRiemannInner::interaction(index_i, dt);

			Real rho_i = rho_n_[index_i];
			Matd tau_i = tau_[index_i];

			Vecd acceleration(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd nablaW_ij = inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];

				// elastic force
				acceleration += (tau_i + tau_[index_j]) * nablaW_ij * Vol_[index_j] / rho_i;
			}

			dvel_dt_[index_i] += acceleration;
		}
		//=================================================================================================//
		DensityRelaxationInnerOldroyd_B::
			DensityRelaxationInnerOldroyd_B(BaseBodyRelationInner &inner_relation)
			: DensityRelaxationDissipativeRiemannInner(inner_relation),
			  tau_(DynamicCast<ViscoelasticFluidParticles>(this, body_->base_particles_)->tau_),
			  dtau_dt_(DynamicCast<ViscoelasticFluidParticles>(this, body_->base_particles_)->dtau_dt_)
		{
			Oldroyd_B_Fluid *oldroy_b_fluid = DynamicCast<Oldroyd_B_Fluid>(this, body_->base_material_);
			mu_p_ = oldroy_b_fluid->ReferencePolymericViscosity();
			lambda_ = oldroy_b_fluid->getReferenceRelaxationTime();
		}
		//=================================================================================================//
		void DensityRelaxationInnerOldroyd_B::interaction(size_t index_i, Real dt)
		{
			DensityRelaxationDissipativeRiemannInner::interaction(index_i, dt);

			Vecd vel_i = vel_n_[index_i];
			Matd tau_i = tau_[index_i];

			Matd stress_rate(0);
			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Vecd nablaW_ij = inner_neighborhood.dW_ij_[n] * inner_neighborhood.e_ij_[n];

				Matd velocity_gradient = -SimTK::outer((vel_i - vel_n_[index_j]), nablaW_ij) * Vol_[index_j];
				stress_rate += ~velocity_gradient * tau_i + tau_i * velocity_gradient - tau_i / lambda_ +
							   (~velocity_gradient + velocity_gradient) * mu_p_ / lambda_;
			}

			dtau_dt_[index_i] = stress_rate;
		}
		//=================================================================================================//
		void DensityRelaxationInnerOldroyd_B::updateRange(const IndexRange &particle_range, Real dt)
		{
			DensityRelaxationDissipativeRiemannInner::updateRange(particle_range, dt);

			for (size_t index_i = particle_range.begin(); index_i != particle_range.end(); ++index_i)
			{
				tau_[index_i] += dtau_dt_[index_i] * dt * 0.5;
			}
		}
		//=================================================================================================//
	}
	//=================================================================================================//
}
//=================================================================================================//