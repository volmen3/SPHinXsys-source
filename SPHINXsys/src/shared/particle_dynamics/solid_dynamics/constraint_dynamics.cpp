/**
 * @file 	constraint_dynamics.cpp
 * @author	Luhui Han, Chi Zhang and Xiangyu Hu
 */

#include "constraint_dynamics.h"
#include "general_dynamics.h"

#include <numeric>

using namespace SimTK;

namespace SPH
{
	namespace solid_dynamics
	{
		//=================================================================================================//
		VelocityConstraint::VelocityConstraint(SPHBody &sph_body)
			: LocalParticleDynamics(sph_body), SolidDataSimple(sph_body),
			  vel_n_(particles_->vel_n_), dvel_dt_(particles_->dvel_dt_),
			  vel_ave_(particles_->vel_ave_), dvel_dt_ave_(particles_->dvel_dt_ave_) {}
		//=================================================================================================//
		void VelocityConstraint::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				vel_n_[index_i] = getVelocity(index_i);
				vel_ave_[index_i] = vel_n_[index_i];
			}
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				dvel_dt_[index_i] = getAcceleration(index_i);
				/** the average values are prescribed also. */
				dvel_dt_ave_[index_i] = dvel_dt_[index_i];
			}
		}
		//=================================================================================================//
		void VelocityConstraint::updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt)
		{
			for (size_t i = entry_range.begin(); i < entry_range.end(); ++i)
			{
				size_t index_i = particle_list[i];

				vel_n_[index_i] = getVelocity(index_i);
				vel_ave_[index_i] = vel_n_[index_i];

				dvel_dt_[index_i] = getAcceleration(index_i);
				/** the average values are prescribed also. */
				dvel_dt_ave_[index_i] = dvel_dt_[index_i];
			}
		}
		//=================================================================================================//
		DisplacementConstraint::
			DisplacementConstraint(SPHBody &body)
			: LocalParticleDynamics(body), SolidDataSimple(body),
			  pos_n_(particles_->pos_n_), pos_0_(particles_->pos_0_) {}
		//=================================================================================================//
		void DisplacementConstraint::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				pos_n_[index_i] += getDisplacement(index_i);
			}
		}
		//=================================================================================================//
		void DisplacementConstraint::updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt)
		{
			for (size_t i = entry_range.begin(); i < entry_range.end(); ++i)
			{
				size_t index_i = particle_list[i];
				pos_n_[index_i] += getDisplacement(index_i);
			}
		}
		//=================================================================================================//
		SoftConstrain::SoftConstrain(BaseBodyRelationInner &inner_relation)
			: VelocityConstraint(*inner_relation.sph_body_),
			  DataDelegateInner<SolidBody, SolidParticles, Solid, DataDelegateEmptyBase>(inner_relation),
			  Vol_(particles_->Vol_)
		{
			particles_->registerAVariable(vel_temp_, "TemporaryVelocity");
			particles_->registerAVariable(dvel_dt_temp_, "TemporaryAcceleration");
		}
		//=================================================================================================//
		void SoftConstrain::initializeRange(const IndexRange &particle_range, Real dt)
		{
			VelocityConstraint::updateRange(particle_range, dt);
		}
		//=================================================================================================//
		void SoftConstrain::initializeList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt)
		{
			VelocityConstraint::updateList(entry_range, particle_list, dt);
		}
		//=================================================================================================//
		void SoftConstrain::interaction(size_t index_i, Real dt)
		{
			Real ttl_weight(Eps);
			Vecd vel_i = vel_n_[index_i];
			Vecd dvel_dt_i = dvel_dt_[index_i];

			const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
			for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
			{
				size_t index_j = inner_neighborhood.j_[n];
				Real weight_j = inner_neighborhood.W_ij_[n] * Vol_[index_j];

				ttl_weight += weight_j;
				vel_i += vel_n_[index_j] * weight_j;
				dvel_dt_i += dvel_dt_[index_j] * weight_j;
			}

			vel_temp_[index_i] = vel_i / ttl_weight;
			dvel_dt_temp_[index_i] = dvel_dt_i / ttl_weight;
		}
		//=================================================================================================//
		void SoftConstrain::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				vel_n_[index_i] = vel_temp_[index_i];
				vel_ave_[index_i] = vel_n_[index_i];
			}
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				dvel_dt_[index_i] = dvel_dt_temp_[index_i];
				dvel_dt_ave_[index_i] = dvel_dt_[index_i];
			}
		}
		//=================================================================================================//
		void SoftConstrain::updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt)
		{
			for (size_t i = entry_range.begin(); i < entry_range.end(); ++i)
			{
				size_t index_i = particle_list[i];
				vel_n_[index_i] = vel_temp_[index_i];
				vel_ave_[index_i] = vel_n_[index_i];
				dvel_dt_[index_i] = dvel_dt_temp_[index_i];
				dvel_dt_ave_[index_i] = dvel_dt_[index_i];
			}
		}
		//=================================================================================================//
		ConstrainBodyMassCenter::
			ConstrainBodyMassCenter(SPHBody &sph_body, Vecd constrain_direction)
			: LocalParticleDynamics(sph_body), SolidDataSimple(sph_body),
			  correction_matrix_(Matd(1.0)), vel_n_(particles_->vel_n_),
			  compute_total_momentum_(sph_body, "Velocity")
		{
			for (int i = 0; i != Dimensions; ++i)
				correction_matrix_[i][i] = constrain_direction[i];
			SimpleDynamicsReduce<VariableSummation<Real>> compute_total_mass_(sph_body, "Mass");
			total_mass_ = compute_total_mass_.parallel_exec();
		}
		//=================================================================================================//
		void ConstrainBodyMassCenter::setupDynamics(Real dt)
		{
			velocity_correction_ = correction_matrix_ * compute_total_momentum_.parallel_exec(dt) / total_mass_;
		}
		//=================================================================================================//
		void ConstrainBodyMassCenter::updateRange(const IndexRange &particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				vel_n_[index_i] -= velocity_correction_;
			}
		}
		//=================================================================================================//
		ConstraintBySimBody::ConstraintBySimBody(SolidBody &solid_body,
												 SolidBodyPartForSimbody &body_part,
												 SimTK::MultibodySystem &MBsystem,
												 SimTK::MobilizedBody &mobod,
												 SimTK::Force::DiscreteForces &force_on_bodies,
												 SimTK::RungeKuttaMersonIntegrator &integ)
			: VelocityConstraint(solid_body),
			  pos_n_(particles_->pos_n_), pos_0_(particles_->pos_0_), 
			  n_(particles_->n_),n_0_(particles_->n_0_),
			  MBsystem_(MBsystem), mobod_(mobod), force_on_bodies_(force_on_bodies), integ_(integ)
		{
			simbody_state_ = &integ_.getState();
			MBsystem_.realize(*simbody_state_, Stage::Acceleration);
			initial_mobod_origin_location_ = mobod_.getBodyOriginLocation(*simbody_state_);
		}
		//=================================================================================================//
		void ConstraintBySimBody::setupDynamics(Real dt)
		{
			body_->setNewlyUpdated();
			simbody_state_ = &integ_.getState();
			MBsystem_.realize(*simbody_state_, Stage::Acceleration);
		}
		//=================================================================================================//
		TotalForceForSimBody::TotalForceForSimBody(SolidBody &solid_body,
												   SolidBodyPartForSimbody &body_part,
												   SimTK::MultibodySystem &MBsystem,
												   SimTK::MobilizedBody &mobod,
												   SimTK::Force::DiscreteForces &force_on_bodies,
												   SimTK::RungeKuttaMersonIntegrator &integ)
			: LocalParticleDynamicsReduce<SimTK::SpatialVec, ReduceSum<SimTK::SpatialVec>>(
				  solid_body, SimTK::SpatialVec((Vec3(0), Vec3(0)))),
			  SolidDataSimple(solid_body),
			  force_from_fluid_(particles_->force_from_fluid_), contact_force_(particles_->contact_force_),
			  pos_n_(particles_->pos_n_),
			  MBsystem_(MBsystem), mobod_(mobod), force_on_bodies_(force_on_bodies), integ_(integ) {}
		//=================================================================================================//
		void TotalForceForSimBody::setupDynamics(Real dt)
		{
			simbody_state_ = &integ_.getState();
			MBsystem_.realize(*simbody_state_, Stage::Acceleration);
			current_mobod_origin_location_ = mobod_.getBodyOriginLocation(*simbody_state_);
		}
		//=================================================================================================//
	}
}
