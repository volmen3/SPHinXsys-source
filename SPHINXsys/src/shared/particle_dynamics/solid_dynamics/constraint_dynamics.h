/* -------------------------------------------------------------------------*
 *								SPHinXsys									*
 * --------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle	*
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for	*
 * physical accurate simulation and aims to model coupled industrial dynamic *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH	*
 * (smoothed particle hydrodynamics), a meshless computational method using	*
 * particle discretization.													*
 *																			*
 * SPHinXsys is partially funded by German Research Foundation				*
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1				*
 * and HU1527/12-1.															*
 *                                                                           *
 * Portions copyright (c) 2017-2020 Technical University of Munich and		*
 * the authors' affiliations.												*
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * --------------------------------------------------------------------------*/
/**
 * @file 	constraint_dynamics.h
 * @brief 	Here, we define the algorithm classes for constrain motion of solid body or body part.
 * @details Note that the constraint is added between the first and second half steps pf stress relaxation.
 * @author	Chi Zhang and Xiangyu Hu
 */

#ifndef CONSTRAINT_DYNAMICS_H
#define CONSTRAINT_DYNAMICS_H

#include "all_particle_dynamics.h"
#include "general_dynamics.h"
#include "base_kernel.h"
#include "body_relation.h"
#include "solid_body.h"
#include "solid_particles.h"
#include "elastic_solid.h"

namespace SPH
{
	template <typename VariableType>
	class BodySummation;
	template <typename VariableType>
	class BodyMoment;

	namespace solid_dynamics
	{
		//----------------------------------------------------------------------
		//		for general solid dynamics
		//----------------------------------------------------------------------
		typedef DataDelegateSimple<SolidBody, SolidParticles, Solid> SolidDataSimple;
		typedef DataDelegateInner<SolidBody, SolidParticles, Solid> SolidDataInner;

		/**
		 * @class BaseConstraint
		 * @brief Constrain a solid body part with prescribed motion.
		 * Note the average values for FSI are prescribed also.
		 */
		class VelocityConstraint : public LocalParticleDynamics, public SolidDataSimple
		{
		public:
			explicit VelocityConstraint(SPHBody &sph_body);
			virtual ~VelocityConstraint(){};

		protected:
			StdLargeVec<Vecd> &vel_n_, &dvel_dt_, &vel_ave_, &dvel_dt_ave_;
			virtual Vecd getVelocity(size_t index_i) { return Vecd(0); };
			virtual Vecd getAcceleration(size_t index_i) { return Vecd(0); };
			void updateRange(const IndexRange &particle_range, Real dt = 0.0);
			void updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0);
		};

		/**
		 * @class DisplacementConstraint
		 * @brief Moves the body into a defined position in a given time interval - position driven boundary condition
		 * Note the average values for FSI are prescribed also.
		 */
		class DisplacementConstraint : public LocalParticleDynamics, public SolidDataSimple
		{
		public:
			explicit DisplacementConstraint(SPHBody &sph_body);
			virtual ~DisplacementConstraint(){};

		protected:
			StdLargeVec<Vecd> &pos_n_, &pos_0_;
			virtual Vecd getDisplacement(size_t index_i) = 0;
			void updateRange(const IndexRange &particle_range, Real dt = 0.0);
			void updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0);
		};

		/**
		 * @class SoftConstrain
		 * @brief Soft the constrain of a solid body or body part
		 */
		class SoftConstrain : public VelocityConstraint,
							  public DataDelegateInner<SolidBody, SolidParticles, Solid, DataDelegateEmptyBase>
		{
		public:
			explicit SoftConstrain(BaseBodyRelationInner &inner_relation);
			virtual ~SoftConstrain(){};
			void initializeRange(const IndexRange &particle_range, Real dt = 0.0);
			void initializeList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0);
			void interaction(size_t index_i, Real dt = 0.0);
			void updateRange(const IndexRange &particle_range, Real dt = 0.0);
			void updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0);

		protected:
			StdLargeVec<Real> &Vol_;
			StdLargeVec<Vecd> vel_temp_, dvel_dt_temp_;
		};

		/**
		 * @class ConstrainBodyMassCenter
		 * @brief Constrain the mass center of a solid body.
		 */
		class ConstrainBodyMassCenter : public LocalParticleDynamics, public SolidDataSimple
		{
		public:
			explicit ConstrainBodyMassCenter(SPHBody &sph_body, Vecd constrain_direction);
			virtual ~ConstrainBodyMassCenter(){};
			virtual void setupDynamics(Real dt = 0.0) override;
			void updateRange(const IndexRange &particle_range, Real dt = 0.0);

		private:
			Real total_mass_;
			Matd correction_matrix_;
			Vecd velocity_correction_;
			StdLargeVec<Vecd> &vel_n_;
			SimpleDynamicsReduce<BodyMoment<Vecd>> compute_total_momentum_;
		};

		/**
		 * @class ConstraintBySimBody
		 * @brief Constrain a solid body part from the motion
		 * computed from Simbody.
		 */
		class ConstraintBySimBody : public VelocityConstraint
		{
		public:
			ConstraintBySimBody(SolidBody &solid_body,
								SolidBodyPartForSimbody &body_part,
								SimTK::MultibodySystem &MBsystem,
								SimTK::MobilizedBody &mobod,
								SimTK::Force::DiscreteForces &force_on_bodies,
								SimTK::RungeKuttaMersonIntegrator &integ);
			virtual ~ConstraintBySimBody(){};

			virtual void setupDynamics(Real dt = 0.0) override;
			void updateRange(const IndexRange &particle_range, Real dt = 0.0);
			void updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> &pos_n_, &pos_0_, &n_, &n_0_;
			SimTK::MultibodySystem &MBsystem_;
			SimTK::MobilizedBody &mobod_;
			SimTK::Force::DiscreteForces &force_on_bodies_;
			SimTK::RungeKuttaMersonIntegrator &integ_;
			const SimTK::State *simbody_state_;
			Vec3d initial_mobod_origin_location_;
		};

		/**
		 * @class TotalForceForSimBody
		 * @brief Compute the force acting on the solid body part
		 * for applying to simbody forces latter
		 */
		class TotalForceForSimBody
			: public LocalParticleDynamicsReduce<SimTK::SpatialVec, ReduceSum<SimTK::SpatialVec>>,
			  public SolidDataSimple
		{
		public:
			TotalForceForSimBody(SolidBody &solid_body,
								 SolidBodyPartForSimbody &body_part,
								 SimTK::MultibodySystem &MBsystem,
								 SimTK::MobilizedBody &mobod,
								 SimTK::Force::DiscreteForces &force_on_bodies,
								 SimTK::RungeKuttaMersonIntegrator &integ);
			virtual ~TotalForceForSimBody(){};

			virtual void setupDynamics(Real dt = 0.0) override;
			SimTK::SpatialVec reduceRange(const IndexRange &particle_range, Real dt = 0.0);
			SimTK::SpatialVec reduceList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0);

		protected:
			StdLargeVec<Vecd> &force_from_fluid_, &contact_force_, &pos_n_;
			SimTK::MultibodySystem &MBsystem_;
			SimTK::MobilizedBody &mobod_;
			SimTK::Force::DiscreteForces &force_on_bodies_;
			SimTK::RungeKuttaMersonIntegrator &integ_;
			const SimTK::State *simbody_state_;
			Vec3d current_mobod_origin_location_;
		};
	}
}
#endif // CONSTRAINT_DYNAMICS_H