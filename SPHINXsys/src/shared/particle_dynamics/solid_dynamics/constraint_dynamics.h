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
 * @brief 	Here, we define the algorithm classes for solid dynamics.
 * @details 	We consider here a weakly compressible solids.
 * @author	Luhui Han, Chi ZHang and Xiangyu Hu
 */

#ifndef CONSTRAINT_DYNAMICS_H
#define CONSTRAINT_DYNAMICS_H

#include "all_particle_dynamics.h"
#include "general_dynamics.h"
#include "base_kernel.h"
#include "all_body_relations.h"
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
		 * @class ConstrainSolidBodyRegion
		 * @brief Constrain a solid body part with prescribed motion.
		 */
		class ConstrainSolidBodyRegion : public PartSimpleDynamicsByParticle, public SolidDataSimple
		{
		public:
			// TODO: use only body part as argment since body can be referred from it already
			ConstrainSolidBodyRegion(SPHBody &sph_body, BodyPartByParticle &body_part);
			virtual ~ConstrainSolidBodyRegion(){};

		protected:
			StdLargeVec<Vecd> &pos_, &pos0_;
			StdLargeVec<Vecd> &n_, &n0_;
			StdLargeVec<Vecd> &vel_, &acc_;
			virtual Vecd getDisplacement(Vecd &pos_0, Vecd &pos_n) { return pos_n; };
			virtual Vecd getVelocity(Vecd &pos_0, Vecd &pos_n, Vecd &vel_n) { return Vecd(0); };
			virtual Vecd getAcceleration(Vecd &pos_0, Vecd &pos_n, Vecd &acc) { return Vecd(0); };
			virtual SimTK::Rotation getBodyRotation(Vecd &pos_0, Vecd &pos_n, Vecd &acc) { return SimTK::Rotation(); }
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class ConstrainSolidBodySurfaceRegion
		 * @brief Constrain the surface particles of a solid body part with prescribed motion.
		 */
		class ConstrainSolidBodySurfaceRegion : public PartSimpleDynamicsByParticle, public SolidDataSimple
		{
		public:
			ConstrainSolidBodySurfaceRegion(SPHBody &body, BodyPartByParticle &body_part);
			virtual ~ConstrainSolidBodySurfaceRegion(){};

			StdLargeVec<bool> &GetApplyConstrainToParticle() { return apply_constrain_to_particle_; }

		protected:
			StdLargeVec<Vecd> &pos_, &pos0_;
			StdLargeVec<Vecd> &vel_, &acc_;
			StdLargeVec<bool> apply_constrain_to_particle_;

			virtual Vecd getDisplacement(Vecd &pos_0, Vecd &pos_n) { return pos_n; };
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class PositionSolidBody
		 * @brief Moves the body into a defined position in a given time interval - position driven boundary condition
		 */
		class PositionSolidBody : public PartSimpleDynamicsByParticle, public SolidDataSimple
		{
		public:
			PositionSolidBody(SPHBody &sph_body, BodyPartByParticle &body_part, Real start_time, Real end_time, Vecd pos_end_center);
			virtual ~PositionSolidBody(){};
			StdLargeVec<Vecd> &GetParticlePos0() { return pos0_; };
			StdLargeVec<Vecd> &GetParticlePosN() { return pos_; };

		protected:
			StdLargeVec<Vecd> &pos_, &pos0_;
			StdLargeVec<Vecd> &vel_, &acc_;
			Real start_time_, end_time_;
			Vecd pos_0_center_, pos_end_center_, translation_;
			Vecd getDisplacement(size_t index_i, Real dt);
			virtual Vecd getVelocity() { return Vecd(0); };
			virtual Vecd getAcceleration() { return Vecd(0); };
			virtual SimTK::Rotation getBodyRotation() { return SimTK::Rotation(); }
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class PositionScaleSolidBody
		 * @brief Scales the body in a given time interval - position driven boundary condition
		 */
		class PositionScaleSolidBody : public PartSimpleDynamicsByParticle, public SolidDataSimple
		{
		public:
			PositionScaleSolidBody(SPHBody &sph_body, BodyPartByParticle &body_part, Real start_time, Real end_time, Real end_scale);
			virtual ~PositionScaleSolidBody(){};
			StdLargeVec<Vecd> &GetParticlePos0() { return pos0_; };
			StdLargeVec<Vecd> &GetParticlePosN() { return pos_; };

		protected:
			StdLargeVec<Vecd> &pos_, &pos0_;
			StdLargeVec<Vecd> &vel_, &acc_;
			Real start_time_, end_time_, end_scale_;
			Vecd pos_0_center_;
			Vecd getDisplacement(size_t index_i, Real dt);
			virtual Vecd getVelocity() { return Vecd(0); };
			virtual Vecd getAcceleration() { return Vecd(0); };
			virtual SimTK::Rotation getBodyRotation() { return SimTK::Rotation(); }
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class TranslateSolidBody
		 * @brief Translates the body in a given time interval -translation driven boundary condition; only moving the body; end position irrelevant;
		 */
		class TranslateSolidBody : public PartSimpleDynamicsByParticle, public SolidDataSimple
		{
		public:
			TranslateSolidBody(SPHBody &sph_body, BodyPartByParticle &body_part, Real start_time, Real end_time, Vecd translation);
			virtual ~TranslateSolidBody(){};

		protected:
			StdLargeVec<Vecd> &pos_, &pos0_;
			StdLargeVec<Vecd> pos_end_;
			StdLargeVec<Vecd> &vel_, &acc_;
			Real start_time_, end_time_;
			Vecd translation_;
			Vecd getDisplacement(size_t index_i, Real dt);
			virtual Vecd getVelocity() { return Vecd(0); };
			virtual Vecd getAcceleration() { return Vecd(0); };
			virtual SimTK::Rotation getBodyRotation() { return SimTK::Rotation(); }
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class TranslateSolidBodyPart
		 * @brief Translates the body in a given time interval -translation driven boundary condition; only moving the body; end position irrelevant;
		 * Only the particles in a given Bounding Box are translated. The Bounding Box is defined for the undeformed shape.
		 */
		class TranslateSolidBodyPart : public TranslateSolidBody
		{
		public:
			TranslateSolidBodyPart(SPHBody &sph_body, BodyPartByParticle &body_part, Real start_time, Real end_time, Vecd translation, BoundingBox bbox);
			virtual ~TranslateSolidBodyPart(){};

		protected:
			BoundingBox bbox_;
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class ConstrainSolidBodyRegionVelocity
		 * @brief Constrain the velocity of a solid body part.
		 */
		class ConstrainSolidBodyRegionVelocity : public ConstrainSolidBodyRegion
		{
		public:
			ConstrainSolidBodyRegionVelocity(SPHBody &sph_body, BodyPartByParticle &body_part,
											 Vecd constrained_direction = Vecd(0))
				: solid_dynamics::ConstrainSolidBodyRegion(sph_body, body_part),
				  constrain_matrix_(Matd(1.0))
			{
				for (int k = 0; k != Dimensions; ++k)
					constrain_matrix_[k][k] = constrained_direction[k];
			};
			virtual ~ConstrainSolidBodyRegionVelocity(){};

		protected:
			Matd constrain_matrix_;
			virtual Vecd getVelocity(Vecd &pos_0, Vecd &pos_n, Vecd &vel_n)
			{
				return constrain_matrix_ * vel_n;
			};
		};

		/**
		 * @class SoftConstrainSolidBodyRegion
		 * @brief Soft the constrain of a solid body part
		 */
		class SoftConstrainSolidBodyRegion : public PartInteractionDynamicsByParticleWithUpdate,
											 public SolidDataInner
		{
		public:
			SoftConstrainSolidBodyRegion(BaseBodyRelationInner &inner_relation, BodyPartByParticle &body_part);
			virtual ~SoftConstrainSolidBodyRegion(){};

		protected:
			StdLargeVec<Real> &Vol_;
			StdLargeVec<Vecd> &vel_, &acc_;
			StdLargeVec<Vecd> vel_temp_, acc_temp_;
			virtual void Interaction(size_t index_i, Real dt = 0.0) override;
			virtual void Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class ClampConstrainSolidBodyRegion
		 * @brief Constrain a solid body part with prescribed motion and smoothing to mimic the clamping effect.
		 */
		class ClampConstrainSolidBodyRegion : public ParticleDynamics<void>
		{
		public:
			ConstrainSolidBodyRegion constraint_;
			SoftConstrainSolidBodyRegion softening_;

			ClampConstrainSolidBodyRegion(BaseBodyRelationInner &inner_relation, BodyPartByParticle &body_part);
			virtual ~ClampConstrainSolidBodyRegion(){};

			virtual void exec(Real dt = 0.0) override;
			virtual void parallel_exec(Real dt = 0.0) override;
		};

		/**
		 * @class ConstrainSolidBodyMassCenter
		 * @brief Constrain the mass center of a solid body.
		 */
		class ConstrainSolidBodyMassCenter : public ParticleDynamicsSimple, public SolidDataSimple
		{
		public:
			explicit ConstrainSolidBodyMassCenter(SPHBody &sph_body, Vecd constrain_direction = Vecd(1.0));
			virtual ~ConstrainSolidBodyMassCenter(){};

		protected:
			virtual void setupDynamics(Real dt = 0.0) override;
			virtual void Update(size_t index_i, Real dt = 0.0) override;

		private:
			Real total_mass_;
			Matd correction_matrix_;
			Vecd velocity_correction_;
			StdLargeVec<Vecd> &vel_;
			BodyMoment<Vecd> compute_total_momentum_;
		};

		/**
		 * @class ConstrainSolidBodyPartBySimBody
		 * @brief Constrain a solid body part from the motion
		 * computed from Simbody.
		 */
		class ConstrainSolidBodyPartBySimBody : public PartSimpleDynamicsByParticle, public SolidDataSimple
		{
		public:
			ConstrainSolidBodyPartBySimBody(SolidBody &solid_body,
											SolidBodyPartForSimbody &body_part,
											SimTK::MultibodySystem &MBsystem,
											SimTK::MobilizedBody &mobod,
											SimTK::Force::DiscreteForces &force_on_bodies,
											SimTK::RungeKuttaMersonIntegrator &integ);
			virtual ~ConstrainSolidBodyPartBySimBody(){};

		protected:
			StdLargeVec<Vecd> &pos_, &pos0_, &vel_, &n_, &n0_;
			SimTK::MultibodySystem &MBsystem_;
			SimTK::MobilizedBody &mobod_;
			SimTK::Force::DiscreteForces &force_on_bodies_;
			SimTK::RungeKuttaMersonIntegrator &integ_;
			const SimTK::State *simbody_state_;
			Vec3d initial_mobod_origin_location_;

			virtual void setupDynamics(Real dt = 0.0) override;
			void virtual Update(size_t index_i, Real dt = 0.0) override;
		};

		/**
		 * @class TotalForceOnSolidBodyPartForSimBody
		 * @brief Compute the force acting on the solid body part
		 * for applying to simbody forces latter
		 */
		class TotalForceOnSolidBodyPartForSimBody
			: public PartDynamicsByParticleReduce<SimTK::SpatialVec, ReduceSum<SimTK::SpatialVec>>,
			  public SolidDataSimple
		{
		public:
			TotalForceOnSolidBodyPartForSimBody(SolidBody &solid_body,
												SolidBodyPartForSimbody &body_part,
												SimTK::MultibodySystem &MBsystem,
												SimTK::MobilizedBody &mobod,
												SimTK::Force::DiscreteForces &force_on_bodies,
												SimTK::RungeKuttaMersonIntegrator &integ);
			virtual ~TotalForceOnSolidBodyPartForSimBody(){};

		protected:
			StdLargeVec<Real> &mass_;
			StdLargeVec<Vecd> &acc_, &acc_prior_, &pos_;
			SimTK::MultibodySystem &MBsystem_;
			SimTK::MobilizedBody &mobod_;
			SimTK::Force::DiscreteForces &force_on_bodies_;
			SimTK::RungeKuttaMersonIntegrator &integ_;
			const SimTK::State *simbody_state_;
			Vec3d current_mobod_origin_location_;

			virtual void SetupReduce() override;
			virtual SimTK::SpatialVec ReduceFunction(size_t index_i, Real dt = 0.0) override;
		};
	}
}
#endif // CONSTRAINT_DYNAMICS_H