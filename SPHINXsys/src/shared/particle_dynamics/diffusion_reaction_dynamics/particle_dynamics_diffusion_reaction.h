/* -----------------------------------------------------------------------------*
 *                               SPHinXsys                                      *
 * -----------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle    *
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for       *
 * physical accurate simulation and aims to model coupled industrial dynamic    *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH      *
 * (smoothed particle hydrodynamics), a meshless computational method using     *
 * particle discretization.                                                     *
 *                                                                              *
 * SPHinXsys is partially funded by German Research Foundation                  *
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1,               *
 * HU1527/12-1 and HU1527/12-4.                                                 *
 *                                                                              *
 * Portions copyright (c) 2017-2022 Technical University of Munich and          *
 * the authors' affiliations.                                                   *
 *                                                                              *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may      *
 * not use this file except in compliance with the License. You may obtain a    *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.           *
 *                                                                              *
 * -----------------------------------------------------------------------------*/
/**
* @file 	particle_dynamics_diffusion_reaction.h
* @brief 	This is the particle dynamics applicable for all type bodies
* 			TODO: there is an issue on applying corrected configuration for contact bodies.
* @author	Xiaojing Tang, Chi ZHang and Xiangyu Hu
*/

#ifndef PARTICLE_DYNAMICS_DIFFUSION_REACTION_H
#define PARTICLE_DYNAMICS_DIFFUSION_REACTION_H

#include "all_particle_dynamics.h"
#include "diffusion_reaction_particles.h"
#include "diffusion_reaction.h"

namespace SPH
{
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	using DiffusionReactionSimpleData =
		DataDelegateSimple<BodyType,
						   DiffusionReactionParticles<BaseParticlesType>,
						   DiffusionReaction<BaseMaterialType>>;

	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	using DiffusionReactionInnerData =
		DataDelegateInner<BodyType,
						  DiffusionReactionParticles<BaseParticlesType>,
						  DiffusionReaction<BaseMaterialType>>;

	template <class BodyType, class BaseParticlesType, class BaseMaterialType,
			  class ContactBodyType, class ContactBaseParticlesType, class ContactBaseMaterialType>
	using DiffusionReactionContactData =
		DataDelegateContact<BodyType,
							DiffusionReactionParticles<BaseParticlesType>,
							DiffusionReaction<BaseMaterialType>,
							ContactBodyType, DiffusionReactionParticles<ContactBaseParticlesType>,
							DiffusionReaction<ContactBaseMaterialType>, DataDelegateEmptyBase>;

	/**
	 * @class  DiffusionReactionInitialCondition
	 * @brief pure abstract class for initial conditions
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class DiffusionReactionInitialCondition
		: public ParticleDynamicsSimple,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
	public:
		explicit DiffusionReactionInitialCondition(BodyType &body);
		virtual ~DiffusionReactionInitialCondition(){};

	protected:
		StdLargeVec<Vecd> &pos_;
		StdVec<StdLargeVec<Real>> &species_n_;
	};

	/**
	 * @class GetDiffusionTimeStepSize
	 * @brief Computing the time step size based on diffusion coefficient and particle smoothing length
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class GetDiffusionTimeStepSize
		: public ParticleDynamics<Real>,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
	public:
		explicit GetDiffusionTimeStepSize(BodyType &body);
		virtual ~GetDiffusionTimeStepSize(){};

		virtual Real exec(Real dt = 0.0) override { return diff_time_step_; };
		virtual Real parallel_exec(Real dt = 0.0) override { return exec(dt); };

	protected:
		Real diff_time_step_;
	};

	/**
	 * @class RelaxationOfAllDiffusionSpeciesInner
	 * @brief Compute the diffusion relaxation process of all species
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class RelaxationOfAllDiffusionSpeciesInner
		: public InteractionDynamicsWithUpdate,
		  public DiffusionReactionInnerData<BodyType, BaseParticlesType, BaseMaterialType>
	{
		/** all diffusion species and diffusion relation. */
		StdVec<BaseDiffusion *> species_diffusion_;
		StdVec<StdLargeVec<Real>> &species_n_;
		StdVec<StdLargeVec<Real>> &diffusion_dt_;
		StdLargeVec<Real> &Vol_;

	protected:
		void initializeDiffusionChangeRate(size_t particle_i);
		void getDiffusionChangeRate(size_t particle_i, size_t particle_j, Vecd &e_ij, Real surface_area_ij);
		virtual void updateSpeciesDiffusion(size_t particle_i, Real dt);
		virtual void Interaction(size_t index_i, Real dt = 0.0) override;
		virtual void Update(size_t index_i, Real dt = 0.0) override;

	public:
		typedef BodyType InnerBodyType;
		typedef BaseParticlesType InnerBaseParticlesType;
		typedef BaseMaterialType InnerBaseMaterialType;
		typedef BaseBodyRelationInner BodyRelationType;
		explicit RelaxationOfAllDiffusionSpeciesInner(BaseBodyRelationInner &inner_relation);
		virtual ~RelaxationOfAllDiffusionSpeciesInner(){};
	};

	/**
	 * @class RelaxationOfAllDiffusionSpeciesComplex
	 * Complex diffusion relaxation between two different bodies
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType,
			  class ContactBodyType, class ContactBaseParticlesType, class ContactBaseMaterialType>
	class RelaxationOfAllDiffusionSpeciesComplex
		: public RelaxationOfAllDiffusionSpeciesInner<BodyType, BaseParticlesType, BaseMaterialType>,
		  public DiffusionReactionContactData<BodyType, BaseParticlesType, BaseMaterialType,
											  ContactBodyType, ContactBaseParticlesType, ContactBaseMaterialType>
	{
		StdVec<BaseDiffusion *> species_diffusion_;
		StdVec<StdLargeVec<Real>> &species_n_;
		StdVec<StdLargeVec<Real>> &diffusion_dt_;
		StdVec<StdLargeVec<Real> *> contact_Vol_;
		StdVec<StdVec<StdLargeVec<Real>> *> contact_species_n_;

	protected:
		void getDiffusionChangeRateContact(size_t particle_i, size_t particle_j, Vecd &e_ij,
										   Real surface_area_ij, const StdVec<StdLargeVec<Real>> &species_n_k);
		virtual void Interaction(size_t index_i, Real dt = 0.0) override;

	public:
		typedef ComplexBodyRelation BodyRelationType;
		explicit RelaxationOfAllDiffusionSpeciesComplex(ComplexBodyRelation &complex_relation);
		virtual ~RelaxationOfAllDiffusionSpeciesComplex(){};
	};

	/**
	 * @class InitializationRK
	 * @brief initialization of a runge-kutta integration scheme
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class InitializationRK
		: public ParticleDynamicsSimple,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
		StdVec<BaseDiffusion *> species_diffusion_;
		StdVec<StdLargeVec<Real>> &species_n_, &species_s_;

		void initializeIntermediateValue(size_t particle_i);
		virtual void Update(size_t index_i, Real dt = 0.0) override;

	public:
		InitializationRK(SPHBody &sph_body, StdVec<StdLargeVec<Real>> &species_s);
		virtual ~InitializationRK(){};
	};

	/**
	 * @class SecondStageRK2
	 * @brief the second stage of the 2nd-order Runge-Kutta scheme
	 */
	template <class FirstStageType>
	class SecondStageRK2 : public FirstStageType
	{
		StdVec<BaseDiffusion *> species_diffusion_;
		StdVec<StdLargeVec<Real>> &species_n_;
		StdVec<StdLargeVec<Real>> &diffusion_dt_;

	protected:
		StdVec<StdLargeVec<Real>> &species_s_;
		virtual void updateSpeciesDiffusion(size_t particle_i, Real dt) override;

	public:
		SecondStageRK2(typename FirstStageType::BodyRelationType &body_relation,
					   StdVec<StdLargeVec<Real>> &species_s);
		virtual ~SecondStageRK2(){};
	};

	/**
	 * @class RelaxationOfAllDiffusionSpeciesRK2
	 * @brief Compute the diffusion relaxation process of all species
	 * with second order Runge-Kutta time stepping
	 */
	template <class FirstStageType>
	class RelaxationOfAllDiffusionSpeciesRK2 : public ParticleDynamics<void>
	{
	protected:
		StdVec<BaseDiffusion *> species_diffusion_;
		/** Intermediate Value */
		StdVec<StdLargeVec<Real>> species_s_;

		InitializationRK<typename FirstStageType::InnerBodyType,
						 typename FirstStageType::InnerBaseParticlesType,
						 typename FirstStageType::InnerBaseMaterialType>
			rk2_initialization_;
		FirstStageType rk2_1st_stage_;
		SecondStageRK2<FirstStageType> rk2_2nd_stage_;

	public:
		explicit RelaxationOfAllDiffusionSpeciesRK2(typename FirstStageType::BodyRelationType &body_relation);
		virtual ~RelaxationOfAllDiffusionSpeciesRK2(){};

		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;
	};

	struct UpdateAReactionSpecies
	{
		Real operator()(Real input, Real production_rate, Real loss_rate, Real dt) const
		{
			return input * exp(-loss_rate * dt) + production_rate * (1.0 - exp(-loss_rate * dt)) / (loss_rate + TinyReal);
		};
	};

	/**
	 * @class RelaxationOfAllReactionsForward
	 * @brief Compute the reaction process of all species by forward splitting
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class RelaxationOfAllReactionsForward
		: public ParticleDynamicsSimple,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
		BaseReactionModel *species_reaction_;
		StdVec<StdLargeVec<Real>> &species_n_;
		UpdateAReactionSpecies updateAReactionSpecies;

	protected:
		virtual void Update(size_t index_i, Real dt = 0.0) override;

	public:
		explicit RelaxationOfAllReactionsForward(BodyType &body);
		virtual ~RelaxationOfAllReactionsForward(){};
	};

	/**
	 * @class RelaxationOfAllReactionsBackward
	 * @brief Compute the reaction process of all species by backward splitting
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class RelaxationOfAllReactionsBackward
		: public ParticleDynamicsSimple,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
		BaseReactionModel *species_reaction_;
		StdVec<StdLargeVec<Real>> &species_n_;
		UpdateAReactionSpecies updateAReactionSpecies;

	protected:
		virtual void Update(size_t index_i, Real dt = 0.0) override;

	public:
		explicit RelaxationOfAllReactionsBackward(BodyType &body);
		virtual ~RelaxationOfAllReactionsBackward(){};
	};

	/**
	 * @class ConstrainDiffusionBodyRegion
	 * @brief set boundary condition for diffusion problem
	 */
	template <class BodyType, class BaseParticlesType, class BodyPartByParticleType, class BaseMaterialType>
	class ConstrainDiffusionBodyRegion
		: public PartSimpleDynamicsByParticle,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
	public:
		ConstrainDiffusionBodyRegion(BodyType &body, BodyPartByParticleType &body_part)
			: PartSimpleDynamicsByParticle(body, body_part),
			  DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>(body),
			  pos_(this->particles_->pos_), species_n_(this->particles_->species_n_){};
		virtual ~ConstrainDiffusionBodyRegion(){};

	protected:
		StdLargeVec<Vecd> &pos_;
		StdVec<StdLargeVec<Real>> &species_n_;
	};

	/**
	 * @class DiffusionBasedMapping
	 * @brief Mapping inside of body according to diffusion.
	 * This is a abstract class to be override for case specific implementation
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class DiffusionBasedMapping
		: public ParticleDynamicsSimple,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
	public:
		explicit DiffusionBasedMapping(BodyType &body)
			: ParticleDynamicsSimple(body),
			  DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>(body),
			  pos_(this->particles_->pos_), species_n_(this->particles_->species_n_){};
		virtual ~DiffusionBasedMapping(){};

	protected:
		StdLargeVec<Vecd> &pos_;
		StdVec<StdLargeVec<Real>> &species_n_;
	};

	/**
	 * @class TotalAveragedParameterOnDiffusionBody
	 * @brief Computing the total averaged parameter on the whole diffusion body.
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class TotalAveragedParameterOnDiffusionBody
		: public ParticleDynamicsReduce<Real, ReduceSum<Real>>,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
	public:
		explicit TotalAveragedParameterOnDiffusionBody(BodyType &body, const std::string &species_name)
			: ParticleDynamicsReduce<Real, ReduceSum<Real>>(body),
			  DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>(body),
			  species_n_(this->particles_->species_n_), species_name_(species_name)
		{
			quantity_name_ = "TotalAveragedParameterOnDiffusionBody";
			initial_reference_ = Real(0);
			phi_ = this->material_->SpeciesIndexMap()[species_name_];
		}
		virtual ~TotalAveragedParameterOnDiffusionBody(){};

	protected:
		StdVec<StdLargeVec<Real>> &species_n_;
		std::string species_name_;
		size_t phi_;
		Real ReduceFunction(size_t index_i, Real dt = 0.0) override
		{
			return species_n_[phi_][index_i] / this->base_particles_->total_real_particles_;
		}
	};

	/**
	 * @class TotalAveragedParameterOnPartlyDiffusionBody
	 * @brief Computing the total averaged parameter on partly diffusion body.
	 */
	template <class BodyType, class BaseParticlesType, class BaseMaterialType>
	class TotalAveragedParameterOnPartlyDiffusionBody
		: public PartDynamicsByParticleReduce<Real, ReduceSum<Real>>,
		  public DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>
	{
	public:
		explicit TotalAveragedParameterOnPartlyDiffusionBody(BodyType &body,
															 BodyPartByParticle &body_part, const std::string &species_name)
			: PartDynamicsByParticleReduce<Real, ReduceSum<Real>>(body, body_part),
			  DiffusionReactionSimpleData<BodyType, BaseParticlesType, BaseMaterialType>(body),
			  species_n_(this->particles_->species_n_), species_name_(species_name)
		{
			quantity_name_ = "TotalAveragedParameterOnPartlyDiffusionBody";
			initial_reference_ = Real(0);
			phi_ = this->material_->SpeciesIndexMap()[species_name_];
		};
		virtual ~TotalAveragedParameterOnPartlyDiffusionBody(){};

	protected:
		StdVec<StdLargeVec<Real>> &species_n_;
		std::string species_name_;
		size_t phi_;
		Real ReduceFunction(size_t index_i, Real dt = 0.0) override
		{
			return species_n_[phi_][index_i] / body_part_particles_.size();
		}
	};
}
#endif // PARTICLE_DYNAMICS_DIFFUSION_REACTION_H