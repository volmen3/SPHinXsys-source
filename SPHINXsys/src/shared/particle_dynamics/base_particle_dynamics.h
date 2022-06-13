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
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1                *
 * and HU1527/12-1.                                                             *
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
 * @file base_particle_dynamics.h
 * @brief This is for the base classes of particle dynamics, which describe the
 * interaction between particles. These interactions are used to define
 * differential operators for surface forces or fluxes in continuum mechanics
 * @author  Xiangyu Hu, Luhui Han and Chi Zhang
 */

#ifndef BASE_PARTICLE_DYNAMICS_H
#define BASE_PARTICLE_DYNAMICS_H

#include "base_data_package.h"
#include "sph_data_containers.h"
#include "functors_iterators.hpp"

#include "neighbor_relation.h"
#include "body_relation.h"
#include "base_body.h"

#include <functional>

using namespace std::placeholders;

namespace SPH
{
	/**
	 * @class GlobalStaticVariables
	 * @brief A place to put all global variables
	 */
	class GlobalStaticVariables
	{
	public:
		explicit GlobalStaticVariables(){};
		virtual ~GlobalStaticVariables(){};

		/** the physical time is global value for all dynamics */
		static Real physical_time_;
	};

	/**
	 * @class BaseParticleDynamics
	 * @brief The new version of base class for all particle dynamics
	 * This class contains the only two interface functions available
	 * for particle dynamics. An specific implementation should be realized.
	 */
	template <class ReturnType>
	class BaseParticleDynamics : public GlobalStaticVariables
	{
	public:
		explicit BaseParticleDynamics(){};
		virtual ~BaseParticleDynamics(){};

		/** The only two functions can be called from outside
		 * One is for sequential execution, the other is for parallel. */
		virtual ReturnType exec(Real dt = 0.0) = 0;
		virtual ReturnType parallel_exec(Real dt = 0.0) = 0;
	};

	/**
	 * @class ParticleDynamics
	 * @brief The basic particle dynamics in which a range of particles are looped.
	 */
	template <typename LoopRange, typename FunctorType = ParticleFunctor>
	class ParticleDynamics : public BaseParticleDynamics<void>
	{
	public:
		ParticleDynamics(LoopRange &loop_range, FunctorType functor)
			: BaseParticleDynamics<void>(),
			  loop_range_(loop_range), functor_(functor){};
		virtual ~ParticleDynamics(){};

		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;

	protected:
		LoopRange &loop_range_;
		FunctorType functor_;
	};

	/**
	 * @class ParticleDynamicsReduce
	 * @brief Base abstract class for reduce
	 */
	template <class LocalReduceType, typename LoopRange,
			  template <typename ReduceFunctorReturnType> typename ReduceFunctorType>
	class ParticleDynamicsReduce : public LocalReduceType,
								   public BaseParticleDynamics<
									   typename LocalReduceType::ReduceReturnType>
	{
		using ReturnType = typename LocalReduceType::ReduceReturnType;

	public:
		template <typename... Args>
		ParticleDynamicsReduce(LoopRange &loop_range, Args &&...args)
			: LocalReduceType(std::forward<Args>(args)...),
			  BaseParticleDynamics<ReturnType>(), loop_range_(loop_range){};
		virtual ~ParticleDynamicsReduce(){};

		virtual ReturnType exec(Real dt = 0.0) override
		{
			LocalReduceType::setupDynamics();
			ReturnType temp = particle_reduce(
				loop_range_, LocalReduceType::reference_, functor_, LocalReduceType::operation_, dt);
			return LocalReduceType::outputResult(temp);
		};
		virtual ReturnType parallel_exec(Real dt = 0.0) override
		{
			LocalReduceType::setupDynamics();
			ReturnType temp = particle_parallel_reduce(
				loop_range_, LocalReduceType::reference_, functor_, LocalReduceType::operation_, dt);
			return LocalReduceType::outputResult(temp);
		};

	protected:
		LoopRange &loop_range_;
		ReduceFunctorType<ReturnType> functor_;
	};
	/**	/**
	 * @class BaseSimpleDynamics
	 * @brief Simple particle dynamics without considering particle interaction
	 */
	template <typename LoopRange, typename FunctorType>
	class BaseSimpleDynamics : public BaseParticleDynamics<void>
	{
		ParticleDynamics<LoopRange, FunctorType> simple_dynamics_;

	public:
		explicit BaseSimpleDynamics(LoopRange &loop_range, FunctorType functor)
			: BaseParticleDynamics<void>(),
			  simple_dynamics_(loop_range, functor){};
		virtual ~BaseSimpleDynamics(){};

		virtual void runSetup(Real dt = 0.0) = 0;
		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;
	};

	/**
	 * @class BaseInteractionDynamics
	 * @brief  This is the class for particle interaction with other particles
	 */
	template <typename LoopRange>
	class BaseInteractionDynamics : public BaseParticleDynamics<void>
	{
		ParticleDynamics<LoopRange, ParticleFunctor> interaction_dynamics_;
		/** pre process such as update ghost state */
		StdVec<BaseParticleDynamics<void> *> pre_processes_;
		/** post process such as impose constraint */
		StdVec<BaseParticleDynamics<void> *> post_processes_;

	public:
		BaseInteractionDynamics(LoopRange &loop_range, ParticleFunctor functor_interaction)
			: BaseParticleDynamics<void>(),
			  interaction_dynamics_(loop_range, functor_interaction){};

		virtual ~BaseInteractionDynamics(){};

		void addPreProcess(BaseParticleDynamics<void> *pre_process) { pre_processes_.push_back(pre_process); };
		void addPostProcess(BaseParticleDynamics<void> *post_process) { post_processes_.push_back(post_process); };

		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;

		virtual void runSetup(Real dt = 0.0) = 0;
		void runInteraction(Real dt = 0.0);
		void runInteraction_parallel(Real dt = 0.0);
	};

	/**
	 * @class BaseInteractionDynamicsAndUpdate
	 * @brief This class includes an interaction and a update steps
	 */
	template <typename LoopRange, typename FunctorType>
	class BaseInteractionDynamicsAndUpdate : public BaseInteractionDynamics<LoopRange>
	{
		ParticleDynamics<LoopRange, FunctorType> update_dynamics_;

	public:
		BaseInteractionDynamicsAndUpdate(
			LoopRange &loop_range, ParticleFunctor functor_interaction,
			FunctorType functor_update)
			: BaseInteractionDynamics<LoopRange>(loop_range, functor_interaction),
			  update_dynamics_(loop_range, functor_update){};
		virtual ~BaseInteractionDynamicsAndUpdate(){};

		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;

		void runUpdate(Real dt = 0.0);
		void runUpdate_parallel(Real dt = 0.0);
	};

	/**
	 * @class BaseInteractionDynamics1Level
	 * @brief This class includes an initialization, an interaction and a update steps
	 */
	template <typename LoopRange, typename FunctorType>
	class BaseInteractionDynamics1Level : public BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>
	{
		ParticleDynamics<LoopRange, FunctorType> initialize_dynamics_;

	public:
		BaseInteractionDynamics1Level(LoopRange &loop_range, FunctorType functor_initialization,
									  ParticleFunctor functor_interaction, FunctorType functor_update)
			: BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>(loop_range, functor_interaction, functor_update),
			  initialize_dynamics_(loop_range, functor_initialization){};
		virtual ~BaseInteractionDynamics1Level(){};

		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;

		void runInitialization(Real dt = 0.0);
		void runInitialization_parallel(Real dt = 0.0);
	};

	/**
	 * @class BaseLocalParticleDynamics
	 * @brief The new version of base class for all local particle dynamics.
	 */
	template <class ReturnType>
	class BaseLocalParticleDynamics
	{
		SPHBody *sph_body_;

	public:
		explicit BaseLocalParticleDynamics(SPHBody &sph_body) : sph_body_(&sph_body){};
		virtual ~BaseLocalParticleDynamics(){};

		typedef ReturnType DynamicsParameterType;
		void setBodyUpdated() { sph_body_->setNewlyUpdated(); };
		/** the function for set global parameters for the particle dynamics */
		virtual ReturnType setupDynamics(Real dt = 0.0) = 0;
	};

	/**
	 * @class LocalParticleDynamics
	 * @brief The new version of base class for all local particle dynamics,
	 * which loops along particles.
	 */
	class LocalParticleDynamics : public BaseLocalParticleDynamics<void>
	{
	public:
		explicit LocalParticleDynamics(SPHBody &sph_body)
			: BaseLocalParticleDynamics<void>(sph_body){};
		virtual ~LocalParticleDynamics(){};

		/** the function for set global parameters for the particle dynamics */
		virtual void setupDynamics(Real dt = 0.0) override{};
	};

	/**
	 * @class LocalParticleDynamicsReduce
	 * @brief The new version of base class for all local particle dynamics.
	 */
	template <typename ReturnType, typename ReduceOperation>
	class LocalParticleDynamicsReduce : public LocalParticleDynamics
	{
	public:
		explicit LocalParticleDynamicsReduce(SPHBody &sph_body, ReturnType reference)
			: LocalParticleDynamics(sph_body), reference_(reference),
			  quantity_name_("ReducedQuantity"){};
		virtual ~LocalParticleDynamicsReduce(){};

		using ReduceReturnType = ReturnType;
		ReturnType InitialReference() { return reference_; };
		std::string QuantityName() { return quantity_name_; };
		ReduceOperation &getReduceOperation() { return operation_; };
		virtual ReturnType outputResult(ReturnType reduced_value) { return reduced_value; }

	protected:
		ReturnType reference_;
		ReduceOperation operation_;
		std::string quantity_name_;
	};

	/**
	 * @class OldParticleDynamics
	 * @brief The base class for all particle dynamics
	 * This class contains the only two interface functions available
	 * for particle dynamics. An specific implementation should be realized.
	 */
	template <class ReturnType = void>
	class OldParticleDynamics : public GlobalStaticVariables
	{
	public:
		explicit OldParticleDynamics(SPHBody &sph_body);
		virtual ~OldParticleDynamics(){};

		SPHBody *getSPHBody() { return sph_body_; };
		/** The only two functions can be called from outside
		 * One is for sequential execution, the other is for parallel. */
		virtual ReturnType exec(Real dt = 0.0) = 0;
		virtual ReturnType parallel_exec(Real dt = 0.0) = 0;

	protected:
		SPHBody *sph_body_;
		SPHAdaptation *sph_adaptation_;
		BaseParticles *base_particles_;

		void setBodyUpdated() { sph_body_->setNewlyUpdated(); };
		/** the function for set global parameters for the particle dynamics */
		virtual void setupDynamics(Real dt = 0.0){};
	};

	/**
	 * @class DataDelegateBase
	 * @brief empty base class mixin template.
	 */
	class DataDelegateEmptyBase
	{
	public:
		explicit DataDelegateEmptyBase(SPHBody &sph_body){};
		virtual ~DataDelegateEmptyBase(){};
	};

	/**
	 * @class DataDelegateSimple
	 * @brief prepare data for simple particle dynamics.
	 */
	template <class BodyType = SPHBody,
			  class ParticlesType = BaseParticles,
			  class MaterialType = BaseMaterial>
	class DataDelegateSimple
	{
	public:
		explicit DataDelegateSimple(SPHBody &sph_body)
			: body_(DynamicCast<BodyType>(this, &sph_body)),
			  particles_(DynamicCast<ParticlesType>(this, sph_body.base_particles_)),
			  material_(DynamicCast<MaterialType>(this, sph_body.base_material_)),
			  sorted_id_(sph_body.base_particles_->sorted_id_),
			  unsorted_id_(sph_body.base_particles_->unsorted_id_){};
		virtual ~DataDelegateSimple(){};

		BodyType *getBody() { return body_; };
		ParticlesType *getParticles() { return particles_; };
		MaterialType *getMaterial() { return material_; };

	protected:
		BodyType *body_;
		ParticlesType *particles_;
		MaterialType *material_;
		StdLargeVec<size_t> &sorted_id_;
		StdLargeVec<size_t> &unsorted_id_;
	};

	/**
	 * @class DataDelegateInner
	 * @brief prepare data for inner particle dynamics
	 */
	template <class BodyType = SPHBody,
			  class ParticlesType = BaseParticles,
			  class MaterialType = BaseMaterial,
			  class BaseDataDelegateType = DataDelegateSimple<BodyType, ParticlesType, MaterialType>>
	class DataDelegateInner : public BaseDataDelegateType
	{
	public:
		explicit DataDelegateInner(BaseBodyRelationInner &body_inner_relation)
			: BaseDataDelegateType(*body_inner_relation.sph_body_),
			  inner_configuration_(body_inner_relation.inner_configuration_){};
		virtual ~DataDelegateInner(){};

	protected:
		/** inner configuration of the designated body */
		ParticleConfiguration &inner_configuration_;
	};

	/**
	 * @class DataDelegateContact
	 * @brief prepare data for contact particle dynamics
	 */
	template <class BodyType = SPHBody,
			  class ParticlesType = BaseParticles,
			  class MaterialType = BaseMaterial,
			  class ContactBodyType = SPHBody,
			  class ContactParticlesType = BaseParticles,
			  class ContactMaterialType = BaseMaterial,
			  class BaseDataDelegateType = DataDelegateSimple<BodyType, ParticlesType, MaterialType>>
	class DataDelegateContact : public BaseDataDelegateType
	{
	public:
		explicit DataDelegateContact(BaseBodyRelationContact &body_contact_relation);
		virtual ~DataDelegateContact(){};

	protected:
		StdVec<ContactBodyType *> contact_bodies_;
		StdVec<ContactParticlesType *> contact_particles_;
		StdVec<ContactMaterialType *> contact_material_;
		/** Configurations for particle interaction between bodies. */
		StdVec<ParticleConfiguration *> contact_configuration_;
	};

	/**
	 * @class DataDelegateComplex
	 * @brief prepare data for complex particle dynamics
	 */
	template <class BodyType = SPHBody,
			  class ParticlesType = BaseParticles,
			  class MaterialType = BaseMaterial,
			  class ContactBodyType = SPHBody,
			  class ContactParticlesType = BaseParticles,
			  class ContactMaterialType = BaseMaterial>
	class DataDelegateComplex : public DataDelegateInner<BodyType, ParticlesType, MaterialType>,
								public DataDelegateContact<BodyType, ParticlesType, MaterialType,
														   ContactBodyType, ContactParticlesType, ContactMaterialType, DataDelegateEmptyBase>
	{
	public:
		explicit DataDelegateComplex(ComplexBodyRelation &body_complex_relation)
			: DataDelegateInner<BodyType, ParticlesType, MaterialType>(body_complex_relation.inner_relation_),
			  DataDelegateContact<BodyType, ParticlesType, MaterialType, ContactBodyType, ContactParticlesType,
								  ContactMaterialType, DataDelegateEmptyBase>(body_complex_relation.contact_relation_){};
		virtual ~DataDelegateComplex(){};
	};

	/**
	 * @class ParticleDynamicsComplex
	 * @brief particle dynamics by considering  contribution from extra contact bodies
	 */
	template <class ParticleDynamicsInnerType, class ContactDataType>
	class ParticleDynamicsComplex : public ParticleDynamicsInnerType, public ContactDataType
	{
	public:
		ParticleDynamicsComplex(BaseBodyRelationInner &inner_relation,
								BaseBodyRelationContact &contact_relation)
			: ParticleDynamicsInnerType(inner_relation), ContactDataType(contact_relation){};

		ParticleDynamicsComplex(ComplexBodyRelation &complex_relation,
								BaseBodyRelationContact &extra_contact_relation);

		virtual ~ParticleDynamicsComplex(){};

	protected:
		virtual void prepareContactData() = 0;
	};
}
#endif // BASE_PARTICLE_DYNAMICS_H