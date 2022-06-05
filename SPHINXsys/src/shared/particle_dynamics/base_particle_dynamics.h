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
	template <class ReturnType = void>
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
	template <typename LoopRange, typename ParticleFunctorType = ParticleFunctor>
	class ParticleDynamics : public BaseParticleDynamics<void>
	{
	public:
		ParticleDynamics(LoopRange &loop_range, ParticleFunctorType particle_functor)
			: BaseParticleDynamics<void>(),
			  loop_range_(loop_range), particle_functor_(particle_functor){};

		virtual ~ParticleDynamics(){};

		virtual void exec(Real dt = 0.0) override
		{
			ParticleIterator(loop_range_, particle_functor_, dt);
		};

		virtual void parallel_exec(Real dt = 0.0) override
		{
			ParticleIterator_parallel(loop_range_, particle_functor_, dt);
		};

	protected:
		LoopRange &loop_range_;
		ParticleFunctorType particle_functor_;
	};

	/**
	 * @class ParticleDynamicsReduce
	 * @brief Base abstract class for reduce
	 */
	template <typename LoopRange, typename ReturnType, typename ReduceOperation,
			  template <typename ReduceReturnType> typename ReduceFunctorType>
	class ParticleDynamicsReduce : public BaseParticleDynamics<ReturnType>
	{
		LoopRange &loop_range_;
		ReturnType &initial_reference_;
		ReduceOperation &reduce_operation_;
		ReduceFunctorType<ReturnType> functor_reduce_;

	public:
		explicit ParticleDynamicsReduce(LoopRange &loop_range, ReturnType &initial_reference,
										ReduceOperation &reduce_operation,
										ReduceFunctorType<ReturnType> functor_reduce)
			: BaseParticleDynamics<ReturnType>(), loop_range_(loop_range),
			  initial_reference_(initial_reference), reduce_operation_(reduce_operation),
			  functor_reduce_(functor_reduce){};
		virtual ~ParticleDynamicsReduce(){};

		virtual ReturnType exec(Real dt = 0.0) override
		{
			return ReduceIterator(loop_range_, initial_reference_, functor_reduce_, reduce_operation_, dt);
		};
		virtual ReturnType parallel_exec(Real dt = 0.0) override
		{
			return ReduceIterator_parallel(loop_range_, initial_reference_, functor_reduce_, reduce_operation_, dt);
		};
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
			: BaseParticleDynamics<void>(), interaction_dynamics_(loop_range, functor_interaction){};

		virtual ~BaseInteractionDynamics(){};

		void addPreProcess(BaseParticleDynamics<void> *pre_process) { pre_processes_.push_back(pre_process); };
		void addPostProcess(BaseParticleDynamics<void> *post_process) { post_processes_.push_back(post_process); };

		virtual void exec(Real dt = 0.0) override
		{
			runSetup(dt);
			runInteraction(dt);
		};

		virtual void parallel_exec(Real dt = 0.0) override
		{
			runSetup(dt);
			runInteraction_parallel(dt);
		};

		virtual void runSetup(Real dt = 0.0) = 0;

		void runInteraction(Real dt = 0.0)
		{
			for (size_t k = 0; k < pre_processes_.size(); ++k)
				pre_processes_[k]->exec(dt);
			interaction_dynamics_.exec(dt);
			for (size_t k = 0; k < post_processes_.size(); ++k)
				post_processes_[k]->exec(dt);
		};

		void runInteraction_parallel(Real dt = 0.0)
		{
			for (size_t k = 0; k < pre_processes_.size(); ++k)
				pre_processes_[k]->parallel_exec(dt);
			interaction_dynamics_.parallel_exec(dt);
			for (size_t k = 0; k < post_processes_.size(); ++k)
				post_processes_[k]->parallel_exec(dt);
		};
	};

	/**
	 * @class BaseInteractionDynamicsWithUpdate
	 * @brief This class includes an interaction and a update steps
	 */
	template <typename LoopRange, typename ParticleFunctorType>
	class BaseInteractionDynamicsWithUpdate : public BaseInteractionDynamics<LoopRange>
	{
		ParticleDynamics<LoopRange, ParticleFunctorType> update_dynamics_;

	public:
		BaseInteractionDynamicsWithUpdate(LoopRange &loop_range, ParticleFunctor functor_interaction,
										  ParticleFunctorType functor_update)
			: BaseInteractionDynamics<LoopRange>(loop_range, functor_interaction),
			  update_dynamics_(loop_range, functor_update){};
		virtual ~BaseInteractionDynamicsWithUpdate(){};

		virtual void exec(Real dt = 0.0) override
		{
			this->runSetup(dt);
			BaseInteractionDynamics<LoopRange>::runInteraction(dt);
			runUpdate(dt);
		};

		virtual void parallel_exec(Real dt = 0.0) override
		{
			this->runSetup(dt);
			BaseInteractionDynamics<LoopRange>::runInteraction_parallel(dt);
			runUpdate_parallel(dt);
		};

		void runUpdate(Real dt = 0.0)
		{
			update_dynamics_.exec(dt);
		};

		void runUpdate_parallel(Real dt = 0.0)
		{
			update_dynamics_.parallel_exec(dt);
		};
	};

	/**
	 * @class BaseInteractionDynamics1Level
	 * @brief This class includes an initialization, an interaction and a update steps
	 */
	template <typename LoopRange, typename ParticleFunctorType>
	class BaseInteractionDynamics1Level : public BaseInteractionDynamicsWithUpdate<LoopRange, ParticleFunctorType>
	{
		ParticleDynamics<LoopRange, ParticleFunctorType> initialize_dynamics_;

	public:
		BaseInteractionDynamics1Level(LoopRange &loop_range, ParticleFunctorType functor_initialization,
									  ParticleFunctor functor_interaction, ParticleFunctorType functor_update)
			: BaseInteractionDynamicsWithUpdate<LoopRange, ParticleFunctorType>(loop_range, functor_interaction, functor_update),
			  initialize_dynamics_(loop_range, functor_initialization){};
		virtual ~BaseInteractionDynamics1Level(){};

		virtual void exec(Real dt = 0.0) override
		{
			this->runSetup(dt);
			runInitialization(dt);
			BaseInteractionDynamics<LoopRange>::runInteraction(dt);
			BaseInteractionDynamicsWithUpdate<LoopRange, ParticleFunctorType>::runUpdate(dt);
		};

		virtual void parallel_exec(Real dt = 0.0) override
		{
			this->runSetup(dt);
			runInitialization_parallel(dt);
			BaseInteractionDynamics<LoopRange>::runInteraction_parallel(dt);
			BaseInteractionDynamicsWithUpdate<LoopRange, ParticleFunctorType>::runUpdate_parallel(dt);
		};

		void runInitialization(Real dt = 0.0)
		{
			initialize_dynamics_.exec(dt);
		};

		void runInitialization_parallel(Real dt = 0.0)
		{
			initialize_dynamics_.parallel_exec(dt);
		};
	};

	/**
	 * @class LocalParticleDynamics
	 * @brief The new version of base class for all local particle dynamics.
	 */
	class LocalParticleDynamics
	{
		SPHBody *sph_body_;

	public:
		explicit LocalParticleDynamics(SPHBody &sph_body) : sph_body_(&sph_body){};
		virtual ~LocalParticleDynamics(){};

		void setBodyUpdated() { sph_body_->setNewlyUpdated(); };
		/** the function for set global parameters for the particle dynamics */
		virtual void setupDynamics(Real dt = 0.0){};
	};

	/**
	 * @class LocalParticleDynamicsReduce
	 * @brief The new version of base class for all local particle dynamics.
	 */
	template <typename ReturnType, typename ReduceOperation>
	class LocalParticleDynamicsReduce : public LocalParticleDynamics
	{
	public:
		explicit LocalParticleDynamicsReduce(SPHBody &sph_body, ReturnType initial_reference)
			: LocalParticleDynamics(sph_body), initial_reference_(initial_reference){};
		virtual ~LocalParticleDynamicsReduce(){};

		ReturnType initial_reference_;
		ReduceOperation reduce_operation_;
		virtual ReturnType outputResult(ReturnType reduced_value) { return reduced_value; }
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