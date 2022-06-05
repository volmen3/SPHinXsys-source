/**
 * @file 	base_particle_dynamics.hpp
 * @brief 	This is the implementation of the template class for 3D build
 * @author	Chi ZHang and Xiangyu Hu
 */

#ifndef BASE_PARTICLE_DYNAMICS_HPP
#define BASE_PARTICLE_DYNAMICS_HPP

#include "base_particle_dynamics.h"
#include "base_body.h"

namespace SPH
{
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void ParticleDynamics<LoopRange, FunctorType>::exec(Real dt)
	{
		particle_for(loop_range_, functor_, dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void ParticleDynamics<LoopRange, FunctorType>::parallel_exec(Real dt)
	{
		particle_parallel_for(loop_range_, functor_, dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename ReturnType, typename ReduceOperation,
			  template <typename ReduceReturnType> typename ReduceFunctorType>
	ReturnType ParticleDynamicsReduce<LoopRange, ReturnType, ReduceOperation,
									  ReduceFunctorType>::exec(Real dt)
	{
		return particle_reduce(loop_range_, reference_, functor_, operation_, dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename ReturnType, typename ReduceOperation,
			  template <typename ReduceReturnType> typename ReduceFunctorType>
	ReturnType ParticleDynamicsReduce<LoopRange, ReturnType, ReduceOperation,
									  ReduceFunctorType>::parallel_exec(Real dt)
	{
		return particle_parallel_reduce(loop_range_, reference_, functor_, operation_, dt);
	}
	//=================================================================================================//
	template <typename LoopRange>
	void BaseInteractionDynamics<LoopRange>::exec(Real dt)
	{
		runSetup(dt);
		runInteraction(dt);
	}
	//=================================================================================================//
	template <typename LoopRange>
	void BaseInteractionDynamics<LoopRange>::parallel_exec(Real dt)
	{
		runSetup(dt);
		runInteraction_parallel(dt);
	}
	//=================================================================================================//
	template <typename LoopRange>
	void BaseInteractionDynamics<LoopRange>::runInteraction(Real dt)
	{
		for (size_t k = 0; k < pre_processes_.size(); ++k)
			pre_processes_[k]->exec(dt);
		interaction_dynamics_.exec(dt);
		for (size_t k = 0; k < post_processes_.size(); ++k)
			post_processes_[k]->exec(dt);
	}
	//=================================================================================================//
	template <typename LoopRange>
	void BaseInteractionDynamics<LoopRange>::runInteraction_parallel(Real dt)
	{
		for (size_t k = 0; k < pre_processes_.size(); ++k)
			pre_processes_[k]->parallel_exec(dt);
		interaction_dynamics_.parallel_exec(dt);
		for (size_t k = 0; k < post_processes_.size(); ++k)
			post_processes_[k]->parallel_exec(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>::exec(Real dt)
	{
		this->runSetup(dt);
		BaseInteractionDynamics<LoopRange>::runInteraction(dt);
		runUpdate(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>::parallel_exec(Real dt)
	{
		this->runSetup(dt);
		BaseInteractionDynamics<LoopRange>::runInteraction_parallel(dt);
		runUpdate_parallel(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>::runUpdate(Real dt)
	{
		update_dynamics_.exec(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>::runUpdate_parallel(Real dt)
	{
		update_dynamics_.parallel_exec(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamics1Level<LoopRange, FunctorType>::exec(Real dt)
	{
		this->runSetup(dt);
		runInitialization(dt);
		BaseInteractionDynamics<LoopRange>::runInteraction(dt);
		BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>::runUpdate(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamics1Level<LoopRange, FunctorType>::parallel_exec(Real dt)
	{
		this->runSetup(dt);
		runInitialization_parallel(dt);
		BaseInteractionDynamics<LoopRange>::runInteraction_parallel(dt);
		BaseInteractionDynamicsAndUpdate<LoopRange, FunctorType>::runUpdate_parallel(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamics1Level<LoopRange, FunctorType>::runInitialization(Real dt)
	{
		initialize_dynamics_.exec(dt);
	}
	//=================================================================================================//
	template <typename LoopRange, typename FunctorType>
	void BaseInteractionDynamics1Level<LoopRange, FunctorType>::runInitialization_parallel(Real dt)
	{
		initialize_dynamics_.parallel_exec(dt);
	}
	//=================================================================================================//
	template <class ReturnType>
	OldParticleDynamics<ReturnType>::OldParticleDynamics(SPHBody &sph_body)
		: GlobalStaticVariables(), sph_body_(&sph_body),
		  sph_adaptation_(sph_body.sph_adaptation_),
		  base_particles_(sph_body.base_particles_) {}
	//=================================================================================================//
	template <class BodyType,
			  class ParticlesType,
			  class MaterialType,
			  class ContactBodyType,
			  class ContactParticlesType,
			  class ContactMaterialType,
			  class BaseDataDelegateType>
	DataDelegateContact<BodyType, ParticlesType, MaterialType, ContactBodyType, ContactParticlesType, ContactMaterialType, BaseDataDelegateType>::
		DataDelegateContact(BaseBodyRelationContact &body_contact_relation) : BaseDataDelegateType(*body_contact_relation.sph_body_)
	{
		RealBodyVector contact_sph_bodies = body_contact_relation.contact_bodies_;
		for (size_t i = 0; i != contact_sph_bodies.size(); ++i)
		{
			contact_bodies_.push_back(DynamicCast<ContactBodyType>(this, contact_sph_bodies[i]));
			contact_particles_.push_back(DynamicCast<ContactParticlesType>(this, contact_sph_bodies[i]->base_particles_));
			contact_material_.push_back(DynamicCast<ContactMaterialType>(this, contact_sph_bodies[i]->base_material_));
			contact_configuration_.push_back(&body_contact_relation.contact_configuration_[i]);
		}
	}
	//=================================================================================================//
	template <class ParticleDynamicsInnerType, class ContactDataType>
	ParticleDynamicsComplex<ParticleDynamicsInnerType, ContactDataType>::
		ParticleDynamicsComplex(ComplexBodyRelation &complex_relation,
								BaseBodyRelationContact &extra_contact_relation)
		: ParticleDynamicsInnerType(complex_relation.inner_relation_),
		  ContactDataType(complex_relation.contact_relation_)
	{
		if (complex_relation.sph_body_ != extra_contact_relation.sph_body_)
		{
			std::cout << "\n Error: the two body_realtions do not have the same source body!" << std::endl;
			std::cout << __FILE__ << ':' << __LINE__ << std::endl;
			exit(1);
		}

		for (auto &extra_body : extra_contact_relation.contact_bodies_)
		{
			// here we first obtain the pointer to the most derived class and then implicitly downcast it to
			// the types defined in the base complex dynamics
			this->contact_bodies_.push_back(extra_body->ThisObjectPtr());
			this->contact_particles_.push_back(extra_body->base_particles_->ThisObjectPtr());
			this->contact_material_.push_back(extra_body->base_material_->ThisObjectPtr());
		}

		for (size_t i = 0; i != extra_contact_relation.contact_bodies_.size(); ++i)
		{
			this->contact_configuration_.push_back(&extra_contact_relation.contact_configuration_[i]);
		}
	}
	//=================================================================================================//
}
#endif // BASE_PARTICLE_DYNAMICS_HPP
