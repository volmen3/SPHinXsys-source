/**
 * @file 	particle_dynamics_bodypart.cpp
 * @brief 	This is the implementation of the template class
 * @author	Chi ZHang and Xiangyu Hu
 */

#include "particle_dynamics_bodypart.h"

namespace SPH
{
	//=================================================================================================//
	PartDynamicsByParticle::PartDynamicsByParticle(SPHBody &sph_body, BodyPartByParticle &body_part)
		: OldParticleDynamics<void>(sph_body),
		  body_part_particles_(body_part.body_part_particles_) {}
	//=================================================================================================//
	PartSimpleDynamicsByParticle::
		PartSimpleDynamicsByParticle(SPHBody &sph_body, BodyPartByParticle &body_part)
		: PartDynamicsByParticle(sph_body, body_part),
		  functor_update_(std::bind(&PartSimpleDynamicsByParticle::Update, this, _1, _2)) {}
	//=================================================================================================//
	void PartSimpleDynamicsByParticle::exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_for(body_part_particles_, functor_update_, dt);
	}
	//=================================================================================================//
	void PartSimpleDynamicsByParticle::parallel_exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_parallel_for(body_part_particles_, functor_update_, dt);
	}
	//=================================================================================================//
	PartInteractionDynamicsByParticle::
		PartInteractionDynamicsByParticle(SPHBody &sph_body, BodyPartByParticle &body_part)
		: PartDynamicsByParticle(sph_body, body_part),
		  functor_interaction_(std::bind(&PartInteractionDynamicsByParticle::Interaction, this, _1, _2)) {}
	//=================================================================================================//
	void PartInteractionDynamicsByParticle::exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_for(body_part_particles_, functor_interaction_, dt);
	}
	//=================================================================================================//
	void PartInteractionDynamicsByParticle::parallel_exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_parallel_for(body_part_particles_, functor_interaction_, dt);
	}
	//=================================================================================================//
	PartInteractionDynamicsByParticleWithUpdate::
		PartInteractionDynamicsByParticleWithUpdate(SPHBody &sph_body, BodyPartByParticle &body_part)
		: PartInteractionDynamicsByParticle(sph_body, body_part),
		  functor_update_(std::bind(&PartInteractionDynamicsByParticleWithUpdate::Update, this, _1, _2)) {}
	//=================================================================================================//
	void PartInteractionDynamicsByParticleWithUpdate::exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_for(body_part_particles_, functor_interaction_, dt);
		particle_for(body_part_particles_, functor_update_, dt);
	}
	//=================================================================================================//
	void PartInteractionDynamicsByParticleWithUpdate::parallel_exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_parallel_for(body_part_particles_, functor_interaction_, dt);
		particle_parallel_for(body_part_particles_, functor_update_, dt);
	}
	//=================================================================================================//
	PartInteractionDynamicsByParticle1Level::
		PartInteractionDynamicsByParticle1Level(SPHBody &sph_body, BodyPartByParticle &body_part)
		: PartInteractionDynamicsByParticleWithUpdate(sph_body, body_part),
		  functor_initialization_(
			  std::bind(&PartInteractionDynamicsByParticle1Level::Initialization, this, _1, _2)) {}
	//=================================================================================================//
	void PartInteractionDynamicsByParticle1Level::exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_for(body_part_particles_, functor_initialization_, dt);
		particle_for(body_part_particles_, functor_interaction_, dt);
		particle_for(body_part_particles_, functor_update_, dt);
	}
	//=================================================================================================//
	void PartInteractionDynamicsByParticle1Level::parallel_exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_parallel_for(body_part_particles_, functor_initialization_, dt);
		particle_parallel_for(body_part_particles_, functor_interaction_, dt);
		particle_parallel_for(body_part_particles_, functor_update_, dt);
	}
	//=================================================================================================//
	PartDynamicsByCell::PartDynamicsByCell(SPHBody &sph_body, BodyPartByCell &body_part)
		: OldParticleDynamics<void>(sph_body),
		  body_part_cells_(body_part.body_part_cells_),
		  functor_update_(std::bind(&PartDynamicsByCell::Update, this, _1, _2)){};
	//=================================================================================================//
	void PartDynamicsByCell::exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_for(body_part_cells_, functor_update_, dt);
	}
	//=================================================================================================//
	void PartDynamicsByCell::parallel_exec(Real dt)
	{
		setBodyUpdated();
		setupDynamics(dt);
		particle_parallel_for(body_part_cells_, functor_update_, dt);
	}
	//=================================================================================================//
}
