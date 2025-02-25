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
 * @file 	diffusion_reaction_particles.h
 * @brief 	This is the derived class of diffusion reaction particles.
 * @author	Xiangyu Hu and Chi Zhang
 */

#ifndef DIFFUSION_REACTION_PARTICLES_H
#define DIFFUSION_REACTION_PARTICLES_H

#include "base_particles.h"
#include "base_body.h"
#include "base_material.h"
#include "diffusion_reaction.h"

namespace SPH
{

	/**
	 * @class DiffusionReactionParticles
	 * @brief A group of particles with diffusion or/and reactions particle data.
	 */
	template <class BaseParticlesType = BaseParticles>
	class DiffusionReactionParticles : public BaseParticlesType
	{
	protected:
		size_t number_of_species_;			 /**< Total number of diffusion and reaction species . */
		size_t number_of_diffusion_species_; /**< Total number of diffusion species . */
		std::map<std::string, size_t> species_indexes_map_;

	public:
		StdVec<StdLargeVec<Real>> species_n_;	 /**< array of diffusion/reaction scalars */
		StdVec<StdLargeVec<Real>> diffusion_dt_; /**< array of the time derivative of diffusion species */

		template <class BaseMaterialType>
		DiffusionReactionParticles(SPHBody &sph_body,
								   DiffusionReaction<BaseMaterialType> *diffusion_reaction_material)
			: BaseParticlesType(sph_body, diffusion_reaction_material),
			  number_of_species_(diffusion_reaction_material->NumberOfSpecies()),
			  number_of_diffusion_species_(diffusion_reaction_material->NumberOfSpeciesDiffusion()),
			  species_indexes_map_(diffusion_reaction_material->SpeciesIndexMap())
		{
			species_n_.resize(number_of_species_);
			diffusion_dt_.resize(number_of_diffusion_species_);
		};
		virtual ~DiffusionReactionParticles(){};

		std::map<std::string, size_t> SpeciesIndexMap() { return species_indexes_map_; };

		virtual void initializeOtherVariables() override
		{
			BaseParticlesType::initializeOtherVariables();
			
			std::map<std::string, size_t>::iterator itr;
			for (itr = species_indexes_map_.begin(); itr != species_indexes_map_.end(); ++itr)
			{
				// Register a specie. 
				this->registerVariable(species_n_[itr->second], itr->first);
				// the scalars will be sorted if particle sorting is called
				// Note that we call a template function from a template class
				this->template registerSortableVariable<Real>(itr->first);
				// add species to basic output particle data
				this->template addVariableToWrite<Real>(itr->first);
			}

			for (size_t m = 0; m < number_of_diffusion_species_; ++m)
			{
				constexpr int type_index = DataTypeIndex<Real>::value;
				//----------------------------------------------------------------------
				//	register reactive change rate terms without giving variable name
				//----------------------------------------------------------------------
				std::get<type_index>(this->all_particle_data_).push_back(&diffusion_dt_[m]);
				diffusion_dt_[m].resize(this->real_particles_bound_, Real(0));
			}
		};

		virtual DiffusionReactionParticles<BaseParticlesType> *ThisObjectPtr() override { return this; };
	};
}
#endif // DIFFUSION_REACTION_PARTICLES_H