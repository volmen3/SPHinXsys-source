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
 * @file 	particle_generator_lattice.h
 * @brief 	This is the base class of particle generator, which generates particles
 * 			with given positions and volumes. The direct generator simply generate
 * 			particle with given position and volume. The lattice generator generate
 * 			at lattice position by check whether the position is contained by a SPH body.
 * @author	Xiangyu Hu, Chi Zhang, Yongchuan Yu
 */

#ifndef PARTICLE_GENERATOR_LATTICE_H
#define PARTICLE_GENERATOR_LATTICE_H

#include "base_particle_generator.h"

namespace SPH
{

	class Shape;
	class ParticleSpacingByBodyShape;
	class ShellParticles;

	/**
	 * @class BaseParticleGeneratorLattice
	 * @brief Base class for generating particles from lattice positions for a body.
	 */
	class BaseParticleGeneratorLattice
	{
	public:
		explicit BaseParticleGeneratorLattice(SPHBody &sph_body);
		virtual ~BaseParticleGeneratorLattice(){};

	protected:
		Real lattice_spacing_;
		BoundingBox domain_bounds_;
		Shape &body_shape_;
	};

	/**
	 * @class ParticleGeneratorLattice
	 * @brief generate particles from lattice positions for a body.
	 */
	class ParticleGeneratorLattice : public BaseParticleGeneratorLattice, public ParticleGenerator
	{
	public:
		explicit ParticleGeneratorLattice(SPHBody &sph_body);
		virtual ~ParticleGeneratorLattice(){};

		virtual void initializeGeometricVariables() override;
	};

	/**
	 * @class ParticleGeneratorMultiResolution
	 * @brief generate multi-resolution particles from lattice positions for a body.
	 */
	class ParticleGeneratorMultiResolution : public ParticleGeneratorLattice
	{
	public:
		explicit ParticleGeneratorMultiResolution(SPHBody &sph_body);
		virtual ~ParticleGeneratorMultiResolution(){};

	protected:
		ParticleSpacingByBodyShape *particle_adaptation_;
		StdLargeVec<Real> &h_ratio_;

		virtual void initializePositionAndVolumetricMeasure(const Vecd &position, Real volume) override;
		virtual void initializeSmoothingLengthRatio(Real local_spacing);
	};

	/**
	 * @class ThickSurfaceParticleGeneratorLattice
	 * @brief Generate thick surface particles from lattice positions for a thin structure defined by a body shape.
	 * @details Here, a thick surface is defined as that the thickness is equal or larger than the proposed particle spacing. 
	 * Note that, this class should not be used for generating the thin surface particles, 
	 * which may be better generated from a geometric surface directly.
	 */
	class ThickSurfaceParticleGeneratorLattice : public BaseParticleGeneratorLattice, public SurfaceParticleGenerator
	{
	public:
		ThickSurfaceParticleGeneratorLattice(SPHBody &sph_body, Real global_avg_thickness);
		virtual ~ThickSurfaceParticleGeneratorLattice(){};

		virtual void initializeGeometricVariables() override;

	protected:
		Real total_volume_; /** Calculated from level set. */
		Real global_avg_thickness_;
		Real particle_spacing_;
		Real avg_particle_volume_;
		size_t number_of_cells_;
		size_t planned_number_of_particles_;
	};
}
#endif // PARTICLE_GENERATOR_LATTICE_H