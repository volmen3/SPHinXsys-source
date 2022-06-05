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
 * @file functors_iterators.h
 * @brief Definition of functors and iterators used for particle dynamics classes.
 * @author  Xiangyu Hu
 */

#ifndef FUNCTORS_ITERATORS_H
#define FUNCTORS_ITERATORS_H

#include "base_data_package.h"
#include "sph_data_containers.h"

#include <functional>

namespace SPH
{

	//----------------------------------------------------------------------
	//	Particle-wise operation and reduce functors 
	//----------------------------------------------------------------------
	typedef std::function<void(size_t, Real)> ParticleFunctor;

	template <class ReturnType>
	using ReduceFunctor = std::function<ReturnType(size_t, Real)>;
	//----------------------------------------------------------------------
	//	Particle-range-wise operation and reduce functors 
	//----------------------------------------------------------------------
	typedef std::function<void(const IndexRange &, Real)> RangeFunctor;

	template <class ReturnType>
	using ReduceRangeFunctor = std::function<ReturnType(const IndexRange &, Real)>;
	//----------------------------------------------------------------------
	//	Particle-list-wise operation and reduce functors 
	//----------------------------------------------------------------------
	typedef std::function<void(const IndexRange &, const IndexVector &, Real)> ListFunctor;

	//----------------------------------------------------------------------
	//	Body-wise iterators (for sequential and parallel computing).
	//----------------------------------------------------------------------
	void particle_for(size_t all_real_particles, const ParticleFunctor &functor, Real dt = 0.0);
	void particle_parallel_for(size_t all_real_particles, const ParticleFunctor &functor, Real dt = 0.0);

	void particle_for(size_t all_real_particles, const RangeFunctor &functor, Real dt = 0.0);
	void particle_parallel_for(size_t all_real_particles, const RangeFunctor &functor, Real dt = 0.0);

	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_reduce(size_t all_real_particles, ReturnType temp,
							  ReduceFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt = 0.0);

	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_parallel_reduce(size_t all_real_particles, ReturnType temp,
									   ReduceFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt = 0.0);

	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_reduce(size_t all_real_particles, ReturnType temp,
							  ReduceRangeFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt = 0.0);

	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_parallel_reduce(size_t all_real_particles, ReturnType temp,
									   ReduceRangeFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt = 0.0);

	void particle_for(SplitCellLists &split_cell_lists, const ParticleFunctor &functor, Real dt = 0.0);
	void particle_parallel_for(SplitCellLists &split_cell_lists, const ParticleFunctor &functor, Real dt = 0.0);

	//----------------------------------------------------------------------
	//	BodyPartByParticle-wise iterators (for sequential and parallel computing).
	//----------------------------------------------------------------------
	void particle_for(IndexVector &body_part_particles, const ParticleFunctor &functor, Real dt = 0.0);
	void particle_parallel_for(IndexVector &body_part_particles, const ParticleFunctor &functor, Real dt = 0.0);

	void particle_for(IndexVector &body_part_particles, const ListFunctor &functor, Real dt = 0.0);
	void particle_parallel_for(IndexVector &body_part_particles, const ListFunctor &functor, Real dt = 0.0);

	//----------------------------------------------------------------------
	//	Reduce (binary) operation functors.
	//----------------------------------------------------------------------
	template <class ReturnType>
	struct ReduceSum
	{
		ReturnType operator()(const ReturnType &x, const ReturnType &y) const { return x + y; };
	};

	struct ReduceMax
	{
		Real operator()(Real x, Real y) const { return SMAX(x, y); };
	};

	struct ReduceMin
	{
		Real operator()(Real x, Real y) const { return SMIN(x, y); };
	};

	struct ReduceOR
	{
		bool operator()(bool x, bool y) const { return x || y; };
	};

	struct ReduceAND
	{
		bool operator()(bool x, bool y) const { return x && y; };
	};

	struct ReduceLowerBound
	{
		Vecd operator()(const Vecd &x, const Vecd &y) const
		{
			Vecd lower_bound;
			for (int i = 0; i < lower_bound.size(); ++i)
				lower_bound[i] = SMIN(x[i], y[i]);
			return lower_bound;
		};
	};

	struct ReduceUpperBound
	{
		Vecd operator()(const Vecd &x, const Vecd &y) const
		{
			Vecd upper_bound;
			for (int i = 0; i < upper_bound.size(); ++i)
				upper_bound[i] = SMAX(x[i], y[i]);
			return upper_bound;
		};
	};
}
#endif // FUNCTORS_ITERATORS_H
