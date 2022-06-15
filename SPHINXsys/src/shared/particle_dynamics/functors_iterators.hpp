/**
 * @file 	functors_iterators.hpp
 * @brief 	This is the implementation of the templates functors and iterators
 * @author	Xiangyu Hu
 */

#ifndef FUNCTORS_ITERATORS_HPP
#define FUNCTORS_ITERATORS_HPP

#include "functors_iterators.h"

namespace SPH
{
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_reduce(size_t all_real_particles, ReturnType temp,
							   ReduceFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt)
	{
		for (size_t i = 0; i < all_real_particles; ++i)
		{
			temp = operation(temp, functor(i, dt));
		}
		return temp;
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_parallel_reduce(size_t all_real_particles, ReturnType temp,
										ReduceFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt)
	{
		return parallel_reduce(
			blocked_range<size_t>(0, all_real_particles),
			temp, [&](const blocked_range<size_t> &r, ReturnType temp0) -> ReturnType
			{
				for (size_t i = r.begin(); i != r.end(); ++i)
				{
					temp0 = operation(temp0, functor(i, dt));
				}
				return temp0; },
			[&](ReturnType x, ReturnType y) -> ReturnType
			{
				return operation(x, y);
			});
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_reduce(size_t all_real_particles, ReturnType temp,
							   ReduceRangeFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt)
	{
		return operation(temp, functor(blocked_range<size_t>(0, all_real_particles), dt));
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_parallel_reduce(size_t all_real_particles, ReturnType temp,
										ReduceRangeFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt)
	{
		return parallel_reduce(
			blocked_range<size_t>(0, all_real_particles),
			temp, [&](const blocked_range<size_t> &r, ReturnType temp0) -> ReturnType
			{ return operation(temp0, functor(r, dt)); },
			[&](ReturnType x, ReturnType y) -> ReturnType
			{
				return operation(x, y);
			});
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_reduce(const IndexVector &body_part_particles, ReturnType temp,
							   ReduceListFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt)
	{
		return operation(temp, functor(blocked_range<size_t>(0, body_part_particles.size()), body_part_particles, dt));
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType particle_parallel_reduce(const IndexVector &body_part_particles, ReturnType temp,
										ReduceListFunctor<ReturnType> &functor, ReduceOperation &operation, Real dt)
	{
		return parallel_reduce(
			blocked_range<size_t>(0, body_part_particles.size()),
			temp, [&](const blocked_range<size_t> &r, ReturnType temp0) -> ReturnType
			{ return operation(temp0, functor(r, body_part_particles, dt)); },
			[&](ReturnType x, ReturnType y) -> ReturnType
			{
				return operation(x, y);
			});
	}
	//=================================================================================================//
}
#endif // FUNCTORS_ITERATORS_HPP
