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
	ReturnType ReduceIterator(size_t total_real_particles, ReturnType temp,
							  ReduceFunctor<ReturnType> &reduce_functor, ReduceOperation &reduce_operation, Real dt)
	{
		for (size_t i = 0; i < total_real_particles; ++i)
		{
			temp = reduce_operation(temp, reduce_functor(i, dt));
		}
		return temp;
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType ReduceIterator_parallel(size_t total_real_particles, ReturnType temp,
									   ReduceFunctor<ReturnType> &reduce_functor, ReduceOperation &reduce_operation, Real dt)
	{
		return parallel_reduce(
			blocked_range<size_t>(0, total_real_particles),
			temp, [&](const blocked_range<size_t> &r, ReturnType temp0) -> ReturnType
			{
				for (size_t i = r.begin(); i != r.end(); ++i)
				{
					temp0 = reduce_operation(temp0, reduce_functor(i, dt));
				}
				return temp0; },
			[&](ReturnType x, ReturnType y) -> ReturnType
			{
				return reduce_operation(x, y);
			});
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType ReduceIterator(size_t total_real_particles, ReturnType temp,
							  ReduceRangeFunctor<ReturnType> &reduce_functor, ReduceOperation &reduce_operation, Real dt)
	{
		return reduce_operation(temp, reduce_functor(blocked_range<size_t>(0, total_real_particles), dt));
	}
	//=================================================================================================//
	template <class ReturnType, typename ReduceOperation>
	ReturnType ReduceIterator_parallel(size_t total_real_particles, ReturnType temp,
									   ReduceRangeFunctor<ReturnType> &reduce_functor, ReduceOperation &reduce_operation, Real dt)
	{
		return parallel_reduce(
			blocked_range<size_t>(0, total_real_particles),
			temp, [&](const blocked_range<size_t> &r, ReturnType temp0) -> ReturnType
			{ return reduce_operation(temp0, reduce_functor(r, dt)); },
			[&](ReturnType x, ReturnType y) -> ReturnType
			{
				return reduce_operation(x, y);
			});
	}
	//=================================================================================================//
}
#endif // FUNCTORS_ITERATORS_HPP
