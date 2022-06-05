/**
 * @file 	functors_iterators.cpp
 * @author	Xiangyu Hu
 */
#include "functors_iterators.h"

#include "cell_linked_list.h"
//=============================================================================================//
namespace SPH
{
    //=============================================================================================//
    void particle_for(size_t all_real_particles, const ParticleFunctor &functor, Real dt)
    {
        for (size_t i = 0; i < all_real_particles; ++i)
            functor(i, dt);
    }
    //=============================================================================================//
    void particle_parallel_for(size_t all_real_particles, const ParticleFunctor &functor, Real dt)
    {
        parallel_for(
            blocked_range<size_t>(0, all_real_particles),
            [&](const blocked_range<size_t> &r)
            {
                for (size_t i = r.begin(); i < r.end(); ++i)
                {
                    functor(i, dt);
                }
            },
            ap);
    }
    //=============================================================================================//
    void particle_for(size_t all_real_particles, const RangeFunctor &functor, Real dt)
    {
        functor(blocked_range<size_t>(0, all_real_particles), dt);
    }
    //=============================================================================================//
    void particle_parallel_for(size_t all_real_particles, const RangeFunctor &functor, Real dt)
    {
        parallel_for(
            blocked_range<size_t>(0, all_real_particles),
            [&](const blocked_range<size_t> &r)
            {
                functor(r, dt);
            },
            ap);
    }
    //=================================================================================================//
    void particle_for(SplitCellLists &split_cell_lists, const ParticleFunctor &functor, Real dt)
    {
        Real dt2 = dt * 0.5;
        // forward sweeping
        for (size_t k = 0; k != split_cell_lists.size(); ++k)
        {
            ConcurrentCellLists &cell_lists = split_cell_lists[k];
            for (size_t l = 0; l != cell_lists.size(); ++l)
            {
                IndexVector &particle_indexes = cell_lists[l]->real_particle_indexes_;
                for (size_t i = 0; i != particle_indexes.size(); ++i)
                {
                    functor(particle_indexes[i], dt2);
                }
            }
        }

        // backward sweeping
        for (size_t k = split_cell_lists.size(); k != 0; --k)
        {
            ConcurrentCellLists &cell_lists = split_cell_lists[k - 1];
            for (size_t l = 0; l != cell_lists.size(); ++l)
            {
                IndexVector &particle_indexes = cell_lists[l]->real_particle_indexes_;
                for (size_t i = particle_indexes.size(); i != 0; --i)
                {
                    functor(particle_indexes[i - 1], dt2);
                }
            }
        }
    }
    //=================================================================================================//
    void particle_parallel_for(SplitCellLists &split_cell_lists, const ParticleFunctor &functor, Real dt)
    {
        Real dt2 = dt * 0.5;
        // forward sweeping
        for (size_t k = 0; k != split_cell_lists.size(); ++k)
        {
            ConcurrentCellLists &cell_lists = split_cell_lists[k];
            parallel_for(
                blocked_range<size_t>(0, cell_lists.size()),
                [&](const blocked_range<size_t> &r)
                {
                    for (size_t l = r.begin(); l < r.end(); ++l)
                    {
                        IndexVector &particle_indexes = cell_lists[l]->real_particle_indexes_;
                        for (size_t i = 0; i < particle_indexes.size(); ++i)
                        {
                            functor(particle_indexes[i], dt2);
                        }
                    }
                },
                ap);
        }

        // backward sweeping
        for (size_t k = split_cell_lists.size(); k != 0; --k)
        {
            ConcurrentCellLists &cell_lists = split_cell_lists[k - 1];
            parallel_for(
                blocked_range<size_t>(0, cell_lists.size()),
                [&](const blocked_range<size_t> &r)
                {
                    for (size_t l = r.begin(); l < r.end(); ++l)
                    {
                        IndexVector &particle_indexes = cell_lists[l]->real_particle_indexes_;
                        for (size_t i = particle_indexes.size(); i != 0; --i)
                        {
                            functor(particle_indexes[i - 1], dt2);
                        }
                    }
                },
                ap);
        }
    }
    //=================================================================================================//
    void particle_for(IndexVector &body_part_particles, const ParticleFunctor &functor, Real dt)
    {
        for (size_t i = 0; i < body_part_particles.size(); ++i)
        {
            functor(body_part_particles[i], dt);
        }
    }
    //=================================================================================================//
    void particle_parallel_for(IndexVector &body_part_particles, const ParticleFunctor &functor, Real dt)
    {
        parallel_for(
            blocked_range<size_t>(0, body_part_particles.size()),
            [&](const blocked_range<size_t> &r)
            {
                for (size_t i = r.begin(); i < r.end(); ++i)
                {
                    functor(body_part_particles[i], dt);
                }
            },
            ap);
    }
    //=================================================================================================//
    void particle_for(IndexVector &body_part_particles, const ListFunctor &functor, Real dt)
    {
        functor(blocked_range<size_t>(0, body_part_particles.size()), body_part_particles, dt);
    }
    //=================================================================================================//
    void particle_parallel_for(IndexVector &body_part_particles, const ListFunctor &functor, Real dt)
    {
        parallel_for(
            blocked_range<size_t>(0, body_part_particles.size()),
            [&](const blocked_range<size_t> &r)
            {
                functor(r, body_part_particles, dt);
            },
            ap);
    }
    //=============================================================================================//
}
//=============================================================================================//