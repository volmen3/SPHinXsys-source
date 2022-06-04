/**
 * @file 	inelastic_dynamics.cpp
 * @author	Xiaojing Tang, Chi Zhang and Xiangyu Hu
 */

#include "inelastic_dynamics.h"

using namespace SimTK;

namespace SPH
{
	namespace solid_dynamics
	{
		//=================================================================================================//
		PlasticStressRelaxationFirstHalf::
			PlasticStressRelaxationFirstHalf(BaseBodyRelationInner &inner_relation) : StressRelaxationFirstHalf(inner_relation),
																					  plastic_solid_(DynamicCast<PlasticSolid>(this, material_))
		{
			numerical_dissipation_factor_ = 0.5;
		}
		//=================================================================================================//
		void PlasticStressRelaxationFirstHalf::initializeRange(const blocked_range<size_t> particle_range, Real dt)
		{
			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				pos_n_[index_i] += vel_n_[index_i] * dt * 0.5;
			}

			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				F_[index_i] += dF_dt_[index_i] * dt * 0.5;
			}

			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				rho_n_[index_i] = rho0_ / SimTK::det(F_[index_i]);
			}

			for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
			{
				stress_PK1_[index_i] = plastic_solid_->PlasticConstitutiveRelation(F_[index_i], index_i, dt);
			}
		}
		//=================================================================================================//
	}
}
