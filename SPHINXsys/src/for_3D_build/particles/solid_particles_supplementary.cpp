#include "solid_particles.h"
#include "solid_particles_variable.h"
#include "base_body.h"

#include <iterator>

namespace SPH
{
	//=================================================================================================//
	Real ElasticSolidParticles::getVonMisesStrain(size_t particle_i)
	{

		Mat3d F = F_[particle_i];
		Mat3d epsilon = 0.5 * (~F * F - Matd(1.0)); // calculation of the Green-Lagrange strain tensor

		Real epsilonxx = epsilon(0, 0);
		Real epsilonyy = epsilon(1, 1);
		Real epsilonzz = epsilon(2, 2);
		Real epsilonxy = epsilon(0, 1);
		Real epsilonxz = epsilon(0, 2);
		Real epsilonyz = epsilon(1, 2);

		return sqrt((1.0 / 3.0) * (std::pow(epsilonxx - epsilonyy, 2.0) +
								   std::pow(epsilonyy - epsilonzz, 2.0) + std::pow(epsilonzz - epsilonxx, 2.0)) +
					2.0 * (std::pow(epsilonxy, 2.0) + std::pow(epsilonyz, 2.0) + std::pow(epsilonxz, 2.0)));
	}
	//=================================================================================================//
	Real ElasticSolidParticles::getVonMisesStrainDynamic(size_t particle_i, Real poisson)
	{
		Mat3d epsilon = getGreenLagrangeStrain(particle_i); // calculation of the Green-Lagrange strain tensor

		Vec3d principal_strains = getPrincipalValuesFromMatrix(epsilon);
		Real eps_1 = principal_strains[0];
		Real eps_2 = principal_strains[1];
		Real eps_3 = principal_strains[2];

		return 1.0 / (1.0 + poisson) * std::sqrt(0.5 * (powerN(eps_1 - eps_2, 2) + powerN(eps_2 - eps_3, 2) + powerN(eps_3 - eps_1, 2)));
	}
	//=============================================================================================//
	void VonMisesStress::update(size_t index_i, Real dt)
	{
		Real J = rho0_ / rho_[index_i];
		Mat3d F = F_[index_i];
		Mat3d stress_PK1 = F * material_->StressPK2(F, index_i);
		Mat3d sigma = (stress_PK1 * ~F) / J;

		Real sigmaxx = sigma(0, 0);
		Real sigmayy = sigma(1, 1);
		Real sigmazz = sigma(2, 2);
		Real sigmaxy = sigma(0, 1);
		Real sigmaxz = sigma(0, 2);
		Real sigmayz = sigma(1, 2);

		derived_variable_[index_i] =
			sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy + sigmazz * sigmazz -
				 sigmaxx * sigmayy - sigmaxx * sigmazz - sigmayy * sigmazz +
				 3.0 * (sigmaxy * sigmaxy + sigmaxz * sigmaxz + sigmayz * sigmayz));
	}
	//=================================================================================================//
}
