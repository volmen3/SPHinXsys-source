/**
 * @file elastic_solid.cpp
 * @author Chi Zhang and Xiangyu Hu
 */

#include "elastic_solid.h"

#include "base_particles.hpp"

#ifdef max
#undef max
#endif

namespace SPH
{
	//=================================================================================================//
	void ElasticSolid::setSoundSpeeds()
	{
		c0_ = sqrt(K0_ / rho0_);
		ct0_ = sqrt(E0_ / rho0_);
		cs0_ = sqrt(G0_ / rho0_);
	};
	//=================================================================================================//
	Matd ElasticSolid::NumericalDampingRightCauchy(
		Matd &F, Matd &dF_dt, Real smoothing_length, size_t particle_index_i)
	{
		Matd strain_rate = 0.5 * (~dF_dt * F + ~F * dF_dt);
		Matd normal_rate = getDiagonal(strain_rate);
		return 0.5 * rho0_ * (cs0_ * (strain_rate - normal_rate) + c0_ * normal_rate) * smoothing_length;
	}
	//=================================================================================================//
	Matd ElasticSolid::NumericalDampingLeftCauchy(
		Matd &F, Matd &dF_dt, Real smoothing_length, size_t particle_index_i)
	{
		Matd strain_rate = 0.5 * (dF_dt * ~F + F * ~dF_dt);
		Matd normal_rate = getDiagonal(strain_rate);
		return 0.5 * rho0_ * (cs0_ * (strain_rate - normal_rate) + c0_ * normal_rate) * smoothing_length;
	}
	//=================================================================================================//
	Real ElasticSolid::PairNumericalDamping(Real dE_dt_ij, Real smoothing_length)
	{
		return 0.5 * rho0_ * c0_ * dE_dt_ij * smoothing_length;
	}
	//=================================================================================================//
	Matd ElasticSolid::DeviatoricKirchhoff(const Matd &deviatoric_be)
	{
		return G0_ * deviatoric_be;
	}
	//=================================================================================================//
	LinearElasticSolid::
		LinearElasticSolid(Real rho0, Real youngs_modulus, Real poisson_ratio) : ElasticSolid(rho0)
	{
		material_type_name_ = "LinearElasticSolid";
		E0_ = youngs_modulus;
		nu_ = poisson_ratio;
		G0_ = getShearModulus(youngs_modulus, poisson_ratio);
		K0_ = getBulkModulus(youngs_modulus, poisson_ratio);
		lambda0_ = getLambda(youngs_modulus, poisson_ratio);
		setSoundSpeeds();
		setContactStiffness(c0_);
	}
	//=================================================================================================//
	Real LinearElasticSolid::getBulkModulus(Real youngs_modulus, Real poisson_ratio)
	{
		return youngs_modulus / 3.0 / (1.0 - 2.0 * poisson_ratio);
	}
	//=================================================================================================//
	Real LinearElasticSolid::getShearModulus(Real youngs_modulus, Real poisson_ratio)
	{
		return 0.5 * youngs_modulus / (1.0 + poisson_ratio);
	}
	//=================================================================================================//
	Real LinearElasticSolid::getLambda(Real youngs_modulus, Real poisson_ratio)
	{
		return nu_ * youngs_modulus / (1.0 + poisson_ratio) / (1.0 - 2.0 * poisson_ratio);
	}
	//=================================================================================================//
	Matd LinearElasticSolid::StressPK2(Matd &F, size_t particle_index_i)
	{
		Matd strain = 0.5 * (~F * F - Matd(1.0));
		return lambda0_ * strain.trace() * Matd(1.0) + 2.0 * G0_ * strain;
	}
	//=================================================================================================//
	Matd LinearElasticSolid::StressCauchy(Matd &almansi_strain, Matd &F, size_t particle_index_i)
	{
		return lambda0_ * almansi_strain.trace() * Matd(1.0) + 2.0 * G0_ * almansi_strain;
	}
	//=================================================================================================//
	Real LinearElasticSolid::VolumetricKirchhoff(Real J)
	{
		return K0_ * J * (J - 1);
	}
	//=================================================================================================//
	Matd NeoHookeanSolid::StressPK2(Matd &F, size_t particle_index_i)
	{
		// This formulation allows negative determinant of F. Please refer
		// Smith et al. (2018) Stable Neo-Hookean Flesh Simulation.
		// ACM Transactions on Graphics, Vol. 37, No. 2, Article 12.
		Matd right_cauchy = ~F * F;
		Real J = det(F);
		return G0_ * Matd(1.0) + (lambda0_ * (J - 1.0) - G0_) * J * inverse(right_cauchy);
	}
	//=================================================================================================//
	Matd NeoHookeanSolid::StressCauchy(Matd &almansi_strain, Matd &F, size_t particle_index_i)
	{
		Real J = det(F);
		Matd B = inverse(-2.0 * almansi_strain + Matd(1.0));
		Matd cauchy_stress = 0.5 * K0_ * (J - 1.0 / J) * Matd(1.0) +
							 G0_ * pow(J, -2.0 / (Real)Dimensions - 1.0) *
								 (B - B.trace() / (Real)Dimensions * Matd(1.0));
		return cauchy_stress;
	}
	//=================================================================================================//
	Real NeoHookeanSolid::VolumetricKirchhoff(Real J)
	{
		return 0.5 * K0_ * (J * J - 1);
	}
	//=================================================================================================//
	Matd NeoHookeanSolidIncompressible::StressPK2(Matd &F, size_t particle_index_i)
	{
		Matd right_cauchy = ~F * F;
		Real I_1 = right_cauchy.trace(); // first strain invariant
		Real I_3 = det(right_cauchy);	 // first strain invariant
		return G0_ * std::pow(I_3, -1.0 / 3.0) * (Matd(1.0) - 1.0 / 3.0 * I_1 * inverse(right_cauchy));
	}
	//=================================================================================================//
	Matd NeoHookeanSolidIncompressible::
		StressCauchy(Matd &almansi_strain, Matd &F, size_t particle_index_i)
	{
		// TODO: implement
		return {};
	}
	//=================================================================================================//
	Real NeoHookeanSolidIncompressible::VolumetricKirchhoff(Real J)
	{
		return 0.5 * K0_ * (J * J - 1);
	}
	//=================================================================================================//
	OrthotropicSolid::OrthotropicSolid(Real rho_0, std::array<Vecd, 3> a, std::array<Real, 3> E,
									   std::array<Real, 3> G, std::array<Real, 3> poisson)
		// set parameters for parent class: LinearElasticSolid
		// we take the max. E and max. poisson to approximate the maximum of
		// the Bulk modulus --> for time step size calculation
		: LinearElasticSolid(rho_0, std::max({E[0], E[1], E[2]}),
							 std::max({poisson[0], poisson[1], poisson[2]})),
		  a_(a), E_(E), G_(G), poisson_(poisson)
	{
		// parameters for derived class
		material_type_name_ = "OrthotropicSolid";
		CalculateA0();
		CalculateAllMu();
		CalculateAllLambda();
	};
	//=================================================================================================//
	Matd OrthotropicSolid::StressPK2(Matd &F, size_t particle_index_i)
	{
		Matd strain = 0.5 * (~F * F - Matd(1.0));
		Matd stress_PK2 = Matd(0);
		for (int i = 0; i < 3; i++)
		{
			// outer sum (a{1-3})
			Matd Summa2 = Matd(0);
			for (int j = 0; j < 3; j++)
			{
				// inner sum (b{1-3})
				Summa2 += Lambda_[i][j] * (CalculateDoubleDotProduct(A_[i], strain) * A_[j] +
										   CalculateDoubleDotProduct(A_[j], strain) * A_[i]);
			}
			stress_PK2 += Mu_[i] * (((A_[i] * strain) + (strain * A_[i])) + 1 / 2 * (Summa2));
		}
		return stress_PK2;
	}
	//=================================================================================================//
	Real OrthotropicSolid::VolumetricKirchhoff(Real J)
	{
		return K0_ * J * (J - 1);
	}
	//=================================================================================================//
	void OrthotropicSolid::CalculateA0()
	{
		A_[0] = SimTK::outer(a_[0], a_[0]);
		A_[1] = SimTK::outer(a_[1], a_[1]);
		A_[2] = SimTK::outer(a_[2], a_[2]);
	}
	//=================================================================================================//
	void OrthotropicSolid::CalculateAllMu()
	{
		// the equations of G_, to calculate Mu the equations must be solved for Mu[0,1,2]
		// G_[0]=2/(Mu_[0]+Mu_[1]);
		// G_[1]=2/(Mu_[1]+Mu_[2]);
		// G_[2]=2/(Mu_[2]+Mu_[0]);

		Mu_[0] = 1 / G_[0] + 1 / G_[2] - 1 / G_[1];
		Mu_[1] = 1 / G_[1] + 1 / G_[0] - 1 / G_[2];
		Mu_[2] = 1 / G_[2] + 1 / G_[1] - 1 / G_[0];
	}
	//=================================================================================================//
	void OrthotropicSolid::CalculateAllLambda()
	{
		// first we calculate the upper left part, a 3x3 matrix of the full compliance matrix
		Matd Compliance = Matd(
			Vecd(1 / E_[0], -poisson_[0] / E_[0], -poisson_[1] / E_[0]),
			Vecd(-poisson_[0] / E_[1], 1 / E_[1], -poisson_[2] / E_[1]),
			Vecd(-poisson_[1] / E_[2], -poisson_[2] / E_[2], 1 / E_[2]));

		// we calculate the inverse of the Compliance matrix, and calculate the lambdas elementwise
		Matd Compliance_inv = SimTK::inverse(Compliance);
		// Lambda_ is a 3x3 matrix
		Lambda_[0][0] = Compliance_inv[0][0] - 2 * Mu_[0];
		Lambda_[1][1] = Compliance_inv[1][1] - 2 * Mu_[1];
		Lambda_[2][2] = Compliance_inv[2][2] - 2 * Mu_[2];
		Lambda_[0][1] = Compliance_inv[0][1];
		Lambda_[0][2] = Compliance_inv[0][2];
		Lambda_[1][2] = Compliance_inv[1][2];
		// the matrix is symmetric
		Lambda_[1][0] = Lambda_[0][1];
		Lambda_[2][0] = Lambda_[0][2];
		Lambda_[2][1] = Lambda_[1][2];
	}
	//=================================================================================================//
	Matd FeneNeoHookeanSolid::StressPK2(Matd &F, size_t particle_index_i)
	{
		Matd right_cauchy = ~F * F;
		Matd strain = 0.5 * (right_cauchy - Matd(1.0));
		Real J = det(F);
		return G0_ / (1.0 - 2.0 * strain.trace() / j1_m_) * Matd(1.0) +
			   (lambda0_ * (J - 1.0) - G0_) * J * inverse(right_cauchy);
	}
	//=================================================================================================//
	Real Muscle::getShearModulus(const Real (&a0)[4], const Real (&b0)[4])
	{
		// This is only the background material property.
		// The previous version seems not correct because it leads to
		// that shear modulus is even bigger than bulk modulus.
		return a0[0];
	}
	//=================================================================================================//
	Real Muscle::getPoissonRatio(Real bulk_modulus, const Real (&a0)[4], const Real (&b0)[4])
	{
		Real shear_modulus = getShearModulus(a0, b0);
		return 0.5 * (3.0 * bulk_modulus - 2.0 * shear_modulus) /
			   (3.0 * bulk_modulus + shear_modulus);
	}
	//=================================================================================================//
	Real Muscle::getYoungsModulus(Real bulk_modulus, const Real (&a0)[4], const Real (&b0)[4])
	{
		return 3.0 * bulk_modulus * (1.0 - 2.0 * getPoissonRatio(bulk_modulus, a0, b0));
	}
	//=================================================================================================//
	Matd Muscle::StressPK2(Matd &F, size_t i)
	{
		Matd right_cauchy = ~F * F;
		Real I_ff_1 = SimTK::dot(right_cauchy * f0_, f0_) - 1.0;
		Real I_ss_1 = SimTK::dot(right_cauchy * s0_, s0_) - 1.0;
		Real I_fs = SimTK::dot(right_cauchy * f0_, s0_);
		Real I_1_1 = right_cauchy.trace() - Real(Dimensions);
		Real J = det(F);
		return a0_[0] * exp(b0_[0] * I_1_1) * Matd(1.0) +
			   (lambda0_ * (J - 1.0) - a0_[0]) * J * inverse(right_cauchy) +
			   2.0 * a0_[1] * I_ff_1 * exp(b0_[1] * I_ff_1 * I_ff_1) * f0f0_ +
			   2.0 * a0_[2] * I_ss_1 * exp(b0_[2] * I_ss_1 * I_ss_1) * s0s0_ +
			   a0_[3] * I_fs * exp(b0_[3] * I_fs * I_fs) * f0s0_;
	}
	//=================================================================================================//
	Real Muscle::VolumetricKirchhoff(Real J)
	{
		return K0_ * J * (J - 1);
	}
	//=================================================================================================//
	Matd LocallyOrthotropicMuscle::StressPK2(Matd &F, size_t i)
	{
		Matd right_cauchy = ~F * F;
		Real I_ff_1 = SimTK::dot(right_cauchy * local_f0_[i], local_f0_[i]) - 1.0;
		Real I_ss_1 = SimTK::dot(right_cauchy * local_s0_[i], local_s0_[i]) - 1.0;
		Real I_fs = SimTK::dot(right_cauchy * local_f0_[i], local_s0_[i]);
		Real I_1_1 = right_cauchy.trace() - Real(Dimensions);
		Real J = det(F);
		return a0_[0] * exp(b0_[0] * I_1_1) * Matd(1.0) +
			   (lambda0_ * (J - 1.0) - a0_[0]) * J * inverse(right_cauchy) +
			   2.0 * a0_[1] * I_ff_1 * exp(b0_[1] * I_ff_1 * I_ff_1) * local_f0f0_[i] +
			   2.0 * a0_[2] * I_ss_1 * exp(b0_[2] * I_ss_1 * I_ss_1) * local_s0s0_[i] +
			   a0_[3] * I_fs * exp(b0_[3] * I_fs * I_fs) * local_f0s0_[i];
	}
	//=================================================================================================//
	void LocallyOrthotropicMuscle::assignBaseParticles(BaseParticles *base_particles)
	{
		Muscle::assignBaseParticles(base_particles);
		initializeFiberAndSheet();
	}
	//=================================================================================================//
	void LocallyOrthotropicMuscle::initializeFiberAndSheet()
	{
		base_particles_->registerVariable(local_f0_, "Fiber");
		base_particles_->registerVariable(local_s0_, "Sheet");
		base_particles_->addVariableNameToList<Vecd>(reload_local_parameters_, "Fiber");
		base_particles_->addVariableNameToList<Vecd>(reload_local_parameters_, "Sheet");
		initializeFiberAndSheetTensors();
	}
	//=================================================================================================//
	void LocallyOrthotropicMuscle::initializeFiberAndSheetTensors()
	{
		base_particles_->registerVariable(local_f0f0_, "FiberFiberTensor", [&](size_t i) -> Matd
										  { return SimTK::outer(local_f0_[i], local_f0_[i]); });
		base_particles_->registerVariable(local_s0s0_, "SheetSheetTensor", [&](size_t i) -> Matd
										  { return SimTK::outer(local_s0_[i], local_s0_[i]); });
		base_particles_->registerVariable(local_f0s0_, "FiberSheetTensor", [&](size_t i) -> Matd
										  { return SimTK::outer(local_f0_[i], local_s0_[i]); });
	}
	//=================================================================================================//
	void LocallyOrthotropicMuscle::readFromXmlForLocalParameters(const std::string &filefullpath)
	{
		BaseMaterial::readFromXmlForLocalParameters(filefullpath);
		size_t total_real_particles = base_particles_->total_real_particles_;
		for (size_t i = 0; i != total_real_particles; i++)
		{
			local_f0f0_[i] = SimTK::outer(local_f0_[i], local_f0_[i]);
			local_s0s0_[i] = SimTK::outer(local_s0_[i], local_s0_[i]);
			local_f0s0_[i] = SimTK::outer(local_f0_[i], local_s0_[i]);
		}
	}
	//=================================================================================================//
}
