/**
 * @file 	beam_pulling.cpp
 * @brief 	This is the test for comparing SPH with ABAQUS.
 * @author 	Anyong Zhang
 */

#include "sphinxsys.h"
 /** Name space. */
using namespace SPH;

/** Geometry parameters. */
Real resolution_ref = 0.005;
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-0.026, -0.026, -0.021),
	Vecd(0.026, 0.026, 0.101));

/** Physical parameters */
Real rho = 1265;	// kg/m^3
Real poisson_ratio = 0.45;	
Real Youngs_modulus = 5e4;	// Pa
Real physical_viscosity = 500;

/** Load Parameters */
Real load_total_force = 12.5; // N


/**
* @brief define the beam body
*/
class Beam : public SolidBody
{
public:
	Beam(SPHSystem& system, const std::string& body_name)
		: SolidBody(system, body_name)
	{
		string fname_ = "./input/beam.stl";
		TriangleMeshShapeSTL stl_shape(fname_, Vecd(0), 0.001);
		body_shape_.add<LevelSetShape>(this, stl_shape, true);
	}

};

/* define load*/
class LoadForce :public PartSimpleDynamicsByParticle, public solid_dynamics::SolidDataSimple
{
public:
	LoadForce(SPHBody& sph_body, BodyPartByParticle& body_part, StdVec<array<Real, 2>> f_arr)
		:PartSimpleDynamicsByParticle(sph_body, body_part),
		solid_dynamics::SolidDataSimple(sph_body),
		dvel_dt_prior(particles_->dvel_dt_prior_),
		force_arr_(f_arr),
		mass_n_(particles_->mass_),
		particles_num_(body_part.body_part_particles_.size())
	{}

protected:
	StdLargeVec<Vecd>& dvel_dt_prior;
	StdLargeVec<Real>& mass_n_;
	StdVec<array<Real, 2>> force_arr_;
	Real total_area_;
	size_t particles_num_;

protected:

	virtual Real getForce(Real time)
	{
		for (size_t i = 1; i < force_arr_.size(); i++)
		{
			if (time >= force_arr_[i - 1][0] && time < force_arr_[i][0])
			{
				Real slope = (force_arr_[i][1] - force_arr_[i - 1][1]) / (force_arr_[i][0] - force_arr_[i - 1][0]);
				Real vel = (time - force_arr_[i - 1][0]) * slope + force_arr_[i - 1][1];
				return vel;
			}
			else if (time > force_arr_.back()[0])
				return force_arr_.back()[1];
		}
	}

	virtual void Update(size_t index_i, Real time = 0.0)
	{
		Vecd normal(0, 0, 1); // pulling direction, i.e. positive z direction
		Real mean_force_ = getForce(time) / particles_num_; // force(norm) on each particle
		dvel_dt_prior[index_i] += (mean_force_ / mass_n_[index_i]) * normal;
	}
};

/**
 *  The main program
 */
int main()
{
	/** Setup the system. Please the make sure the global domain bounds are correctly defined. */
	SPHSystem system(system_domain_bounds, resolution_ref);
	system.restart_step_ = 0;
	GlobalStaticVariables::physical_time_ = 0;
	In_Output in_output(system);

	/** Import a beam body, with corresponding material and particles. */
	Beam beam_body(system, "Beam_Body");
	ElasticSolidParticles beam_body_particles(beam_body,
								makeShared<LinearElasticSolid>(rho, Youngs_modulus, poisson_ratio));

	
	/** topology */
	BodyRelationInner beam_body_inner(beam_body);

	 /** initialize a time step */
	TimeStepInitialization beam_initialize(beam_body);

	/** Corrected configuration. */
	solid_dynamics::CorrectConfiguration corrected_configuration(beam_body_inner);

	/** Time step size calculation. */
	solid_dynamics::AcousticTimeStepSize computing_time_step_size(beam_body);
	solid_dynamics::UpdateElasticNormalDirection update_beam_normal(beam_body);

	/** active and passive stress relaxation. */
	//solid_dynamics::StressRelaxationFirstHalf stress_relaxation_first_half(beam_body_inner);
	solid_dynamics::KirchhoffParticleStressRelaxationFirstHalf stress_relaxation_first_half(beam_body_inner);
	//solid_dynamics::KirchhoffStressRelaxationFirstHalf stress_relaxation_first_half(beam_body_inner);

	solid_dynamics::StressRelaxationSecondHalf stress_relaxation_second_half(beam_body_inner);

	/** specify end-time for defining the force-time profile */
	Real end_time = 1;

	/** === define load === */
	/** create a brick to tag the surface */
	Vecd half_size_0(0.03, 0.03, resolution_ref);
	TriangleMeshShapeBrick end_0(half_size_0, 1, Vecd(0.00, 0.00, 0.1));
	BodyRegionByParticle load_surface(beam_body, "Load_Surface", end_0);
	StdVec<array<Real, 2>> force_over_time = {
		{0.0,0.0},
		{0.1 * end_time, 0.1 * load_total_force},
		{0.4 * end_time, load_total_force},
		{end_time, load_total_force}
	};
	LoadForce pull_force(beam_body, load_surface, force_over_time);
	cout << "load surface particle number: " << load_surface.body_part_particles_.size() << endl;


	//=== define constraint ===
	/* create a brick to tag the region */
	Vecd half_size_1(0.03, 0.03, 0.02);
	TriangleMeshShapeBrick end_1(half_size_1, 1, Vecd(0.0, 0.0, -0.02));
	BodyRegionByParticle holder(beam_body, "Holder", end_1);
	solid_dynamics::ConstrainSolidBodyRegion constrain_holder(beam_body, holder);

	/** Damping with the solid body*/
	DampingWithRandomChoice<DampingPairwiseInner<Vec3d>>
		beam_damping(beam_body_inner, 0.1, "Velocity", physical_viscosity);

	/** Output */
	BodyStatesRecordingToVtp write_states(in_output, system.real_bodies_);

	/* time step begins */
	GlobalStaticVariables::physical_time_ = 0.0;
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();

	/** apply initial condition */
	corrected_configuration.parallel_exec();
	write_states.writeToFile(0);

	/** Setup physical parameters. */
	int ite = 0;
	Real output_period = end_time / 200.0;
	Real dt = 0.0;

	/** Statistics for computing time. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	/**
	 * Main loop
	 */
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integration_time = 0.0;
		while (integration_time < output_period)
		{
			if (ite % 100 == 0)
			{
				std::cout << "N=" << ite << " Time: "
					<< GlobalStaticVariables::physical_time_ << "	dt: "
					<< dt << "\n";
			}
			
			beam_initialize.parallel_exec();
			pull_force.parallel_exec(GlobalStaticVariables::physical_time_);

			/** Stress relaxation and damping. */
			stress_relaxation_first_half.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			beam_damping.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			stress_relaxation_second_half.parallel_exec(dt);

			ite++;
			dt = system.getSmallestTimeStepAmongSolidBodies();
			integration_time += dt;
			GlobalStaticVariables::physical_time_ += dt;
		}
		tick_count t2 = tick_count::now();
		write_states.writeToFile();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}

	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	return 0;
}
