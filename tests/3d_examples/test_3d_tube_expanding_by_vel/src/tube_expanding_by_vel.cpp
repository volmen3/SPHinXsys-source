/**
 * @file 	tube_expanding_by_velocity.cpp
 * @brief 	This is the test for expanding a tube by the velocity on inside surface.
 * @author 	Anyong Zhang
 */
#include "sphinxsys.h"
 /** Name space. */
using namespace SPH;

/** particle resolution */
Real resolution_ref = 0.00015;

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-.002, -0.005, -0.005),
	Vecd(0.062, 0.005, 0.005));

/** Physical parameters */
Real rho = 8000.0;  // kg/m^3
Real poisson_ratio = 0.33;
Real Youngs_modulus = 193000; // P
Real physical_viscosity = 1000.;


/**
* @brief define the tube body
*/
class Tube : public SolidBody
{
public:
	Tube(SPHSystem& system, const std::string& body_name)
		: SolidBody(system, body_name)
	{
		string fname_ = "./input/tube.stl";
		TriangleMeshShapeSTL stl_shape(fname_, Vecd(0), 0.001);
		body_shape_.add<LevelSetShape>(this, stl_shape, true);
	}

};


class ExpandingByVel : public PartSimpleDynamicsByParticle, public solid_dynamics::SolidDataSimple
{
public:
	ExpandingByVel(SPHBody& sph_body, BodyPartByParticle& body_part, StdVec<array<Real, 2>> vel_arr)
		:PartSimpleDynamicsByParticle(sph_body, body_part),
		solid_dynamics::SolidDataSimple(sph_body),
		pos_0_(particles_->pos_0_),
		pos_n_(particles_->pos_n_),
		vel_n_(particles_->vel_n_),
		vel_array_(vel_arr)
	{}


protected:
	StdLargeVec<Vecd>& pos_0_;
	StdLargeVec<Vecd>& pos_n_;
	StdLargeVec<Vecd>& vel_n_;
	StdVec<array<Real, 2>> vel_array_;

protected:
	/** linear interpolation for velocity-time profile */
	virtual Real getVel(Real time)
	{
		for (size_t i = 1; i < vel_array_.size(); i++)
		{
			if (time >= vel_array_[i - 1][0] && time < vel_array_[i][0])
			{
				Real slope = (vel_array_[i][1] - vel_array_[i - 1][1]) / (vel_array_[i][0] - vel_array_[i - 1][0]);
				Real vel = (time - vel_array_[i - 1][0]) * slope + vel_array_[i - 1][1];
				return vel;
			}
			else if (time > vel_array_.back()[0])
				return 0;
		}
	}

	virtual void Update(size_t index_i, Real time = 0)
	{
		/* centeral axis of tube is x-axis */
		Vecd radial_direction = pos_0_[index_i] - Vecd(pos_0_[index_i][0], 0, 0);
		radial_direction = radial_direction.normalize();
		Real vel = getVel(time);

		vel_n_[index_i][1] = vel * radial_direction[1];
		vel_n_[index_i][2] = vel * radial_direction[2];
	}
};


/**
 *  The main program
 */
int main()
{
	/** Setup the system */
	SPHSystem system(system_domain_bounds, resolution_ref);
	system.restart_step_ = 0;
	GlobalStaticVariables::physical_time_ = 0;
	In_Output in_output(system);

	bool relaxation_button = false; // relax particles firstly.
	if (relaxation_button)
	{
		system.run_particle_relaxation_ = true;
		system.reload_particles_ = false;
	}
	else
	{
		system.run_particle_relaxation_ = false;
		system.reload_particles_ = true;
	}


	/** Import a tube body, corresponding material, particles. */
	Tube tube_body(system, "TubeBody");
	SharedPtr<ParticleGenerator> tube_particle_generator = makeShared<ParticleGeneratorLattice>();
	if (!system.run_particle_relaxation_ && system.reload_particles_)
		tube_particle_generator = makeShared<ParticleGeneratorReload>(in_output, tube_body.getBodyName());
	ElasticSolidParticles tube_body_particles(tube_body,
		makeShared<LinearElasticSolid>(rho, Youngs_modulus, poisson_ratio),
		tube_particle_generator);

	/*Run particle relaxation for all solid body */
	StdVec<SolidBody*> all_body{ &tube_body };
	if (system.run_particle_relaxation_)
	{
		for (size_t i = 0; i < all_body.size(); i++)
		{
			SolidBody* curr_body_ptr = all_body[i];
			auto& curr_body = *curr_body_ptr;
			BodyRelationInner body_inner(curr_body);
			RandomizePartilePosition random_inserted_body_particles(curr_body);
			//BodyStatesRecordingToVtp
			ReloadParticleIO write_particle_reload_files(in_output, { curr_body_ptr });
			relax_dynamics::RelaxationStepInner relaxation_step_inner(body_inner);
			random_inserted_body_particles.parallel_exec(0.25);
			relaxation_step_inner.surface_bounding_.parallel_exec();
			int ite_p = 0;
			while (ite_p < 1000)
			{
				relaxation_step_inner.parallel_exec();
				++ite_p;
				if (ite_p % 200 == 0)
					cout << fixed << setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
			}
			cout << curr_body.getBodyName() << " particles relaxation finish !\n";
			write_particle_reload_files.writeToFile(0);
		}
		cout << "all solid body particles relaxation finish !\n";
		return 0;
	}

	/** topology */
	BodyRelationInner beam_body_inner(tube_body);

	//-------- common particle dynamics ----------------------------------------
	TimeStepInitialization tube_initialize(tube_body);

	/**
	 * This section define all numerical methods will be used in this case.
	 */
	 /** Corrected configuration. */
	solid_dynamics::CorrectConfiguration corrected_configuration(beam_body_inner);
	/** Time step size calculation. */
	solid_dynamics::AcousticTimeStepSize computing_time_step_size(tube_body);
	/** active and passive stress relaxation. */
	//solid_dynamics::StressRelaxationFirstHalf stress_relaxation_first_half(beam_body_inner);
	solid_dynamics::KirchhoffParticleStressRelaxationFirstHalf stress_relaxation_first_half(beam_body_inner);
	//solid_dynamics::KirchhoffStressRelaxationFirstHalf stress_relaxation_first_half(beam_body_inner);
	
	solid_dynamics::StressRelaxationSecondHalf stress_relaxation_second_half(beam_body_inner);

	/* define velocity-time profile */
	Real end_time = 0.1;
	StdVec<array<Real, 2>> vel_over_time = { {0, 0},{0.015,0.04},{0.03,0},{end_time,0} };
	
	/* identify inside surface */
	Real R_in = 0.0015;			// inside raduis = 1.5 mm
	Real half_length = 0.025;
	/* create a cylinder to tag the inside surface particles */
	TriangleMeshShapeCylinder shape_cylinder(SimTK::UnitVec3(1, 0, 0),
		(R_in + resolution_ref), half_length, 1, Vecd(0.03, 0, 0));
	BodyRegionByParticle inside_surface(tube_body, "inside_surf", shape_cylinder);

	ExpandingByVel expanding(tube_body, inside_surface, vel_over_time);
	/** Damping with the solid body*/
	DampingWithRandomChoice<DampingPairwiseInner<Vec3d>>
		tube_damping(beam_body_inner, 0.1, "Velocity", physical_viscosity);
	/** Output */
	BodyStatesRecordingToVtp write_states(in_output, system.real_bodies_);

	/**
	 * From here the time stepping begines.
	 * Set the starting time.
	 */
	GlobalStaticVariables::physical_time_ = 0.0;
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
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

			tube_initialize.parallel_exec();

			Real time = GlobalStaticVariables::physical_time_;
			expanding.parallel_exec(time);
			stress_relaxation_first_half.parallel_exec(dt);

			tube_damping.parallel_exec(dt);

			expanding.parallel_exec(time);
			stress_relaxation_second_half.parallel_exec(dt);

			ite++;
			dt = computing_time_step_size.parallel_exec();
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
