/**
 * @file 	ball_shell_collision.cpp
 * @brief 	an elastic ball bouncing within a confined shell boundary
 * @details This is a case to test elasticSolid -> shell impact/collision.
 * @author 	Massoud Rezavand, Virtonomy GmbH
 */
#include "sphinxsys.h" //SPHinXsys Library.
using namespace SPH;   // Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real resolution_ref = 0.025; /**< reference resolution. */
Real circle_radius = 2.0;
Vec2d circle_center(2.0, 2.0);
Real thickness = resolution_ref * 1.; /**< shell thickness. */
Real level_set_refinement_ratio = resolution_ref / (0.1 * thickness);
BoundingBox system_domain_bounds(Vec2d(-thickness, -thickness), Vec2d(2.0 * circle_radius + thickness, 2.0 * circle_radius + thickness));
Vec2d ball_center(3.0, 1.5);
Real ball_radius = 0.5;
StdVec<Vecd> beam_observation_location = {ball_center};
Real gravity_g = 1.0;
//----------------------------------------------------------------------
//	Global parameters on material properties
//----------------------------------------------------------------------
Real rho0_s = 1.0e3;
Real Youngs_modulus = 2.0e4;
Real poisson = 0.45;
Real physical_viscosity = 1.0e6;
//----------------------------------------------------------------------
//	Bodies with cases-dependent geometries (ComplexShape).
//----------------------------------------------------------------------
class WallBoundary : public ComplexShape
{
public:
	explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<GeometricShapeBall>(circle_center, circle_radius + resolution_ref);
		subtract<GeometricShapeBall>(circle_center, circle_radius);
	}
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	/** Tag for running particle relaxation for the initially body-fitted distribution */
	sph_system.run_particle_relaxation_ = true;
	/** Tag for starting with relaxed body-fitted particles distribution */
	sph_system.reload_particles_ = false;
	/** Tag for computation from restart files. 0: start with initial condition */
	sph_system.restart_step_ = 0;
	/** Handle command line arguments. */
	sph_system.handleCommandlineOptions(ac, av);
	/** I/O environment. */
	InOutput in_output(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody ball(sph_system, makeShared<GeometricShapeBall>(ball_center, ball_radius, "BallBody"));
	ball.defineParticlesAndMaterial<ElasticSolidParticles, NeoHookeanSolid>(rho0_s, Youngs_modulus, poisson);
	if (!sph_system.run_particle_relaxation_ && sph_system.reload_particles_)
	{
		ball.generateParticles<ParticleGeneratorReload>(in_output, ball.getBodyName());
	}
	else
	{
		ball.defineBodyLevelSetShape()->writeLevelSet(ball);
		ball.generateParticles<ParticleGeneratorLattice>();
	}

	// Note the wall boundary here has sharp corner, and is a numerical invalid elastic shell structure,
	// and its dynamics is not able to be modeled by the shell dynamics in SPHinXsys in the current version.
	// Here, we use it simply as a rigid shell.
	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
	wall_boundary.defineAdaptation<SPHAdaptation>(1.15, 1.0);
	// here dummy linear elastic solid is use because no solid dynamics in particle relaxation
	wall_boundary.defineParticlesAndMaterial<ShellParticles, LinearElasticSolid>(1.0, 1.0, 0.0);
	if (!sph_system.run_particle_relaxation_ && sph_system.reload_particles_)
	{
		wall_boundary.generateParticles<ParticleGeneratorReload>(in_output, wall_boundary.getBodyName());
	}
	else
	{
		wall_boundary.defineBodyLevelSetShape(level_set_refinement_ratio)->writeLevelSet(wall_boundary);
		wall_boundary.generateParticles<ThickSurfaceParticleGeneratorLattice>(thickness);
	}

	if (!sph_system.run_particle_relaxation_ && !sph_system.reload_particles_)
	{
		std::cout << "Error: This case requires reload shell particles for simulation!" << std::endl;
		return 0;
	}

	ObserverBody ball_observer(sph_system, "BallObserver");
	ball_observer.generateParticles<ObserverParticleGenerator>(beam_observation_location);
	//----------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//----------------------------------------------------------------------
	if (sph_system.run_particle_relaxation_)
	{
		//----------------------------------------------------------------------
		//	Define body relation map used for particle relaxation.
		//----------------------------------------------------------------------
		BodyRelationInner ball_inner(ball);
		BodyRelationInner wall_boundary_inner(wall_boundary);
		//----------------------------------------------------------------------
		//	Define the methods for particle relaxation for ball.
		//----------------------------------------------------------------------
		RandomizeParticlePosition ball_random_particles(ball);
		relax_dynamics::RelaxationStepInner ball_relaxation_step_inner(ball_inner);
		//----------------------------------------------------------------------
		//	Define the methods for particle relaxation for wall boundary.
		//----------------------------------------------------------------------
		RandomizeParticlePosition wall_boundary_random_particles(wall_boundary);
		relax_dynamics::ShellRelaxationStepInner
			relaxation_step_wall_boundary_inner(wall_boundary_inner, thickness, level_set_refinement_ratio);
		relax_dynamics::ShellNormalDirectionPrediction shell_normal_prediction(wall_boundary_inner, thickness, cos(Pi / 3.75));
		wall_boundary.addBodyStateForRecording<int>("UpdatedIndicator");
		//----------------------------------------------------------------------
		//	Output for particle relaxation.
		//----------------------------------------------------------------------
		BodyStatesRecordingToVtp write_relaxed_particles(in_output, sph_system.real_bodies_);
		MeshRecordingToPlt write_mesh_cell_linked_list(in_output, wall_boundary, wall_boundary.cell_linked_list_);
		ReloadParticleIO write_particle_reload_files(in_output, {&ball, &wall_boundary});
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		ball_random_particles.parallel_exec(0.25);
		wall_boundary_random_particles.parallel_exec(0.25);

		relaxation_step_wall_boundary_inner.mid_surface_bounding_.parallel_exec();
		write_relaxed_particles.writeToFile(0);
		wall_boundary.updateCellLinkedList();
		write_mesh_cell_linked_list.writeToFile(0);
		//----------------------------------------------------------------------
		//	From here iteration for particle relaxation begins.
		//----------------------------------------------------------------------
		int ite = 0;
		int relax_step = 1000;
		while (ite < relax_step)
		{
			ball_relaxation_step_inner.parallel_exec();
			for (int k = 0; k < 2; ++k)
				relaxation_step_wall_boundary_inner.parallel_exec();
			ite += 1;
			if (ite % 100 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
				write_relaxed_particles.writeToFile(ite);
			}
		}
		std::cout << "The physics relaxation process of ball particles finish !" << std::endl;
		shell_normal_prediction.exec();
		write_relaxed_particles.writeToFile(ite);
		write_particle_reload_files.writeToFile(0);
		return 0;
	}
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInner ball_inner(ball);
	SolidBodyRelationContact ball_contact(ball, {&wall_boundary});
	BodyRelationContact ball_observer_contact(ball_observer, {&ball});
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	/** Define external force.*/
	Gravity gravity(Vec2d(0.0, -gravity_g));
	TimeStepInitialization ball_initialize_timestep(ball, gravity);
	solid_dynamics::CorrectConfiguration ball_corrected_configuration(ball_inner);
	solid_dynamics::AcousticTimeStepSize ball_get_time_step_size(ball);
	/** stress relaxation for the balls. */
	solid_dynamics::StressRelaxationFirstHalf ball_stress_relaxation_first_half(ball_inner);
	solid_dynamics::StressRelaxationSecondHalf ball_stress_relaxation_second_half(ball_inner);
	/** Algorithms for solid-solid contact. */
	solid_dynamics::ShellContactDensity ball_update_contact_density(ball_contact);
	solid_dynamics::ContactForceFromWall ball_compute_solid_contact_forces(ball_contact);
	DampingWithRandomChoice<solid_dynamics::PairwiseFrictionFromWall> ball_friction(0.1, ball_contact, physical_viscosity);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp body_states_recording(in_output, sph_system.real_bodies_);
	RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
		write_ball_center_displacement("Position", in_output, ball_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	ball_corrected_configuration.parallel_exec();

	/** Initial states output. */
	body_states_recording.writeToFile(0);
	/** Main loop. */
	int ite = 0;
	Real T0 = 10.0;
	Real End_Time = T0;
	Real D_Time = 0.01 * T0;
	Real Dt = 0.1 * D_Time;
	Real dt = 0.0;
	//----------------------------------------------------------------------
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;
		while (integration_time < D_Time)
		{
			Real relaxation_time = 0.0;
			while (relaxation_time < Dt)
			{
				ball_initialize_timestep.parallel_exec();
				if (ite % 100 == 0)
				{
					std::cout << "N=" << ite << " Time: "
							  << GlobalStaticVariables::physical_time_ << "	dt: " << dt << "\n";
				}
				ball_update_contact_density.parallel_exec();
				ball_compute_solid_contact_forces.parallel_exec();
				ball_stress_relaxation_first_half.parallel_exec(dt);
				ball_friction.parallel_exec(dt);
				ball_stress_relaxation_second_half.parallel_exec(dt);

				ball.updateCellLinkedList();
				ball_contact.updateConfiguration();

				ite++;
				Real dt_ball = ball_get_time_step_size.parallel_exec();
				dt = dt_ball;
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}

			write_ball_center_displacement.writeToFile(ite);
		}
		tick_count t2 = tick_count::now();
		body_states_recording.writeToFile(ite);
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	if (sph_system.generate_regression_data_)
	{
		write_ball_center_displacement.generateDataBase(1.0e-2);
	}
	else
	{
		write_ball_center_displacement.newResultTest();
	}

	return 0;
}
