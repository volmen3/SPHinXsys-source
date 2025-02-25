/**
 * @file 	sliding.cpp
 * @brief 	a 2D elastic cube slides on a rigid slope.
 * @details This is the a case for test collision dynamics for
 * 			understanding SPH method for complex simulation.
 * @author 	chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //	SPHinXsys Library.
using namespace SPH;   //	Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 20.0; /**< box length. */
Real DH = 13.0; /**< box height. */
Real L = 1.0;
Real slop_h = 11.55;
Real resolution_ref = L / 10.0; /**< reference particle spacing. */
Real BW = resolution_ref * 4;	/**< wall width for BCs. */
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(25, 15));
// Observer location
StdVec<Vecd> observation_location = {Vecd(7.2, 9.8)};
//----------------------------------------------------------------------
//	Global parameters on material properties
//----------------------------------------------------------------------
Real rho0_s = 1.0e3;
Real Youngs_modulus = 5.0e5;
Real poisson = 0.45;
Real gravity_g = 9.8;
Real physical_viscosity = 1000000.0;
//----------------------------------------------------------------------
//	Cases-dependent geometries
//----------------------------------------------------------------------
class WallBoundary : public MultiPolygonShape
{
public:
	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		/** Geometry definition. */
		std::vector<Vecd> wall_shape{Vecd(0, 0), Vecd(0, slop_h), Vecd(DL, slop_h), Vecd(0, 0)};
		multi_polygon_.addAPolygon(wall_shape, ShapeBooleanOps::add);
	}
};

class Cube : public MultiPolygonShape
{
public:
	explicit Cube(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		/** Geometry definition. */
		std::vector<Vecd> cubic_shape;
		cubic_shape.push_back(Vecd(BW, slop_h + resolution_ref));
		cubic_shape.push_back(Vecd(BW, slop_h + L + resolution_ref));
		cubic_shape.push_back(Vecd(BW + L, slop_h + L + resolution_ref));
		cubic_shape.push_back(Vecd(BW + L, slop_h + resolution_ref));
		cubic_shape.push_back(Vecd(BW, slop_h + resolution_ref));
		multi_polygon_.addAPolygon(cubic_shape, ShapeBooleanOps::add);
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
// handle command line arguments
#ifdef BOOST_AVAILABLE
	sph_system.handleCommandlineOptions(ac, av);
#endif	/** output environment. */
	InOutput in_output(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles
	//----------------------------------------------------------------------
	SolidBody free_cube(sph_system, makeShared<Cube>("FreeCube"));
	free_cube.defineParticlesAndMaterial<ElasticSolidParticles, LinearElasticSolid>(rho0_s, Youngs_modulus, poisson);
	free_cube.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("Wall"));
	wall_boundary.defineParticlesAndMaterial<SolidParticles, LinearElasticSolid>(rho0_s, Youngs_modulus, poisson);
	wall_boundary.generateParticles<ParticleGeneratorLattice>();

	ObserverBody cube_observer(sph_system, "CubeObserver");
	cube_observer.generateParticles<ObserverParticleGenerator>(observation_location);
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInner free_cube_inner(free_cube);
	SolidBodyRelationContact free_cube_contact(free_cube, {&wall_boundary});
	BodyRelationContact cube_observer_contact(cube_observer, {&free_cube});
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	Gravity gravity(Vecd(0.0, -gravity_g));
	Transform2d transform2d(Rotation2d(-0.5235));
	SimpleDynamics<TranslationAndRotation> wall_boundary_rotation(wall_boundary, transform2d);
	SimpleDynamics<TranslationAndRotation> free_cube_rotation(free_cube, transform2d);
	TimeStepInitialization free_cube_initialize_timestep(free_cube, gravity);
	/** Kernel correction. */
	solid_dynamics::CorrectConfiguration free_cube_corrected_configuration(free_cube_inner);
	/** Time step size. */
	solid_dynamics::AcousticTimeStepSize free_cube_get_time_step_size(free_cube);
	/** stress relaxation for the solid body. */
	solid_dynamics::StressRelaxationFirstHalf free_cube_stress_relaxation_first_half(free_cube_inner);
	solid_dynamics::StressRelaxationSecondHalf free_cube_stress_relaxation_second_half(free_cube_inner);
	/** Algorithms for solid-solid contact. */
	solid_dynamics::ContactDensitySummation free_cube_update_contact_density(free_cube_contact);
	solid_dynamics::ContactForceFromWall free_cube_compute_solid_contact_forces(free_cube_contact);
	/** Damping*/
	DampingWithRandomChoice<DampingPairwiseInner<Vec2d>>
		damping(0.5, free_cube_inner,"Velocity", physical_viscosity);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	/** Output the body states. */
	BodyStatesRecordingToVtp body_states_recording(in_output, sph_system.real_bodies_);
	/** Observer and output. */
	RegressionTestEnsembleAveraged<ObservedQuantityRecording<Vecd>>
		write_free_cube_displacement("Position", in_output, cube_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	GlobalStaticVariables::physical_time_ = 0.0;
	wall_boundary_rotation.parallel_exec();
	free_cube_rotation.parallel_exec();
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	free_cube_corrected_configuration.parallel_exec();
	//----------------------------------------------------------------------
	//	Initial states output.
	//----------------------------------------------------------------------
	body_states_recording.writeToFile(0);
	write_free_cube_displacement.writeToFile(0);
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int ite = 0;
	Real T0 = 2.5;
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
				free_cube_initialize_timestep.parallel_exec();
				if (ite % 100 == 0)
				{
					std::cout << "N=" << ite << " Time: "
							  << GlobalStaticVariables::physical_time_ << "	dt: " << dt
							  << "\n";
				}
				free_cube_update_contact_density.parallel_exec();
				free_cube_compute_solid_contact_forces.parallel_exec();
				free_cube_stress_relaxation_first_half.parallel_exec(dt);
				free_cube_stress_relaxation_second_half.parallel_exec(dt);

				free_cube.updateCellLinkedList();
				free_cube_contact.updateConfiguration();

				ite++;
				dt = free_cube_get_time_step_size.parallel_exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}
			write_free_cube_displacement.writeToFile(ite);
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
		// The lift force at the cylinder is very small and not important in this case.
		write_free_cube_displacement.generateDataBase({1.0e-2, 1.0e-2}, {1.0e-2, 1.0e-2});
	}
	else
	{
	write_free_cube_displacement.newResultTest();
	}

	return 0;
}
