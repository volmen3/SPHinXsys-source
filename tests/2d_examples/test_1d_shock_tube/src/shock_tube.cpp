/**
 * @file 	shock_tube.cpp
 * @brief 	This is a test to show the standard Sod shock tube case.
 * @details See https://doi.org/10.1016/j.jcp.2010.08.019 for the detailed problem setup.
 * @author 	Zhentong Wang and Xiangyu Hu
 */
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 5.0;							 /**< Tube length. */
Real particle_spacing_ref = 1.0 / 200.0; /**< Initial reference particle spacing. */
Real DH = particle_spacing_ref * 4;		 /**< Tube height. */
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-2.0 / 5.0 * DL, 0.0), Vec2d(3.0 / 5.0 * DL, DH));
Real rho0_l = 1.0;	  /**< initial density of left state. */
Real rho0_r = 0.125;  /**< initial density of right state. */
Vecd velocity_l(0.0); /**< initial velocity of left state. */
Vecd velocity_r(0.0); /**< initial velocity of right state. */
Real p_l = 1.0;		  /**< initial pressure of left state. */
Real p_r = 0.1;		  /**< initial pressure of right state. */
//----------------------------------------------------------------------
//	Global parameters on material properties
//----------------------------------------------------------------------
Real heat_capacity_ratio = 1.4; /**< heat capacity ratio. */
//----------------------------------------------------------------------
//	Cases-dependent geometry
//----------------------------------------------------------------------
class WaveBlock : public MultiPolygonShape
{
public:
	explicit WaveBlock(const std::string &body_name)
		: MultiPolygonShape(body_name)
	{
		std::vector<Vecd> waves_block_shape{
			Vecd(-2.0 / 5.0 * DL, 0.0), Vecd(-2.0 / 5.0 * DL, DH), Vecd(3.0 / 5.0 * DL, DH),
			Vecd(3.0 / 5.0 * DL, 0.0), Vecd(-2.0 / 5.0 * DL, 0.0)};
		multi_polygon_.addAPolygon(waves_block_shape, ShapeBooleanOps::add);
	}
};
//----------------------------------------------------------------------
//	Case-dependent initial condition.
//----------------------------------------------------------------------
class WavesInitialCondition
	: public eulerian_compressible_fluid_dynamics::CompressibleFluidInitialCondition
{
public:
	explicit WavesInitialCondition(EulerianFluidBody &water)
		: eulerian_compressible_fluid_dynamics::CompressibleFluidInitialCondition(water){};

protected:
	void Update(size_t index_i, Real dt) override
	{
		if (pos_[index_i][0] < DL / 10.0)
		{
			// initial left state pressure,momentum and energy profile
			rho_[index_i] = rho0_l;
			p_[index_i] = p_l;
			Real rho_e = p_[index_i] / (gamma_ - 1.0);
			vel_[index_i] = velocity_l;
			mom_[index_i] = rho0_l * velocity_l;
			E_[index_i] = rho_e + 0.5 * rho_[index_i] * vel_[index_i].normSqr();
		}
		if (pos_[index_i][0] > DL / 10.0)
		{
			// initial right state pressure,momentum and energy profile
			rho_[index_i] = rho0_r;
			p_[index_i] = p_r;
			Real rho_e = p_[index_i] / (gamma_ - 1.0);
			vel_[index_i] = velocity_r;
			mom_[index_i] = rho0_r * velocity_r;
			E_[index_i] = rho_e + 0.5 * rho_[index_i] * vel_[index_i].normSqr();
		}
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
	SPHSystem sph_system(system_domain_bounds, particle_spacing_ref);
	// handle command line arguments
	sph_system.handleCommandlineOptions(ac, av);
	// output environment.
	InOutput in_output(sph_system);
	//----------------------------------------------------------------------
	//	Create body, materials and particles.
	//----------------------------------------------------------------------
	EulerianFluidBody wave_body(sph_system, makeShared<WaveBlock>("WaveBody"));
	wave_body.defineParticlesAndMaterial<CompressibleFluidParticles, CompressibleFluid>(rho0_l, heat_capacity_ratio);
	wave_body.generateParticles<ParticleGeneratorLattice>();
	wave_body.addBodyStateForRecording<Real>("TotalEnergy");
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The inner relation defines the particle configuration for particles within a body.
	//	The contact relation defines the particle configuration between the bodies.
	//----------------------------------------------------------------------
	BodyRelationInner wave_body_inner(wave_body);
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	WavesInitialCondition waves_initial_condition(wave_body);
	// Initialize particle acceleration.
	eulerian_compressible_fluid_dynamics::CompressibleFlowTimeStepInitialization initialize_wave_step(wave_body);
	// Periodic BCs in y direction.
	PeriodicConditionUsingCellLinkedList periodic_condition_y(wave_body, wave_body.getBodyShapeBounds(), yAxis);
	// Time step size with considering sound wave speed.
	eulerian_compressible_fluid_dynamics::AcousticTimeStepSize get_wave_time_step_size(wave_body);
	// Pressure, density and energy relaxation algorithm by use HLLC Riemann solver.
	eulerian_compressible_fluid_dynamics::PressureRelaxationHLLCRiemannInner pressure_relaxation(wave_body_inner);
	eulerian_compressible_fluid_dynamics::DensityAndEnergyRelaxationHLLCRiemannInner density_and_energy_relaxation(wave_body_inner);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations, observations of the simulation.
	//	Regression tests are also defined here.
	//----------------------------------------------------------------------
	BodyStatesRecordingToPlt body_states_recording(in_output, sph_system.real_bodies_);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	waves_initial_condition.exec();
	sph_system.initializeSystemCellLinkedLists();
	periodic_condition_y.update_cell_linked_list_.parallel_exec();
	sph_system.initializeSystemConfigurations();
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	size_t number_of_iterations = sph_system.restart_step_;
	int screen_output_interval = 100;
	Real End_Time = 0.2; /**< End time. */
	Real D_Time = 0.01;	 /**< Time stamps for output of body states. */
	//----------------------------------------------------------------------
	// Output the start states of bodies.
	//----------------------------------------------------------------------
	body_states_recording.writeToFile(0);
	//----------------------------------------------------------------------
	//	Statistics for computing CPU time.
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;
		//	Integrate time (loop) until the next output time.
		while (integration_time < D_Time)
		{
			initialize_wave_step.parallel_exec();
			Real dt = get_wave_time_step_size.parallel_exec();
			// Dynamics including pressure and density and energy relaxation.
			integration_time += dt;
			pressure_relaxation.parallel_exec(dt);
			density_and_energy_relaxation.parallel_exec(dt);
			GlobalStaticVariables::physical_time_ += dt;

			if (number_of_iterations % screen_output_interval == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
						  << GlobalStaticVariables::physical_time_
						  << "	dt = " << dt << "\n";
			}
			number_of_iterations++;
		}

		tick_count t2 = tick_count::now();
		body_states_recording.writeToFile();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	cout << "Total wall time for computation: " << tt.seconds() << " seconds." << endl;

	return 0;
}
