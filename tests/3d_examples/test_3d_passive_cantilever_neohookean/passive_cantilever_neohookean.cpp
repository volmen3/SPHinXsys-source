/**
 * @file passive_cantilever_neohookean.cpp
 * @brief This is the example of cantilever with simple neohookean tissue model
 * @author Bence Rochlitz, Chi Zhang  and Xiangyu Hu
 * @ref 	doi.org/10.1016/j.jcp.2013.12.012
 */
#include "sphinxsys.h"
/** Name space. */
using namespace SPH;
/** Geometry parameters. */
Real PL = 0.1;
Real PH = 0.04;
Real PW = 0.04;
Real SL = 0.02;
Real resolution_ref = PH / 6.0; /**< Initial particle spacing. */
Real BW = resolution_ref * 4;	/**< Boundary width. */
Vecd halfsize_cantilever(0.5 * (PL + SL), 0.5 * PH, 0.5 * PW);
Vecd translation_cantilever(0.5 * (PL - SL), 0.5 * PH, 0.5 * PW);
Vecd halfsize_holder(0.5 * SL, 0.5 * PH, 0.5 * PW);
Vecd translation_holder(-0.5 * SL, 0.5 * PH, 0.5 * PW);
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-SL, 0, 0), Vecd(PL, PH, PH));
// Observer location
StdVec<Vecd> observation_location = {Vecd(PL, PH, PW)};
/** For material properties of the solid. */
Real rho0_s = 1265.0;			// Gheorghe 2019
Real poisson = 0.45;			// nearly incompressible
Real Youngs_modulus = 5e4;		// Sommer 2015
Real physical_viscosity = 50.0; // physical damping, here we choose the same value as numerical viscosity
Real gravity_g = 9.8;			/**< Value of gravity. */
Real time_to_full_gravity = 0.0;

/** Define the cantilever body. */
class Cantilever : public ComplexShape
{
public:
	explicit Cantilever(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(translation_cantilever, halfsize_cantilever);
		add<TransformShape<GeometricShapeBox>>(translation_holder, halfsize_holder);
	}
};
/**
 * define time dependent gravity
 */
class TimeDependentGravity : public Gravity
{
public:
	explicit TimeDependentGravity(Vecd gravity_vector)
		: Gravity(gravity_vector) {}
	virtual Vecd InducedAcceleration(Vecd &position) override
	{
		Real current_time = GlobalStaticVariables::physical_time_;
		return current_time < time_to_full_gravity ? current_time * global_acceleration_ / time_to_full_gravity : global_acceleration_;
	}
};
/**
 *  The main program
 */
int main(int ac, char *av[])
{
	/** Setup the system. */
	SPHSystem system(system_domain_bounds, resolution_ref);
// handle command line arguments
#ifdef BOOST_AVAILABLE
	system.handleCommandlineOptions(ac, av);
#endif	/** output environment. */

	/** create a Cantilever body, corresponding material, particles and reaction model. */
	SolidBody cantilever_body(system, makeShared<Cantilever>("CantileverBody"));
	cantilever_body.defineParticlesAndMaterial<ElasticSolidParticles, NeoHookeanSolid>(rho0_s, Youngs_modulus, poisson);
	cantilever_body.generateParticles<ParticleGeneratorLattice>();
	/** Define Observer. */
	ObserverBody cantilever_observer(system, "CantileverObserver");
	cantilever_observer.generateParticles<ObserverParticleGenerator>(observation_location);

	/** topology */
	BodyRelationInner cantilever_body_inner(cantilever_body);
	BodyRelationContact cantilever_observer_contact(cantilever_observer, {&cantilever_body});

	//-------- common particle dynamics ----------------------------------------
	TimeDependentGravity gravity(Vec3d(0.0, -gravity_g, 0.0));
	TimeStepInitialization initialize_time_step(cantilever_body, gravity);

	/**
	 * This section define all numerical methods will be used in this case.
	 */
	/** Corrected configuration. */
	solid_dynamics::CorrectConfiguration
		corrected_configuration(cantilever_body_inner);
	/** Time step size calculation. */
	solid_dynamics::AcousticTimeStepSize
		computing_time_step_size(cantilever_body);
	/** active and passive stress relaxation. */
	solid_dynamics::StressRelaxationFirstHalf
		stress_relaxation_first_half(cantilever_body_inner);
	/** Setup the damping stress, if you know what you are doing. */
	// stress_relaxation_first_step.setupDampingStressFactor(1.0);
	solid_dynamics::StressRelaxationSecondHalf
		stress_relaxation_second_half(cantilever_body_inner);
	/** Constrain the holder. */
	BodyRegionByParticle holder(cantilever_body,
								makeShared<TransformShape<GeometricShapeBox>>(translation_holder, halfsize_holder, "Holder"));
	solid_dynamics::ConstrainSolidBodyRegion constrain_holder(cantilever_body, holder);
	DampingWithRandomChoice<DampingBySplittingInner<Vec3d>>
		muscle_damping(0.1, cantilever_body_inner, "Velocity", physical_viscosity);
	/** Output */
	InOutput in_output(system);
	BodyStatesRecordingToVtp write_states(in_output, system.real_bodies_);
	RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
		write_displacement("Position", in_output, cantilever_observer_contact);
	/**
	 * From here the time stepping begins.
	 * Set the starting time.
	 */
	GlobalStaticVariables::physical_time_ = 0.0;
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	corrected_configuration.parallel_exec();
	write_states.writeToFile(0);
	write_displacement.writeToFile(0);
	/** Setup physical parameters. */
	int ite = 0;
	Real end_time = 1.0;
	Real output_period = end_time / 100.0;
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

			initialize_time_step.parallel_exec(); // gravity force
			stress_relaxation_first_half.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			muscle_damping.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			stress_relaxation_second_half.parallel_exec(dt);

			ite++;
			dt = computing_time_step_size.parallel_exec();
			integration_time += dt;
			GlobalStaticVariables::physical_time_ += dt;
		}
		write_displacement.writeToFile(ite);
		tick_count t2 = tick_count::now();
		write_states.writeToFile();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	if (system.generate_regression_data_)
	{
		write_displacement.generateDataBase(1.0e-2);
	}
	else
	{
		write_displacement.newResultTest();
	}

	return 0;
}
