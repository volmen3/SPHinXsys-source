/**
 * @file 	muscle_soft_body_contact.cpp
 * @brief 	This is the test for muscle compression with our new contact model. 
 * Different particle resolutions are used for the two soft bodies that are in contact.
 * @author 	Chi Zhang, Bence Rochlitz and Xiangyu Hu
 */
#include "sphinxsys.h"
/** Name space. */
using namespace SPH;
/** Geometry parameters. */
Real L = 0.04;
Real PL = 0.1;
Real resolution_ref = L / 12.0;
Real BW = resolution_ref * 4;
Vecd halfsize_myocardium(0.5 * L, 0.5 * L, 0.5 * L);
Vecd translation_myocardium(0.5 * L, 0.0, 0.0);
Vecd halfsize_stationary_plate(0.5 * BW, 0.5 * L + BW, 0.5 * L + BW);
Vecd translation_stationary_plate(-0.5 * BW, 0.0, 0.0);
Vecd halfsize_moving_plate(0.5 * BW, 0.5 * PL, 0.5 * PL);
Vecd translation_moving_plate(L + BW, 0.0, 0.0);
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-BW, -0.5 * PL, -0.5 * PL),
								 Vecd(2.0 * L + BW, 0.5 * PL, 0.5 * PL));

/** For material properties of the solid. */
Real rho0_s = 1265.0;
Real poisson = 0.45;
Real Youngs_modulus = 5e4;
Real physical_viscosity = 200.0;
/** Define the myocardium body shape. */
class Myocardium : public ComplexShape
{
public:
	explicit Myocardium(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(translation_myocardium, halfsize_myocardium);
		add<TransformShape<GeometricShapeBox>>(translation_stationary_plate, halfsize_stationary_plate);
	}
};
/**
* @brief define the moving plate shape
*/
class MovingPlate : public ComplexShape
{
public:
	explicit MovingPlate(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(translation_moving_plate, halfsize_moving_plate);
	}
};
/**
 *  The main program
 */
int main()
{
	/** Setup the system. Please the make sure the global domain bounds are correctly defined. */
	SPHSystem system(system_domain_bounds, resolution_ref);
	/** Creat a Myocardium body, corresponding material, particles and reaction model. */
	SolidBody myocardium_body(system, makeShared<Myocardium>("MyocardiumBody"));
	myocardium_body.defineParticlesAndMaterial<ElasticSolidParticles, NeoHookeanSolid>(rho0_s, Youngs_modulus, poisson);
	myocardium_body.generateParticles<ParticleGeneratorLattice>();
	/** Plate. */
	SolidBody moving_plate(system, makeShared<MovingPlate>("MovingPlate"));
	moving_plate.sph_adaptation_->resetAdaptationRatios(1.15, 1.5);
	moving_plate.defineParticlesAndMaterial<ElasticSolidParticles, NeoHookeanSolid>(rho0_s, Youngs_modulus, poisson);
	moving_plate.generateParticles<ParticleGeneratorLattice>();
	/** topology */
	BodyRelationInner myocardium_body_inner(myocardium_body);
	BodyRelationInner moving_plate_inner(moving_plate);
	SolidBodyRelationContact myocardium_plate_contact(myocardium_body, {&moving_plate});
	SolidBodyRelationContact plate_myocardium_contact(moving_plate, {&myocardium_body});
	/** 
	 * This section define all numerical methods will be used in this case.
	 */
	/** initialize a time step */
	TimeStepInitialization myocardium_initialize_time_step(myocardium_body);
	Gravity gravity(Vecd(-100.0, 0.0, 0.0));
	TimeStepInitialization plate_initialize_time_step(moving_plate, gravity);
	/** Corrected configuration. */
	solid_dynamics::CorrectConfiguration corrected_configuration(myocardium_body_inner);
	solid_dynamics::CorrectConfiguration corrected_configuration_2(moving_plate_inner);
	/** active and passive stress relaxation. */
	solid_dynamics::KirchhoffStressRelaxationFirstHalf stress_relaxation_first_half(myocardium_body_inner);
	solid_dynamics::StressRelaxationSecondHalf stress_relaxation_second_half(myocardium_body_inner);
	solid_dynamics::KirchhoffStressRelaxationFirstHalf stress_relaxation_first_half_2(moving_plate_inner);
	solid_dynamics::StressRelaxationSecondHalf stress_relaxation_second_half_2(moving_plate_inner);
	//stress_relaxation_first_half_2.post_processes_(spring_constraint);
	/** Algorithms for solid-solid contact. */
	solid_dynamics::ContactDensitySummation myocardium_update_contact_density(myocardium_plate_contact);
	solid_dynamics::ContactDensitySummation plate_update_contact_density(plate_myocardium_contact);
	solid_dynamics::ContactForce myocardium_compute_solid_contact_forces(myocardium_plate_contact);
	solid_dynamics::ContactForce plate_compute_solid_contact_forces(plate_myocardium_contact);

	/** Constrain the holder. */
	BodyRegionByParticle holder(myocardium_body, 
		makeShared<TransformShape<GeometricShapeBox>>(translation_stationary_plate, halfsize_stationary_plate, "Holder"));
	solid_dynamics::ConstrainSolidBodyRegion constrain_holder(myocardium_body, holder);
	/** Add spring constraint on the plate. */
	solid_dynamics::SpringDamperConstraintParticleWise spring_constraint(moving_plate, Vecd(0.2, 0, 0), 0.01);

	/** Damping with the solid body*/
	DampingWithRandomChoice<DampingPairwiseInner<Vec3d>>
		muscle_damping(0.2, myocardium_body_inner, "Velocity", physical_viscosity);
	DampingWithRandomChoice<DampingPairwiseInner<Vec3d>>
		plate_damping(0.2, moving_plate_inner, "Velocity", physical_viscosity);
	/** Output */
	InOutput in_output(system);
	BodyStatesRecordingToVtp write_states(in_output, system.real_bodies_);

	/**
	 * From here the time stepping begins.
	 * Set the starting time.
	 */
	GlobalStaticVariables::physical_time_ = 0.0;
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	/** apply initial condition */
	corrected_configuration.parallel_exec();
	corrected_configuration_2.parallel_exec();
	write_states.writeToFile(0);
	/** Setup physical parameters. */
	int ite = 0;
	Real end_time = 0.1;
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
			/** Gravity. */
			myocardium_initialize_time_step.parallel_exec();
			plate_initialize_time_step.parallel_exec();

			spring_constraint.parallel_exec();

			/** Contact model for myocardium. */
			myocardium_update_contact_density.parallel_exec();
			myocardium_compute_solid_contact_forces.parallel_exec();
			/** Contact model for plate. */
			plate_update_contact_density.parallel_exec();
			plate_compute_solid_contact_forces.parallel_exec();

			/** Stress relaxation and damping. */
			stress_relaxation_first_half.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			muscle_damping.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			stress_relaxation_second_half.parallel_exec(dt);

			stress_relaxation_first_half_2.parallel_exec(dt);
			plate_damping.parallel_exec(dt);
			stress_relaxation_second_half_2.parallel_exec(dt);

			ite++;
			dt = system.getSmallestTimeStepAmongSolidBodies();
			integration_time += dt;
			GlobalStaticVariables::physical_time_ += dt;

			myocardium_body.updateCellLinkedList();
			moving_plate.updateCellLinkedList();

			myocardium_plate_contact.updateConfiguration();
			plate_myocardium_contact.updateConfiguration();
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
