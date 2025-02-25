/**
 * @file 	2d_plate.cpp
 * @brief 	This is the benchmark test of the shell.
 * @details  We consider point force force apply on a 2D plate.
 * @author 	Dong Wu, Chi Zhang and Xiangyu Hu
 */

#include "sphinxsys.h"

using namespace SPH;

//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real PL = 10.0;									  /** Length of the square plate. */
Vec2d n_0 = Vec2d(0.0, 1.0);					  /** Pseudo-normal. */
Real thickness = 1.0;							  /** Thickness of the square plate. */
int particle_number = 40;						  /** Particle number in the direction of the length */
Real resolution_ref = PL / (Real)particle_number; /** Initial reference particle spacing. */
int BWD = 1;									  /** number of boundary particles layers . */
Real BW = resolution_ref * (Real)BWD;			  /** Boundary width, determined by specific layer of boundary particles. */
BoundingBox system_domain_bounds(Vec2d(-BW, -0.5 * resolution_ref), Vec2d(PL + BW, 0.5 * resolution_ref));
StdVec<Vecd> observation_location = {Vecd(0.5 * PL, 0.0)};
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_s = 1.0;				   /** Normalized density. */
Real Youngs_modulus = 1.3024653e6; /** Normalized Youngs Modulus. */
Real poisson = 0.3;				   /** Poisson ratio. */
Real physical_viscosity = 400.0;   /** physical damping, here we choose the same value as numerical viscosity. */
//----------------------------------------------------------------------
//	Point forces properties
//----------------------------------------------------------------------
Real F_full = 50.0e3;
std::vector<Vecd> point_force{Vec2d(0.0, F_full)};
std::vector<Vecd> reference_position{Vec2d(0.5 * PL, 0.0)};
Real time_to_full_external_force = 0.05;
//----------------------------------------------------------------------
//	Derived classes used in the case
//----------------------------------------------------------------------
class PlateParticleGenerator : public SurfaceParticleGenerator
{
public:
	explicit PlateParticleGenerator(SPHBody &sph_body) : SurfaceParticleGenerator(sph_body){};
	virtual void initializeGeometricVariables() override
	{
		// the plate and boundary
		for (int i = 0; i < (particle_number + 2 * BWD); i++)
		{
			Real x = resolution_ref * i - BW + resolution_ref * 0.5;
			initializePositionAndVolumetricMeasure(Vecd(x, 0.0), resolution_ref);
			initializeSurfaceProperties(n_0, thickness);
		}
	};
};

class BoundaryGeometry : public BodyPartByParticle
{
public:
	BoundaryGeometry(SPHBody &body, const std::string &body_part_name)
		: BodyPartByParticle(body, body_part_name)
	{
		TaggingParticleMethod tagging_particle_method = std::bind(&BoundaryGeometry::tagManually, this, _1);
		tagParticles(tagging_particle_method);
	};
	virtual ~BoundaryGeometry(){};

private:
	void tagManually(size_t index_i)
	{
		if (base_particles_->pos_[index_i][0] < 0.0 || base_particles_->pos_[index_i][0] > PL)
		{
			body_part_particles_.push_back(index_i);
		}
	};
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up -- a SPHSystem
	//----------------------------------------------------------------------
	SPHSystem system(system_domain_bounds, resolution_ref);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody plate_body(system, makeShared<DefaultShape>("PlateBody"));
	plate_body.defineParticlesAndMaterial<ShellParticles, LinearElasticSolid>(rho0_s, Youngs_modulus, poisson);
	plate_body.generateParticles<PlateParticleGenerator>();
	plate_body.addBodyStateForRecording<Vecd>("PriorAcceleration");

	ObserverBody plate_observer(system, "PlateObserver");
	plate_observer.generateParticles<ObserverParticleGenerator>(observation_location);
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInner plate_body_inner(plate_body);
	BodyRelationContact plate_observer_contact(plate_observer, {&plate_body});
	//----------------------------------------------------------------------
	//	Define all numerical methods which are used in this case.
	//----------------------------------------------------------------------
	/** Corrected configuration. */
	thin_structure_dynamics::ShellCorrectConfiguration corrected_configuration(plate_body_inner);
	/** Time step size. */
	thin_structure_dynamics::ShellAcousticTimeStepSize computing_time_step_size(plate_body);
	/** stress relaxation. */
	thin_structure_dynamics::ShellStressRelaxationFirstHalf stress_relaxation_first_half(plate_body_inner, 3, true);
	thin_structure_dynamics::ShellStressRelaxationSecondHalf stress_relaxation_second_half(plate_body_inner);
	thin_structure_dynamics::DistributingPointForcesToShell
		apply_point_force(plate_body, point_force, reference_position, time_to_full_external_force, resolution_ref);
	/** Constrain the Boundary. */
	BoundaryGeometry boundary_geometry(plate_body, "BoundaryGeometry");
	thin_structure_dynamics::ConstrainShellBodyRegion constrain_holder(plate_body, boundary_geometry);
	DampingWithRandomChoice<DampingPairwiseInner<Vec2d>>
		plate_position_damping(0.2, plate_body_inner, "Velocity", physical_viscosity);
	DampingWithRandomChoice<DampingPairwiseInner<Vec2d>>
		plate_rotation_damping(0.2, plate_body_inner, "AngularVelocity", physical_viscosity);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	InOutput in_output(system);
	BodyStatesRecordingToVtp write_states(in_output, system.real_bodies_);
	RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
		write_plate_max_displacement("Position", in_output, plate_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	corrected_configuration.parallel_exec();
	apply_point_force.getWeight();
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_states.writeToFile(0);
	write_plate_max_displacement.writeToFile(0);
	//----------------------------------------------------------------------
	//	Basic control parameters for time stepping.
	//----------------------------------------------------------------------
	int ite = 0;
	Real end_time = 0.8;
	Real output_period = end_time / 100.0;
	Real dt = 0.0;
	/** Statistics for computing time. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	Main loop of time stepping starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integral_time = 0.0;
		while (integral_time < output_period)
		{
			if (ite % 1000 == 0)
			{
				std::cout << "N=" << ite << " Time: "
						  << GlobalStaticVariables::physical_time_ << "	dt: "
						  << dt << "\n";
			}
			apply_point_force.getForce();
			apply_point_force.parallel_exec(dt);
			stress_relaxation_first_half.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			plate_position_damping.parallel_exec(dt);
			plate_rotation_damping.parallel_exec(dt);
			constrain_holder.parallel_exec(dt);
			stress_relaxation_second_half.parallel_exec(dt);

			ite++;
			dt = computing_time_step_size.parallel_exec();
			integral_time += dt;
			GlobalStaticVariables::physical_time_ += dt;
		}
		write_plate_max_displacement.writeToFile(ite);
		tick_count t2 = tick_count::now();
		write_states.writeToFile();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	write_plate_max_displacement.newResultTest();

	return 0;
}
