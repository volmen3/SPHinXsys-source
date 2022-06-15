/**
 * @file 	regression_test.cpp
 * @brief 	This is a test case based on diffusion, which can be used to
			validate the generation of the converged database in a regression test.
			It can be run successfully (using CMake's CTest) in Linux system installed with Python 3.
 * @author 	Bo Zhang and Xiangyu Hu
 */

#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH;   // namespace cite here
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 0.2;
Real H = 0.2;
Real resolution_ref = H / 40.0;
Real BW = resolution_ref * 4;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Global parameters on material properties
//----------------------------------------------------------------------
Real diffusion_coff = 1.0e-3;
Real bias_coff = 0.0;
Real alpha = Pi / 4.0;
Vec2d bias_direction(cos(alpha), sin(alpha));
Real initial_temperature = 0.0;
Real high_temperature = 1.0;
Real low_temperature = 0.0;
//----------------------------------------------------------------------
//	Cases-dependent 2D geometries
//----------------------------------------------------------------------
MultiPolygon createDiffusionDomain()
{
	// thermal solid domain geometry.
	std::vector<Vecd> diffusion_domain;
	diffusion_domain.push_back(Vecd(-BW, -BW));
	diffusion_domain.push_back(Vecd(-BW, H + BW));
	diffusion_domain.push_back(Vecd(L + BW, H + BW));
	diffusion_domain.push_back(Vecd(L + BW, -BW));
	diffusion_domain.push_back(Vecd(-BW, -BW));

	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(diffusion_domain, ShapeBooleanOps::add);
	return multi_polygon;
}

MultiPolygon createInnerDomain()
{
	// thermal solid inner domain geometry.
	std::vector<Vecd> inner_domain;
	inner_domain.push_back(Vecd(0.0, 0.0));
	inner_domain.push_back(Vecd(0.0, H));
	inner_domain.push_back(Vecd(L, H));
	inner_domain.push_back(Vecd(L, 0.0));
	inner_domain.push_back(Vecd(0.0, 0.0));

	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(inner_domain, ShapeBooleanOps::add);

	return multi_polygon;
}

MultiPolygon createLeftBoundary()
{
	// left isothermal boundary geometry.
	std::vector<Vecd> left_boundary;
	left_boundary.push_back(Vecd(-BW, -BW));
	left_boundary.push_back(Vecd(-BW, H + BW));
	left_boundary.push_back(Vecd(0.0, H));
	left_boundary.push_back(Vecd(0.0, 0.0));
	left_boundary.push_back(Vecd(-BW, -BW));

	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(left_boundary, ShapeBooleanOps::add);

	return multi_polygon;
}

MultiPolygon createOtherBoundaries()
{
	// other side isothermal boundary geometry.
	std::vector<Vecd> other_boundaries;
	other_boundaries.push_back(Vecd(-BW, -BW));
	other_boundaries.push_back(Vecd(0.0, 0.0));
	other_boundaries.push_back(Vecd(L, 0.0));
	other_boundaries.push_back(Vecd(L, H));
	other_boundaries.push_back(Vecd(0.0, L));
	other_boundaries.push_back(Vecd(-BW, H + BW));
	other_boundaries.push_back(Vecd(L + BW, H + BW));
	other_boundaries.push_back(Vecd(L + BW, -BW));
	other_boundaries.push_back(Vecd(-BW, -BW));

	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(other_boundaries, ShapeBooleanOps::add);

	return multi_polygon;
}
//----------------------------------------------------------------------
//	Cases-dependent diffusion material
//----------------------------------------------------------------------
class DiffusionMaterial : public DiffusionReaction<Solid>
{
public:
	DiffusionMaterial() : DiffusionReaction<Solid>({"Phi"})
	{
		initializeAnDiffusion<DirectionalDiffusion>("Phi", "Phi", diffusion_coff, bias_coff, bias_direction);
	};
};
//----------------------------------------------------------------------
//	Case-dependent initial condition.
//----------------------------------------------------------------------
class DiffusionInitialCondition
	: public DiffusionReactionInitialCondition<SolidBody, SolidParticles, Solid>
{
protected:
	StdLargeVec<Real> &species_;

public:
	explicit DiffusionInitialCondition(SolidBody &diffusion_body)
		: DiffusionReactionInitialCondition<SolidBody, SolidParticles, Solid>(diffusion_body),
		  species_(particles_->species_n_[material_->SpeciesIndexMap()["Phi"]]){};

	void updateRange(const IndexRange &particle_range, Real dt)
	{

		for (size_t index_i = particle_range.begin(); index_i < particle_range.end(); ++index_i)
		{
			if (pos_n_[index_i][0] >= 0 && pos_n_[index_i][0] <= L && pos_n_[index_i][1] >= 0 && pos_n_[index_i][1] <= H)
			{
				species_[index_i] = initial_temperature;
			}
		}
	};
};
//----------------------------------------------------------------------
//	Set boundary condition by constrain to a constant temperature.
//----------------------------------------------------------------------
class ConstraintTemperature
	: public ConstraintSpecies<SolidBody, SolidParticles, Solid>
{
protected:
	Real constrained_value_;
	StdLargeVec<Real> &species_;

public:
	explicit ConstraintTemperature(SolidBody &diffusion_body, Real constrained_value)
		: ConstraintSpecies<SolidBody, SolidParticles, Solid>(diffusion_body),
		  constrained_value_(constrained_value),
		  species_(particles_->species_n_[material_->SpeciesIndexMap()["Phi"]]){};

	void updateList(const IndexRange &entry_range, const IndexVector &particle_list, Real dt = 0.0)
	{

		for (size_t i = entry_range.begin(); i != entry_range.end(); ++i)
		{
			size_t index_i = particle_list[i];
			species_[index_i] = constrained_value_;
		}
	}
};
//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
class DiffusionBodyRelaxation
	: public RelaxationOfAllDiffusionSpeciesRK2<
		  RelaxationOfAllDiffusionSpeciesInner<SolidBody, SolidParticles, Solid>>
{
public:
	explicit DiffusionBodyRelaxation(BodyRelationInner &body_inner_relation)
		: RelaxationOfAllDiffusionSpeciesRK2(body_inner_relation){};
	virtual ~DiffusionBodyRelaxation(){};
};
//----------------------------------------------------------------------
//	an observer body to measure temperature at given positions.
//----------------------------------------------------------------------
class TemperatureObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	explicit TemperatureObserverParticleGenerator(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
	{
		/** A line of measuring points at the middle line. */
		size_t number_of_observation_points = 11;
		Real range_of_measure = L - BW;
		Real start_of_measure = BW;

		for (size_t i = 0; i < number_of_observation_points; ++i)
		{
			Vec2d point_coordinate(0.5 * L, start_of_measure + range_of_measure * (Real)i / (Real)(number_of_observation_points - 1));
			positions_.push_back(point_coordinate);
		}
	}
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	/** output environment. */
	InOutput in_output(sph_system);
	//----------------------------------------------------------------------
	//	Create body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody diffusion_body(sph_system, makeShared<MultiPolygonShape>(createDiffusionDomain(), "DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles>, DiffusionMaterial>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	//----------------------------------------------------------------------
	//	Observer body
	//----------------------------------------------------------------------
	ObserverBody temperature_observer(sph_system, "TemperatureObserver");
	temperature_observer.generateParticles<TemperatureObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInner diffusion_body_inner_relation(diffusion_body);
	BodyRelationContact temperature_observer_contact(temperature_observer, {&diffusion_body});
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
	/** Left wall boundary conditions */
	BodyRegionByParticle left_boundary(diffusion_body, makeShared<MultiPolygonShape>(createLeftBoundary()));
	PartByParticleSimpleDynamics<ConstraintTemperature> left_boundary_condition(left_boundary, diffusion_body, high_temperature);
	/** Other wall boundary conditions */
	BodyRegionByParticle other_boundaries(diffusion_body, makeShared<MultiPolygonShape>(createOtherBoundaries()));
	PartByParticleSimpleDynamics<ConstraintTemperature> other_boundaries_condition(other_boundaries, diffusion_body, low_temperature);
	/** Corrected configuration for diffusion body. */
	InteractionDynamics<solid_dynamics::CorrectConfiguration> correct_configuration(diffusion_body_inner_relation);
	/** Time step size calculation. */
	DynamicsParameter<GetDiffusionTimeStepSize<SolidBody, SolidParticles, Solid>> get_time_step_size(diffusion_body);
	/** Diffusion process for diffusion body. */
	DiffusionBodyRelaxation diffusion_relaxation(diffusion_body_inner_relation);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations, observations of the simulation.
	//	Regression tests are also defined here.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(in_output, sph_system.real_bodies_);
	RegressionTestEnsembleAveraged<ObservedQuantityRecording<Real>>
		write_solid_temperature("Phi", in_output, temperature_observer_contact);
	BodyRegionByParticle inner_domain(diffusion_body, makeShared<MultiPolygonShape>(createInnerDomain()));
	
	RegressionTestDynamicTimeWarping<BodyReducedQuantityRecording<
		BodyPartByParticleAverage<SpeciesSummation<SolidBody, SolidParticles, Solid>>>>
		write_solid_average_temperature_part(in_output, inner_domain, diffusion_body, "Phi");
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	correct_configuration.parallel_exec();
	setup_diffusion_initial_condition.parallel_exec();
	left_boundary_condition.parallel_exec();
	other_boundaries_condition.parallel_exec();
	/** Output global basic parameters. */
	write_states.writeToFile(0);
	write_solid_temperature.writeToFile(0);
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int ite = 0;
	Real T0 = 20.0;
	Real End_Time = T0;
	Real Output_Time = 0.1 * End_Time;
	Real Observe_time = 0.1 * Output_Time;
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
		while (integration_time < Output_Time)
		{
			Real relaxation_time = 0.0;
			while (relaxation_time < Observe_time)
			{
				if (ite % 1 == 0)
				{
					std::cout << "N=" << ite << " Time: "
							  << GlobalStaticVariables::physical_time_ << "	dt: "
							  << dt << "\n";
				}

				diffusion_relaxation.parallel_exec(dt);
				left_boundary_condition.parallel_exec();
				other_boundaries_condition.parallel_exec();
				ite++;
				dt = get_time_step_size.parallel_exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;

				if (ite % 100 == 0)
				{
					write_solid_temperature.writeToFile(ite);
					write_solid_average_temperature_part.writeToFile(ite);
					write_states.writeToFile(ite);
				}
			}
		}

		tick_count t2 = tick_count::now();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();
	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	//----------------------------------------------------------------------
	//	@ensemble_average_method.
	//	The first argument is the threshold of mean value convergence.
	//	The second argument is the threshold of variance convergence.
	//----------------------------------------------------------------------
	write_solid_temperature.generateDataBase(0.001, 0.001);
	//----------------------------------------------------------------------
	//	@dynamic_time_warping_method.
	//	The value is the threshold of dynamic time warping (dtw) distance.
	//----------------------------------------------------------------------
	write_solid_average_temperature_part.generateDataBase(0.001);

	return 0;
}