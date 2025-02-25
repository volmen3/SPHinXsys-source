/**
 * @file 	airfoil_2d.cpp
 * @brief 	This is the test of using levelset to generate body fitted SPH particles.
 * @details	We use this case to test the particle generation and relaxation with a complex geometry (2D).
 *			Before the particles are generated, we clean the sharp corners and other unresolvable surfaces.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */

#include "sphinxsys.h"

using namespace SPH;

//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string airfoil_flap_front = "./input/airfoil_flap_front.dat";
std::string airfoil_wing = "./input/airfoil_wing.dat";
std::string airfoil_flap_rear = "./input/airfoil_flap_rear.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 1.25;				/**< airfoil length rear part. */
Real DL1 = 0.25;			/**< airfoil length front part. */
Real DH = 0.25;				/**< airfoil height. */
Real resolution_ref = 0.02; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL1, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
class ImportModel : public MultiPolygonShape
{
public:
	explicit ImportModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
	{
		multi_polygon_.addAPolygonFromFile(airfoil_flap_front, ShapeBooleanOps::add);
		multi_polygon_.addAPolygonFromFile(airfoil_wing, ShapeBooleanOps::add);
		multi_polygon_.addAPolygonFromFile(airfoil_flap_rear, ShapeBooleanOps::add);
	}
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
	//----------------------------------------------------------------------
	//	Build up -- a SPHSystem
	//----------------------------------------------------------------------
	SPHSystem system(system_domain_bounds, resolution_ref);
	/** Tag for run particle relaxation for the initial body fitted distribution. */
	system.run_particle_relaxation_ = true;
// handle command line arguments
#ifdef BOOST_AVAILABLE
	system.handleCommandlineOptions(ac, av);
#endif
	/** output environment. */
	InOutput in_output(system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	RealBody airfoil(system, makeShared<ImportModel>("AirFoil"));
	airfoil.defineAdaptation<ParticleSpacingByBodyShape>(1.15, 1.0, 3);
	airfoil.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(airfoil);
	airfoil.defineParticlesAndMaterial();
	airfoil.generateParticles<ParticleGeneratorMultiResolution>();
	airfoil.addBodyStateForRecording<Real>("SmoothingLengthRatio");
	//----------------------------------------------------------------------
	//	Define simple file input and outputs functions.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp airfoil_recording_to_vtp(in_output, {&airfoil});
	MeshRecordingToPlt cell_linked_list_recording(in_output, airfoil, airfoil.cell_linked_list_);
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies,
	//	basically, in the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInnerVariableSmoothingLength airfoil_inner(airfoil);
	//----------------------------------------------------------------------
	//	Methods used for particle relaxation.
	//----------------------------------------------------------------------
	RandomizeParticlePosition random_airfoil_particles(airfoil);
	relax_dynamics::RelaxationStepInner relaxation_step_inner(airfoil_inner, true);
	relax_dynamics::UpdateSmoothingLengthRatioByBodyShape update_smoothing_length_ratio(airfoil);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	random_airfoil_particles.parallel_exec(0.25);
	relaxation_step_inner.surface_bounding_.parallel_exec();
	update_smoothing_length_ratio.parallel_exec();
	airfoil.updateCellLinkedList();
	//----------------------------------------------------------------------
	//	First output before the simulation.
	//----------------------------------------------------------------------
	airfoil_recording_to_vtp.writeToFile(0);
	cell_linked_list_recording.writeToFile(0);
	//----------------------------------------------------------------------
	//	Particle relaxation time stepping start here.
	//----------------------------------------------------------------------
	int ite_p = 0;
	while (ite_p < 2000)
	{
		update_smoothing_length_ratio.parallel_exec();
		relaxation_step_inner.parallel_exec();
		ite_p += 1;
		if (ite_p % 100 == 0)
		{
			std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the airfoil N = " << ite_p << "\n";
			airfoil_recording_to_vtp.writeToFile(ite_p);
		}
	}
	std::cout << "The physics relaxation process of airfoil finish !" << std::endl;

	return 0;
}
