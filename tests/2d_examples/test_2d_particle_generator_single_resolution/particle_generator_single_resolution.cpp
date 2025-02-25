/**
* @file 	particle_generator_single_resolution.cpp
* @brief 	This is the test of using levelset to generate particles with single resolution and relax particles.
* @details	We use this case to test the particle generation and relaxation by levelset for a complex geometry (2D).
*			Before particle generation, we clean the level set, then do re-initialization.

* @author 	Yongchuan Yu and Xiangyu Hu
*/

#include "sphinxsys.h"

using namespace SPH;

//----------------------------------------------------------------------
//	Set the file path to the data file
//----------------------------------------------------------------------
std::string input_body = "./input/SPHinXsys-2d.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters
//----------------------------------------------------------------------
Real DL = 2.3;							/**< InputBody length right part. */
Real DL1 = 2.3;							/**< InputBody length left part. */
Real DH = 4.5;							/**< InputBody height. */
Real resolution_ref = (DL + DL1) / 120; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL1, 0), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	Shape of the InputBody
//----------------------------------------------------------------------
class InputBody : public MultiPolygonShape
{
public:
	explicit InputBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(input_body, ShapeBooleanOps::add);
	}
};
//----------------------------------------------------------------------
//	The main program
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up -- a SPHSystem
	//----------------------------------------------------------------------
	SPHSystem system(system_domain_bounds, resolution_ref);
	InOutput in_output(system); // output environment
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	RealBody input_body(system, makeShared<InputBody>("SPHInXsysLogo"));
	input_body.defineBodyLevelSetShape();
	input_body.defineParticlesAndMaterial();
	input_body.generateParticles<ParticleGeneratorLattice>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInner input_body_inner(input_body);
	//----------------------------------------------------------------------
	//	Methods used for particle relaxation.
	//----------------------------------------------------------------------
	RandomizeParticlePosition random_input_body_particles(input_body);
	relax_dynamics::RelaxationStepInner relaxation_step_inner(input_body_inner, true);
	//----------------------------------------------------------------------
	//	Define simple file input and outputs functions.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp input_body_recording_to_vtp(in_output, input_body);
	MeshRecordingToPlt cell_linked_list_recording(in_output, input_body, input_body.cell_linked_list_);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	random_input_body_particles.parallel_exec(0.25);
	relaxation_step_inner.surface_bounding_.parallel_exec();
	input_body.updateCellLinkedList();
	//----------------------------------------------------------------------
	//	First output before the simulation.
	//----------------------------------------------------------------------
	input_body_recording_to_vtp.writeToFile();
	cell_linked_list_recording.writeToFile();
	//----------------------------------------------------------------------
	//	Particle relaxation time stepping start here.
	//----------------------------------------------------------------------
	int ite_p = 0;
	while (ite_p < 1000)
	{
		relaxation_step_inner.parallel_exec();
		ite_p += 1;
		if (ite_p % 100 == 0)
		{
			std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the airfoil N = " << ite_p << "\n";
			input_body_recording_to_vtp.writeToFile(ite_p);
		}
	}
	std::cout << "The physics relaxation process of airfoil finish !" << std::endl;

	return 0;
}
