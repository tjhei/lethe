/* ---------------------------------------------------------------------
 *  Unit test for the integration of a cut mesh using decomposition
 *  Test 1 - Integrate a square cut by a vertical line
 *  Test 2 - Integrate a square cut by a diagonal line
 *  Test 3 - Integrate a square minus a circle
 *  The variable tested for is the volume of the integration
 * ---------------------------------------------------------------------
 *
 * Author: Bruno Blais, Polytechnique Montreal
 */


//BASE
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

//NUMERICS
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

//GRID
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>

//LACS
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>


//DOFS
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


// Distributed
#include <deal.II/distributed/tria.h>

#include "ibcomposer.h"
#include "ibcombiner.h"
#include "iblevelsetfunctions.h"
#include "write_data.h"
#include "../tests.h"

// Triangles decomposition tools
#include "nouvtriangles.h"

using namespace dealii;

double integrate_sub_element( Triangulation<2> &sub_triangulation)
{
  double area=0;

  // Create a dummy dof_handler to handle the 4 dofts
  DoFHandler<2>                  sub_dof_handler(sub_triangulation);
  // Create a FE system for this element
  FESystem<2>                    sub_fe(FE_Q<2>(1),1);
  sub_dof_handler.distribute_dofs(sub_fe);

  // Create a mapping for this new element
  const MappingQ<2>      sub_mapping (1);
  QGauss<2>              sub_quadrature_formula(4);

  // Integrate over this element, in this case we only integrate
  // over the quadrature to calculate the area
  FEValues<2> sub_fe_values (sub_fe, sub_quadrature_formula,
                         update_quadrature_points | update_JxW_values);
  const unsigned int   n_q_points    = sub_quadrature_formula.size();

  typename DoFHandler<2>::active_cell_iterator
      sub_cell = sub_dof_handler.begin_active(),
      sub_endc = sub_dof_handler.end();
  for (; sub_cell!=sub_endc; ++sub_cell)
  {
    sub_fe_values.reinit (sub_cell);
    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      area+=sub_fe_values.JxW (q_index);
    }
  }
  return area;
}
double area_integrator(int refinement_level,   std::vector<IBLevelSetFunctions<2> *> ib_functions, std::string output_name)
{
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
  unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
  unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));


  // Create triangulation and square mesh
  parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                         (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             -1,1);
  triangulation.refine_global(refinement_level);


//  IBComposer<2> ib_composer(&triangulation,ib_functions);

//  // Calculate the distance
//  ib_composer.calculateDistance();

//  //Get the distance into a local vector
//  TrilinosWrappers::MPI::Vector levelSet_distance=ib_composer.getDistance();

  // Get DOF handler and output the IB distance field
  //DoFHandler<2> *dof_handler(ib_composer.getDoFHandler());
  //write_ib_scalar_data<2>(triangulation,*dof_handler,mpi_communicator,levelSet_distance,output_name);

  DoFHandler<2>                  dof_handler(triangulation);
  FESystem<2>                    fe(FE_Q<2>(1),1);
  dof_handler.distribute_dofs(fe);

  // Quadrature formula for the element
  QGauss<2>            quadrature_formula(1);
  const unsigned int   n_q_points = quadrature_formula.size();


  // Get the position of the support points
  const MappingQ<2>      mapping (1);

  FEValues<2> fe_values (mapping,
                         fe,
                         quadrature_formula,
                         update_values |
                         update_quadrature_points |
                         update_JxW_values
                         );

  std::map< types::global_dof_index, Point< 2 > > support_points;
  DoFTools::map_dofs_to_support_points ( mapping, dof_handler,support_points );

  // Instantiations for the decomposition of the elements
  std::vector<int>                     corresp(9);
  Vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
  Vector<node_status>    No_pts_solid(4);
  int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
  std::vector<Point<2> >               num_elem(6);

  // Set the values of the dof points position
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;       // Number of dofs per cells.
  Vector<double>                  distance(dofs_per_cell); // Array for the distances associated with the DOFS
  Vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell

  IBCombiner<2>  ib_combiner(ib_functions);

  double area=0;

  typename DoFHandler<2>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit(cell);
    cell->get_dof_indices (local_dof_indices);

    for (unsigned int dof_index=0 ; dof_index < local_dof_indices.size() ; ++dof_index)
    {
      dofs_points[dof_index] = support_points[local_dof_indices[dof_index]];
      distance[dof_index]    = ib_combiner.value(dofs_points[dof_index]);
//      std::cout << "Point : " << dofs_points[dof_index] << ", distance : " << distance[dof_index] << std::endl;
    }
//    std::cout << "\n" << std::endl;
    // Decompose the geometry
    nouvtriangles(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, dofs_points, distance);

    if (nb_poly==0 && (distance[0]>0))
    {
      fe_values.reinit(cell);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        area+=fe_values.JxW (q_index);
      }
    }
    if (nb_poly==-1)
    {
      // Create triangulation points
      std::vector<Point<2> > triangulation_points(GeometryInfo<2>::vertices_per_cell);
      // Create 4 random points:
      for (unsigned int i_pt =0 ; i_pt < 4 ; ++i_pt)
        triangulation_points[i_pt]=decomp_elem[i_pt];

      // Prepare cell data
      std::vector<CellData<2> > cells (1);
      for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i)
          cells[0].vertices[i] = i;
      cells[0].material_id = 0;

      Triangulation<2> sub_triangulation;
      sub_triangulation.create_triangulation (triangulation_points, cells, SubCellData());
      area+= integrate_sub_element(sub_triangulation);
    }
    if (nb_poly==1)
    {
      // Create triangulation points
      std::vector<Point<2> > triangulation_points(3);
      // Create 4 random points:
      for (unsigned int i_pt =0 ; i_pt < 3 ; ++i_pt)
      {
        triangulation_points[i_pt]=decomp_elem[i_pt];
      }
      Triangulation<2> sub_triangulation;
      GridGenerator::simplex(sub_triangulation,triangulation_points);
      area+= integrate_sub_element(sub_triangulation);
    }
    if (nb_poly==3)
    {
      //Create new triangulation and integrate
      for (int sub_element = 0 ; sub_element<nb_poly ; ++sub_element)
      {
        // Create triangulation points
        std::vector<Point<2> > triangulation_points(3);
        // Create 4 random points:
        for (unsigned int i_pt =0 ; i_pt < 3 ; ++i_pt)
        {
          triangulation_points[i_pt]=decomp_elem[3*sub_element+i_pt];
        }
        Triangulation<2> sub_triangulation;
        GridGenerator::simplex(sub_triangulation,triangulation_points);
        area += integrate_sub_element(sub_triangulation);
      }
    }

  }
  return area;
}

void cut_square()
{
  // Generate the IB composer
  Point<2> center1(0.04,0);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=-1; normal[1]=0;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  double area = area_integrator(2, ib_functions,"IB_B_02_square");
  double error = area- (2*1.04) ;
  std::cout << "Plane : " << 1 <<" Error : " << error << std::endl;
  if (!approximatelyEqual(error,0,1e-10)) throw std::runtime_error("Test 1 - Wrong area for cut square");

}

void cut_square_angle1()
{
  // Generate the IB composer
  double pt=0.03;
  Point<2> center1(pt,1);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=-1; normal[1]=-1;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  double area = area_integrator(2, ib_functions,"IB_B_02_angle");
  double error = std::abs(area- (2*(1+(pt))+ (1-pt)*(0.5*(2+2-0.97))));
  std::cout << "Plane : " << 2 <<" Error : " << error << std::endl;
  if (error>1e-12) throw std::runtime_error("Test 2.1 - Wrong area for squared cut angle");
}

void cut_square_angle2()
{
  // Generate the IB composer
  double pt=0.03;
  Point<2> center1(pt,-1);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=-1; normal[1]=1;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  double area = area_integrator(2, ib_functions,"IB_B_02_angle");
  double error = std::abs(4-(0.97*0.97/2+area));
  std::cout << "Plane : " << 2 <<" Error : " << error << std::endl;
  if (error>1e-12) throw std::runtime_error("Test 2.2 - Wrong area for squared cut angle");
}

void cut_square_angle3()
{
  // Generate the IB composer
  double pt=-0.03;
  Point<2> center1(pt,1);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=1; normal[1]=-1;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  double area = area_integrator(2, ib_functions,"IB_B_02_angle");
  double error = std::abs(4-(0.97*0.97/2+area));
  std::cout << "Plane : " << 2 <<" Error : " << error << std::endl;
  if (error>1e-12) throw std::runtime_error("Test 2.3 - Wrong area for squared cut angle");
}

void cut_square_angle4()
{
  // Generate the IB composer
  double pt=-0.03;
  Point<2> center1(pt,-1);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=1; normal[1]=1;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  double area = area_integrator(2, ib_functions,"IB_B_02_angle");
  double error = std::abs(4-(0.97*0.97/2+area));
  std::cout << "Plane : " << 2 <<" Error : " << error << std::endl;
  if (error>1e-12) throw std::runtime_error("Test 2.3 - Wrong area for squared cut angle");
}


// Square with a hole inside
void square_hole()
{
  // Generate the IB composer
  Point<2> center1(0.0111,0.002547);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,3> angular;
  angular[0]=0; angular[1]=0; angular[2]=0;
  double radius =0.4711; double T_scal=1; bool inside=0;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetCircle<2> circle1(center1,velocity,angular, T_scal, inside, radius);
  ib_functions.push_back(&circle1);
  for (int i=3 ; i<11 ; ++i)
  {
    double area = area_integrator(i, ib_functions,"IB_B_02_circle");
    std::cout << "Circle : " << i <<" Error : " << area- (4.-radius*radius*M_PI) << std::endl;
  }
}

int
main(int argc, char* argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    initlog();
    cut_square();
    cut_square_angle1();
    cut_square_angle2();
    cut_square_angle3();
    cut_square_angle4();
    square_hole();


  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what()  << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
