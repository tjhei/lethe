//BASE
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

//GRID
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>

//LACS
#include <deal.II/lac/trilinos_vector.h>


//DOFS
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


// Distributed
#include <deal.II/distributed/tria.h>

#include "ibcomposer.h"
#include "iblevelsetfunctions.h"
#include "write_data.h"
#include "../tests.h"

// Mes ajouts so far
#include "nouvtriangles.h"
#include "area.h"


using namespace dealii;

void test1_loop_composed_distance()
{
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
  unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
  unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

  // Create triangulation and square mesh
  parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                         (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             -2,2);

  // Refine it to get an interesting number of elements
  triangulation.refine_global(5);

  // Set-up the center, velocity and angular velocity of circle
  Point<2> center1(0.2356,-0.0125);
  Tensor<1,2> velocity;
  velocity[0]=1.;
  velocity[1]=0.;
  Tensor<1,3> angular;
  angular[0]=0;
  angular[1]=0;
  angular[2]=0;
  Tensor<1,1> T_scal;
  T_scal[0]=1;
  double radius =1.23549;
  bool inside=0;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetCircle<2> circle1(center1,velocity,angular, T_scal, inside, radius);
  ib_functions.push_back(&circle1);
  IBComposer<2> ib_composer(&triangulation,ib_functions);

  // Calculate the distance
  ib_composer.calculateDistance();

  //Get the distance into a local vector
  TrilinosWrappers::MPI::Vector levelSet_distance=ib_composer.getDistance();

  // Get DOF handler and output the IB distance field
  DoFHandler<2> *dof_handler(ib_composer.getDoFHandler());
  write_ib_scalar_data<2>(triangulation,*dof_handler,mpi_communicator,levelSet_distance,"ls_composed_distance");


  // Loop over all elements and extract the distances into a local array
  FESystem<2> *fe(ib_composer.getFESystem());
  QGauss<2>              quadrature_formula(1);
  const MappingQ<2>      mapping (1);
  std::map< types::global_dof_index, Point< 2 > > support_points;
  DoFTools::map_dofs_to_support_points ( mapping, *dof_handler,support_points );
  FEValues<2> fe_values (mapping,
                         *fe,
                         quadrature_formula,
                         update_values |
                         update_quadrature_points |
                         update_JxW_values
                         );
  const unsigned int   dofs_per_cell = fe->dofs_per_cell;         // Number of dofs per cells.
  const unsigned int   n_q_points    = quadrature_formula.size(); // quadrature on normal elements
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell
  std::vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<double>  distance                  (dofs_per_cell); // Array for the distances associated with the DOFS

  std::vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
  int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
  double                               fluid_area = 0;
  double                               area_temp;
  double areaa = 3.141 * radius * radius ;
  std::vector<Point<2> >               num_elem(6);
  std::vector<int>                     corresp(9);
  std::vector<int>                     No_pts_fluid(4);

  Point<2> a;
  a[0]=0;
  a[1]=0;

  typename DoFHandler<2>::active_cell_iterator
  cell = dof_handler->begin_active(),
  endc = dof_handler->end();
  for (; cell!=endc; ++cell)
  {
    area_temp = 0.0;
    std::fill(decomp_elem.begin(), decomp_elem.end(), a);
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices (local_dof_indices);

      for (unsigned int dof_index=0 ; dof_index < local_dof_indices.size() ; ++dof_index)
      {
        distance[dof_index] = levelSet_distance[local_dof_indices[dof_index]];
        dofs_points[dof_index] = support_points[local_dof_indices[dof_index]];
        //std::cout << /* "Dof number : " << local_dof_indices[dof_index] << */ " - Point : " << dofs_points[dof_index] <<" - Distance : " << distance[dof_index] << std::endl;
      }
      nouvtriangles(corresp, No_pts_fluid, num_elem, decomp_elem, &nb_poly, dofs_points, distance); // on a plus besoin de decomp_elem en théorie parce aue la combinaison de corresp et num_elem nous donne les coordonnées de la décomposition, à voir si on garde ou pas

     //area_temp = area(nb_poly, decomp_elem, distance, dofs_points);
     //areaa += area_temp; // CALCUL DE L'AIRE DE LA ZONE FLUIDE.

     /*
     if (nb_poly != 0) {std::cout << "Coor elem : "   << dofs_points[0] << ", "  << dofs_points[1] << ", "  << dofs_points[2] << ", "  << dofs_points[3] << "\n "  <<
                                     "Val f dist : "  << distance[0] << ", "  << distance[1] << ", "  << distance[2] << ", "  << distance[3] << "\n "  <<
                                     "Pts fluid : "  << No_pts_fluid[0] << ", " << No_pts_fluid[1] << ", " <<No_pts_fluid[2] << ", " <<No_pts_fluid[3] << "\n " <<
                                     "Num elem : "    << num_elem[0] << ", " << num_elem[1] << ", " << num_elem[2] << ", " << num_elem[3] << ", " << num_elem[4] << ", " << num_elem[5] << "\n " <<
                                     "Corresp : "     << corresp[0] << ", " << corresp[1] << ", " << corresp[2] << ", " << corresp[3] << ", " << corresp[4] << ", " << corresp[5] << ", " << corresp[6] << ", " << corresp[7] << ", " << corresp[8] << "\n " <<

                                     "\n \n" <<std::endl;}
    */

    }
  }
  //{std::cout << "erreur sur l'aire de la zone fluide : "<< areaa -16.0 << "\n \n" <<std::endl;}
}

int
main(int argc, char* argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    initlog();
    test1_loop_composed_distance();
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
