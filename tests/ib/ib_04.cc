/*The function given here integrates the weak form of the
 * heat equation, for a domain totally fluid (no decomposition here).
 * The right hand side is chosen so that the solution we should get
 * is T = cos(pi*x/4)cos(pi*y/4).
 * */



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
#include "iblevelsetfunctions.h"
#include "write_data.h"
#include "../tests.h"




using namespace dealii;

void Temperature_field_full_fluid()

{
//This one is to check if we can do the same as step-3 by calculating the elementary matrices in each element by ourselves


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
  double T_scal;
  T_scal=1;
  double radius =1.0;
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
  QGauss<2>              quadrature_formula(8);
  const MappingQ<2>      mapping (1);
  std::map< types::global_dof_index, Point< 2 > > support_points;
  DoFTools::map_dofs_to_support_points ( mapping, *dof_handler,support_points );
  FEValues<2> fe_values (*fe, quadrature_formula,
                                  update_values | update_gradients | update_JxW_values|
                         update_quadrature_points ); /*(mapping,
                         *fe,
                         quadrature_formula,
                         update_values |
                         update_quadrature_points |
                         update_JxW_values
                         )*/
  const unsigned int   dofs_per_cell = fe->dofs_per_cell;         // Number of dofs per cells.
  const unsigned int   n_q_points    = quadrature_formula.size(); // quadrature on normal elements
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell
  std::vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<double>  distance                  (dofs_per_cell); // Array for the distances associated with the DOFS

  double                               Tdirichlet = 1.0;

  SparsityPattern                      sparsity_pattern;
  SparseMatrix<double>                 system_matrix;
  DynamicSparsityPattern               dsp(dof_handler->n_dofs());

  Vector<double>                       solution;
  Vector<double>                       system_rhs;

  DoFTools::make_sparsity_pattern (*dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler->n_dofs());
  system_rhs.reinit (dof_handler->n_dofs());

  FullMatrix<double> cell_mat(dofs_per_cell, dofs_per_cell); // elementary matrix


//  double matelem[4][4];

  Vector<double> cell_rhs(dofs_per_cell);

  typename DoFHandler<2>::active_cell_iterator
  cell = dof_handler->begin_active(),
  endc = dof_handler->end();
  for (; cell!=endc; ++cell)
  {


    if (cell->is_locally_owned())
    {
      cell_rhs=0;
      cell_mat=0;

      fe_values.reinit(cell);
      cell->get_dof_indices (local_dof_indices);

      for (unsigned int dof_index=0 ; dof_index < local_dof_indices.size() ; ++dof_index)
      {
        distance[dof_index] = 1; //levelSet_distance[local_dof_indices[dof_index]];
        dofs_points[dof_index] = support_points[local_dof_indices[dof_index]];
         // We thus resolve -Laplacian(u) = f in the domain, u = 0 on the boundary of the domain
         // where f is determined so that the solution is u = cos(pi*x/4)cos(pi*y/4)
       }

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i){

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_mat(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            2*M_PI_4*M_PI_4*cos(M_PI_4*fe_values.quadrature_point (q_index)(0))*cos(M_PI_4*fe_values.quadrature_point(q_index)(1)) * // f(x_q)
                            fe_values.JxW(q_index)); } }          // dx
        }
    // Assembling and solving further

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        system_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           cell_mat[i][j]);


    for (unsigned int i=0; i<dofs_per_cell; ++i)
      system_rhs(local_dof_indices[i]) += cell_rhs[i];
    }


  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (*dof_handler,
                                            0,
                                            Functions::ZeroFunction<2>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  DataOut<2> data_out;
  data_out.attach_dof_handler (*dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output ("solution.gpl");
  data_out.write_gnuplot (output);

}

int
main(int argc, char* argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    initlog();
    Temperature_field_full_fluid();

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
