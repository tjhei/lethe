#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>

//LACS
#include <deal.II/lac/trilinos_vector.h>

//NUMERICS
#include <deal.II/numerics/data_out.h>



//DOFS
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

//FE
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

// Distributed
#include <deal.II/distributed/tria.h>

#include "iblevelsetfunctions.h"

#include "../tests.h"

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

using namespace dealii;

int
main(int argc, char* argv[])
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
    unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
    unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

    parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                           (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));

    GridGenerator::hyper_cube (triangulation,
                               -1,1);

    triangulation.refine_global(7);


    DoFHandler<2>                  dof_handler(triangulation);
    FESystem<2>                    fe(FE_Q<2>(2), 1);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    IndexSet                         locally_owned_dofs;
    IndexSet                         locally_relevant_dofs;
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);
    TrilinosWrappers::MPI::Vector    levelSet_vector;
    levelSet_vector.reinit (locally_owned_dofs, mpi_communicator);
    levelSet_vector=-7;

    const MappingQ<2>      mapping (1);
    std::map< types::global_dof_index, Point< 2 > > support_points;
    //support_points.(locally_owned_dofs.size());
    DoFTools::map_dofs_to_support_points
        ( mapping, dof_handler,support_points );

    Point<2> center(0,0);
    Tensor<1,2> velocity;
    velocity[0]=0.;
    velocity[1]=0.;
    Tensor<1,3> angular;
    angular[0]=0;
    angular[1]=0;
    angular[2]=0;

    IBLevelSetCircle<2> circle(center,velocity,angular,0.3);
    auto d = locally_owned_dofs.begin(), enddof=locally_owned_dofs.end();
    for (; d!=enddof;++d)
    {
      levelSet_vector(*d)=circle.distance(support_points[*d]);
//      std::cout << "---------------" <<  std::endl;

//      std::cout << "MPI process : "<< this_mpi_process << " " << " dof no       : " << *d << std::endl;

//      std::cout << " Value of VEC : " << levelSet_vector(*d) << std::endl;
//      levelSet_vector(*d)=999.;
//      std::cout << " Value of VEC : " << levelSet_vector(*d) << std::endl;
//      std::cout << " Support point: " << support_points[*d]  << std::endl;
//      std::cout << " Value of VEC : " << levelSet_vector(*d) << std::endl;

//      std::cout << "---------------" << std::endl;

    }
//    std::cout << "finished " << this_mpi_process <<std::endl;
//    std::this_thread::sleep_for (std::chrono::seconds(3+this_mpi_process*2));

    //DataOut<2> data_out;
    //
    //data_out.attach_dof_handler (dof_handler);
    //data_out.add_data_vector (levelSet_vector, "Level_Set");
    //
    //data_out.build_patches ();
    //
    //std::string fname= "level_set.vtk";
    //
    //std::ofstream output (fname.c_str());
    //data_out.write_vtk (output);


    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");
    TrilinosWrappers::MPI::Vector    levelSet_vector_global;
    levelSet_vector_global.reinit(locally_relevant_dofs);
    levelSet_vector_global=levelSet_vector;
    data_out.add_data_vector (levelSet_vector_global, "Level_Set");



    data_out.build_patches (mapping);
    std::string fname= "level_set";

    const std::string filename = (fname+
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;
           i<Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.push_back (fname+
                             "." +
                             Utilities::int_to_string (i, 4) +
                             ".vtu");

      std::string pvtu_filename = (fname+
                                   ".pvtu");
      std::ofstream master_output ((pvtu_filename).c_str());
      data_out.write_pvtu_record (master_output, filenames);

    }

//    DoFTools::map_dofs_to_support_points()
    //const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    //std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    //typename DoFHandler<2>::active_cell_iterator
    //cell = dof_handler.begin_active(),
    //endc = dof_handler.end();
    //for (; cell!=endc; ++cell)
    //{
    //    if (cell->is_locally_owned())
    //    {
    //      cell->get_dof_indices (local_dof_indices);
    //      for (unsigned int d=0 ; d<dofs_per_cell ; ++d)
    //      {
    //        std::cout << "d " << local_dof_indices[d] << "position " << cell-> << std::endl;
    //      }
    //
    //    }
    //}

    //typename DoFHandler<2>::
    //levelSet_vector.begin()



    //Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    //
    //ParameterHandler prm;
    //NavierStokesSolverParameters<2> NSparam;
    //NSparam.declare(prm);
    //NSparam.parse(prm);
    //
    //// Manually alter some of the default parameters of the solver
    //NSparam.restartParameters.checkpoint=true;
    //NSparam.restartParameters.frequency=1;
    //NSparam.nonLinearSolver.verbosity=Parameters::NonLinearSolver::quiet;
    //NSparam.linearSolver.verbosity=Parameters::LinearSolver::quiet;
    //NSparam.boundaryConditions.createDefaultNoSlip();
    //
    //RestartNavierStokes<2> problem_2d(NSparam,NSparam.femParameters.velocityOrder,NSparam.femParameters.pressureOrder);
    //problem_2d.run();
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
