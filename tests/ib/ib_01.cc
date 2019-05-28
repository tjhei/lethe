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
#include "write_data.h"
#include "../tests.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds



using namespace dealii;

void test1_distance_verification()
{
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
  unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
  unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

  parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                         (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             -1,1);
  triangulation.refine_global(2);
  DoFHandler<2>                  dof_handler(triangulation);
  FESystem<2>                    fe(FE_Q<2>(2), 1);

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  IndexSet                         locally_owned_dofs;
  IndexSet                         locally_relevant_dofs;
  locally_owned_dofs = dof_handler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler,
                                           locally_relevant_dofs);
  TrilinosWrappers::MPI::Vector    levelSet_distance;
  levelSet_distance.reinit (locally_owned_dofs, mpi_communicator);

  const MappingQ<2>      mapping (1);
  std::map< types::global_dof_index, Point< 2 > > support_points;
  //support_points.(locally_owned_dofs.size());
  DoFTools::map_dofs_to_support_points
      ( mapping, dof_handler,support_points );

  Point<2> center(0,0);
  Tensor<1,2> velocity;
  velocity[0]=1.;
  velocity[1]=0.;
  Tensor<1,3> angular;
  angular[0]=0;
  angular[1]=0;
  angular[2]=1;
  double radius =0.3;

  IBLevelSetCircle<2> circle(center,velocity,angular,radius);
  auto d = locally_owned_dofs.begin(), enddof=locally_owned_dofs.end();
  for (; d!=enddof;++d)
  {
    levelSet_distance(*d)=circle.distance(support_points[*d]);
    //levelSet_velocity(*d)=circle.velocity(support_points[*d]);
  }

  TrilinosWrappers::MPI::Vector    levelSet_distance_global;
  levelSet_distance_global.reinit(locally_relevant_dofs);
  levelSet_distance_global=levelSet_distance;
  d = locally_relevant_dofs.begin();
  enddof=locally_relevant_dofs.end();
  for (; d!=enddof;++d)
  {
    double x = support_points[*d][0];
    double y = support_points[*d][1];
    double analytical_distance = std::sqrt(x*x+y*y)-radius;
    double imposed_distance = levelSet_distance_global[*d];
    if (!approximatelyEqual(analytical_distance,imposed_distance,1.e-10))
    {
      throw std::runtime_error("Test 1 - Level Set distance is not imposed correctly");
    }
  }

  write_ib_scalar_data<2>(triangulation,dof_handler,mpi_communicator,levelSet_distance_global,"ls_distance");
}

void test2_velocity_verification()
{
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
  unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
  unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

  parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                         (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             -2,2);
  triangulation.refine_global(2);
  DoFHandler<2>                  dof_handler(triangulation);
  FESystem<2>                    fe(FE_Q<2>(2), 2);

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  IndexSet                         locally_owned_dofs;
  IndexSet                         locally_relevant_dofs;
  locally_owned_dofs = dof_handler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler,
                                           locally_relevant_dofs);
  TrilinosWrappers::MPI::Vector    levelSet_velocity;
  levelSet_velocity.reinit (locally_owned_dofs, mpi_communicator);

  const MappingQ<2>      mapping (1);
  std::map< types::global_dof_index, Point< 2 > > support_points;
  //support_points.(locally_owned_dofs.size());
  DoFTools::map_dofs_to_support_points
      ( mapping, dof_handler,support_points );

  Point<2> center(0,0);
  Tensor<1,2> velocity;
  velocity[0]=1.;
  velocity[1]=0.;
  Tensor<1,3> angular;
  angular[0]=0;
  angular[1]=0;
  angular[2]=1;
  double radius =1.;

  IBLevelSetCircle<2> circle(center,velocity,angular,radius);
  Vector<double> velocity_values(2);
  //std::cout << "number of dofs " << dof_handler.n_dofs() << std::endl;
  auto d = locally_owned_dofs.begin(), enddof=locally_owned_dofs.end();
  for (; d!=enddof;++d)
  {
    circle.velocity(support_points[*d],velocity_values);
    //std::cout << "Pt : " << *d << " - " << support_points[*d] << std::endl;
    levelSet_velocity[*d]=velocity_values[0];
    //std::cout << "Velocity value : " << levelSet_velocity[*d] << std::endl;
    ++d;
    //std::cout << "Pt : " << *d << " - " << support_points[*d] << std::endl;
    levelSet_velocity[*d]=velocity_values[1];
    //std::cout << "Velocity value : " << levelSet_velocity[*d] << std::endl;
  }

  TrilinosWrappers::MPI::Vector    levelSet_velocity_global;
  levelSet_velocity_global.reinit(locally_relevant_dofs);
  levelSet_velocity_global=levelSet_velocity;
  d = locally_relevant_dofs.begin();
  enddof=locally_relevant_dofs.end();
  for (; d!=enddof;++d)
  {
    double x = support_points[*d][0];
    double y = support_points[*d][1];
    double analytical_velocity_x = -angular[2] * y + velocity[0];
    double analytical_velocity_y = angular[2] * x + velocity[1];

    double imposed_velocity = levelSet_velocity_global[*d];
    if (!approximatelyEqual(analytical_velocity_x,imposed_velocity,1.e-10))
    {
      throw std::runtime_error("Test 2 - Level Set X velocity is not imposed correctly");
    }
    ++d;
    imposed_velocity = levelSet_velocity_global[*d];
    if (!approximatelyEqual(analytical_velocity_y,imposed_velocity,1.e-10))
    {
      throw std::runtime_error("Test 2 - Level Set Y velocity is not imposed correctly");
    }
  }

  write_ib_vector_data<2>(triangulation,dof_handler,mpi_communicator,levelSet_velocity_global,"ls_velocity");
}

void test3_complete_verification()
{
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
  unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
  unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

  parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                         (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             -2,2);
  triangulation.refine_global(2);
  DoFHandler<2>                  dof_handler(triangulation);
  FESystem<2>                    fe(FE_Q<2>(1),1,FE_Q<2>(1), 2);

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  IndexSet                         locally_owned_dofs;
  IndexSet                         locally_relevant_dofs;
  locally_owned_dofs = dof_handler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler,
                                           locally_relevant_dofs);
  TrilinosWrappers::MPI::Vector    levelSet_ib;
  levelSet_ib.reinit (locally_owned_dofs, mpi_communicator);

  const MappingQ<2>      mapping (1);
  std::map< types::global_dof_index, Point< 2 > > support_points;
  //support_points.(locally_owned_dofs.size());
  DoFTools::map_dofs_to_support_points
      ( mapping, dof_handler,support_points );

  Point<2> center(0,0);
  Tensor<1,2> velocity;
  velocity[0]=1.;
  velocity[1]=0.;
  Tensor<1,3> angular;
  angular[0]=0;
  angular[1]=0;
  angular[2]=1;
  double radius =1.;

  IBLevelSetCircle<2> circle(center,velocity,angular,radius);
  Vector<double> velocity_values(2);
  //std::cout << "number of dofs " << dof_handler.n_dofs() << std::endl;
  auto d = locally_owned_dofs.begin(), enddof=locally_owned_dofs.end();
  for (; d!=enddof;++d)
  {
//    std::cout << "Pt : " << *d << " - " << support_points[*d] << std::endl;
    levelSet_ib[*d]=circle.distance(support_points[*d]);
    ++d;
    circle.velocity(support_points[*d],velocity_values);
//    std::cout << "Pt : " << *d << " - " << support_points[*d] << std::endl;
    levelSet_ib[*d]=velocity_values[0];
    ++d;
//    std::cout << "Pt : " << *d << " - " << support_points[*d] << std::endl;
    levelSet_ib[*d]=velocity_values[1];
  }

  d = locally_owned_dofs.begin();
  enddof=locally_owned_dofs.end();
  for (; d!=enddof;++d)
  {
    double x = support_points[*d][0];
    double y = support_points[*d][1];
    double analytical_velocity_x = -angular[2] * y + velocity[0];
    double analytical_velocity_y = angular[2] * x + velocity[1];
    double analytical_distance = std::sqrt(x*x+y*y)-radius;
    double imposed_distance = levelSet_ib[*d];
    if (!approximatelyEqual(analytical_distance,imposed_distance,1.e-10))
    {
      throw std::runtime_error("Test 3 - Level Set distance is not imposed correctly");
    }
    ++d;
    double imposed_velocity = levelSet_ib[*d];
    if (!approximatelyEqual(analytical_velocity_x,imposed_velocity,1.e-10))
    {
      throw std::runtime_error("Test 3 - Level Set X velocity is not imposed correctly");
    }
    ++d;
    imposed_velocity = levelSet_ib[*d];
    if (!approximatelyEqual(analytical_velocity_y,imposed_velocity,1.e-10))
    {
      throw std::runtime_error("Test 3 - Level Set Y velocity is not imposed correctly");
    }
  }
  TrilinosWrappers::MPI::Vector    levelSet_ib_global;
  levelSet_ib_global.reinit(locally_relevant_dofs);
  levelSet_ib_global=levelSet_ib;
  write_ib_data<2>(triangulation,dof_handler,mpi_communicator,levelSet_ib_global,"ls_all");
}


int
main(int argc, char* argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    initlog();
    test1_distance_verification();
    test2_velocity_verification();
    test3_complete_verification();
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
