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


#ifndef LETHE_WRITE_IB_DATA_H
#define LETHE_WRITE_IB_DATA_H

using namespace dealii;

template <int dim>
void write_ib_scalar_data(parallel::distributed::Triangulation<dim> &triangulation,
                   DoFHandler<dim> &dof_handler,
                   MPI_Comm &mpi_communicator,
                   TrilinosWrappers::MPI::Vector &levelSet_vector_global,
                   std::string fname
                   )
{
  const MappingQ<dim>      mapping (1);
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");
  data_out.add_data_vector (levelSet_vector_global, "Level_Set");

  data_out.build_patches (mapping);
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
}

template <int dim>
void write_ib_vector_data(parallel::distributed::Triangulation<dim> &triangulation,
                   DoFHandler<dim> &dof_handler,
                   MPI_Comm &mpi_communicator,
                   TrilinosWrappers::MPI::Vector &levelSet_vector_global,
                   std::string fname
                   )
{
  const MappingQ<dim>      mapping (1);
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  std::vector<std::string> solution_names (dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (levelSet_vector_global, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);

  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");


  data_out.build_patches (mapping);
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
}

template <int dim>
void write_ib_data(parallel::distributed::Triangulation<dim> &triangulation,
                   DoFHandler<dim> &dof_handler,
                   MPI_Comm &mpi_communicator,
                   TrilinosWrappers::MPI::Vector &levelSet_vector_global,
                   std::string fname
                   )
{
  const MappingQ<dim>      mapping (1);
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  std::vector<std::string> solution_names ("distance");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(1,DataComponentInterpretation::component_is_scalar);
  for (int i = 0 ; i <dim ; ++i)
  {
    solution_names.push_back("velocity");
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  }
  data_out.add_data_vector (levelSet_vector_global, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);

  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");


  data_out.build_patches (mapping);
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
}

#endif
