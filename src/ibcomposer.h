//LACS
#include <deal.II/lac/trilinos_vector.h>

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

#include "iblevelsetfunctions.h"

#include <deal.II/base/timer.h>


using namespace dealii;


#ifndef LETHE_IBCOMPOSER_H
#define LETHE_IBCOMPOSER_H

template <int dim>
class IBComposer
{
public:
  IBComposer(parallel::distributed::Triangulation<dim> * p_triangulation, std::vector<IBLevelSetFunctions<dim>* > p_functions):
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(p_triangulation),
    fe(FE_Q<dim>(1), 1),
    dof_handler(*triangulation),
    functions(p_functions)
  {
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);
    levelSet_distance.reinit(locally_relevant_dofs,mpi_communicator);
    levelSet_distance_local.reinit(locally_relevant_dofs,mpi_communicator);
  }

  void setFunctions(std::vector<IBLevelSetFunctions<dim>* > p_functions)
  {
    functions=p_functions;
  }

  void calculateDistance()
  {
    const MappingQ<2>      mapping (1);
    std::map< types::global_dof_index, Point< 2 > > support_points;
    DoFTools::map_dofs_to_support_points ( mapping, dof_handler,support_points );
    auto d = locally_owned_dofs.begin(), enddof=locally_owned_dofs.end();
    for (; d!=enddof;++d)
    {
      double dist=DBL_MAX;
      for (unsigned ib=0 ; ib < functions.size() ; ++ib)
        dist = std::min(dist,functions[ib]->distance(support_points[*d]));

      levelSet_distance(*d)=dist;
    }
  }

  // Distance vector
  TrilinosWrappers::MPI::Vector getDistance(){return levelSet_distance;}
  DoFHandler<dim>*              getDoFHandler(){return &dof_handler;}
  FESystem<dim>*                getFESystem(){return &fe;}
private:
  MPI_Comm                                    mpi_communicator;
  parallel::distributed::Triangulation<dim> * triangulation;
  FESystem<dim>                               fe;
  DoFHandler<dim>                             dof_handler;
  IndexSet                                    locally_owned_dofs;
  IndexSet                                    locally_relevant_dofs;
  std::vector<IBLevelSetFunctions<dim>* >     functions;
  TrilinosWrappers::MPI::Vector               levelSet_distance_local;
  TrilinosWrappers::MPI::Vector               levelSet_distance;

};



#endif // IBCOMPOSER_H
