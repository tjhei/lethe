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

void test1_square_integral()
{
  // Create a dummy empty triangulation
  Triangulation<2> triangulation (typename Triangulation<2>::MeshSmoothing
                                 (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));


  // Create triangulation points
  std::vector<Point<2> > triangulation_points(GeometryInfo<2>::vertices_per_cell);
  // Create 4 random points:
  triangulation_points[0]=Point<2>(0,0);
  triangulation_points[1]=Point<2>(0.5,0.2);
  triangulation_points[2]=Point<2>(0.1,0.2);
  triangulation_points[3]=Point<2>(1,1);

  // Prepare cell data
  std::vector<CellData<2> > cells (1);
  for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i)
      cells[0].vertices[i] = i;
  cells[0].material_id = 0;

  // Create a triangulation with that point
  triangulation.create_triangulation (triangulation_points, cells, SubCellData());

  // Create a dummy dof_handler to handle the 4 dofts
  DoFHandler<2>                  dof_handler(triangulation);
  // Create a FE system for this element
  FESystem<2>                    fe(FE_Q<2>(1),1);

  // Create a mapping for this new element
  const MappingQ<2>      mapping (1);

  // Quadrature on this new element
  QGauss<2>  quadrature_formula(4);



  // Integrate over this element, in this case we only integrate
  // over the quadrature to calculate the area
  FEValues<2> fe_values (fe, quadrature_formula,
                         update_quadrature_points | update_JxW_values);
  const unsigned int   n_q_points    = quadrature_formula.size();
  double area=0;

  typename DoFHandler<2>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      area+=fe_values.JxW (q_index);
    }
  }

  std::cout << " The area of the element is " << area <<  std::endl;

  // Write the result to paraview to visualize the element itself
  DataOut<2> data_out;

  data_out.attach_dof_handler (dof_handler);

  data_out.build_patches ();

  std::string fname="parallepiped.vtk";

  std::ofstream output (fname.c_str());
  data_out.write_vtk (output);
}


int
main(int argc, char* argv[])
{
  try
  {
    initlog();
    test1_square_integral();
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

