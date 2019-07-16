/* ---------------------------------------------------------------------
 *  Unit test for the decomposition of elements into sub triangulations
 *  Test 1 - Decompose a square into a smaller square and integrate
 *  Test 2 - Decompose a square into a single triangle and integrate
 *  Test 3 - Decompose a square into three triangles and integrate
 *
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

#include "iblevelsetfunctions.h"
#include "write_data.h"
#include "../tests.h"

// Triangles decomposition tools
#include "nouvtriangles.h"

using namespace dealii;

void test1_square_subelement()
{
  // Create triangulation and square mesh
  Triangulation<2> triangulation (typename Triangulation<2>::MeshSmoothing
                                  (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             0,1);

  // Create a dummy dof_handler to handle the 4 dofts
  DoFHandler<2>                  dof_handler(triangulation);
  // Create a FE system for this element
  FESystem<2>                    fe(FE_Q<2>(1),1);
  dof_handler.distribute_dofs(fe);

  // Quadrature formula for the element
  QGauss<2>              quadrature_formula(1);

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

  // Create a distance array
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;         // Number of dofs per cells.
  Vector<double>  distance       (dofs_per_cell); // Array for the distances associated with the DOFS
  distance[0]=0.75;
  distance[1]=-0.25;
  distance[2]=0.5;
  distance[3]=-0.5;


  // Instantiations for the decomposition of the elements
  std::vector<int>                     corresp(9);
  Vector<Point<2> >                    decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
  Vector<node_status>                  No_pts_solid(4);
  int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
  std::vector<Point<2> >               num_elem(6);

  // Set the values of the dof points position
  Vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell
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
    }
    // Decompose the geometry
    nouvtriangles(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, dofs_points, distance);

    if (nb_poly != -1 ) throw std::runtime_error("Test 1 - Decomposition lead a non-quad element");

    std::cout << " Decomposed geometry is : " << nb_poly << std::endl;

    //Create new triangulation and integrate
    {
      // Create triangulation points
      std::vector<Point<2> > triangulation_points(GeometryInfo<2>::vertices_per_cell);
      // Create 4 random points:
      for (unsigned int i_pt =0 ; i_pt < 4 ; ++i_pt)
      {
        triangulation_points[i_pt]=decomp_elem[i_pt];
      }


      // Prepare cell data
      std::vector<CellData<2> > cells (1);
      for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i)
          cells[0].vertices[i] = i;
      cells[0].material_id = 0;

      Triangulation<2> sub_triangulation;
      sub_triangulation.create_triangulation (triangulation_points, cells, SubCellData());
      // Create a dummy dof_handler to handle the 4 dofts
      DoFHandler<2>                  sub_dof_handler(sub_triangulation);
      // Create a FE system for this element
      FESystem<2>                    sub_fe(FE_Q<2>(1),1);
      sub_dof_handler.distribute_dofs(fe);

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
    }
  }

  deallog << "Triangulation area is : " << area  << std::endl;
  std::cout << "Triangulation area is : " << area << std::endl;
  if (!approximatelyEqual(area,0.625,1e-10)) throw std::runtime_error("Test 1 - Wrong area for sub decomposed quad element");
}

void test2_single_triangle_subelement()
{
  // Create triangulation and square mesh
  Triangulation<2> triangulation (typename Triangulation<2>::MeshSmoothing
                                  (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             0,1);

  // Create a dummy dof_handler to handle the 4 dofts
  DoFHandler<2>                  dof_handler(triangulation);
  // Create a FE system for this element
  FESystem<2>                    fe(FE_Q<2>(1),1);
  dof_handler.distribute_dofs(fe);

  // Quadrature formula for the element
  QGauss<2>              quadrature_formula(1);

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

  // Create a distance array
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;         // Number of dofs per cells.
  Vector<double>  distance       (dofs_per_cell); // Array for the distances associated with the DOFS
  distance[0]=0.5;
  distance[1]=-0.5;
  distance[2]=-0.5;
  distance[3]=-sqrt(0.5*0.5+0.5*0.5);


  // Instantiations for the decomposition of the elements
  std::vector<int>                     corresp(9);
  Vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
  Vector<node_status>    No_pts_solid(4);
  int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
  std::vector<Point<2> >               num_elem(6);

  // Set the values of the dof points position
  Vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell
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
    }
    // Decompose the geometry
    nouvtriangles(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, dofs_points, distance);

    if (nb_poly != 1 ) throw std::runtime_error("Test 2 - Decomposition did not lead to a single triangle");

    std::cout << " Decomposed geometry is : " << nb_poly << std::endl;

    //Create new triangulation and integrate
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
      // Create a dummy dof_handler to handle the 4 dofts
      DoFHandler<2>                  sub_dof_handler(sub_triangulation);
      // Create a FE system for this element
      FESystem<2>                    sub_fe(FE_Q<2>(1),1);
      sub_dof_handler.distribute_dofs(sub_fe);

      // Create a mapping for this new element
      const MappingQ<2>      sub_mapping (1);
      QGauss<2>              sub_quadrature_formula(1);

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
    }
  }

  if (!approximatelyEqual(area,0.125,1e-10)) throw std::runtime_error("Test 2 - Wrong area for sub decomposed one tri element");
  deallog << "Triangulation area is : " << area  << std::endl;
  std::cout << "Triangulation area is : " << area << std::endl;
}

void test3_triple_triangle_subelement()
{
  // Create triangulation and square mesh
  Triangulation<2> triangulation (typename Triangulation<2>::MeshSmoothing
                                  (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             0,1);

  // Create a dummy dof_handler to handle the 4 dofts
  DoFHandler<2>                  dof_handler(triangulation);
  // Create a FE system for this element
  FESystem<2>                    fe(FE_Q<2>(1),1);
  dof_handler.distribute_dofs(fe);

  // Quadrature formula for the element
  QGauss<2>              quadrature_formula(1);

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

  // Create a distance array
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;         // Number of dofs per cells.
  Vector<double>  distance       (dofs_per_cell); // Array for the distances associated with the DOFS
  distance[0]=sqrt(0.5*0.5+0.5*0.5);
  distance[1]=0.5;
  distance[2]=0.5;
  distance[3]=-0.5;



  // Instantiations for the decomposition of the elements
  std::vector<int>                     corresp(9);
  Vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
  Vector<node_status>                     No_pts_solid(4);
  int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
  std::vector<Point<2> >               num_elem(6);

  // Set the values of the dof points position
  Vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell
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
    }
    // Decompose the geometry
    nouvtriangles(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, dofs_points, distance);

    if (nb_poly != 3 ) throw std::runtime_error("Test 3 - Decomposition did not lead to a single triangle");

    std::cout << " Decomposed geometry is : " << nb_poly << std::endl;

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
      // Create a dummy dof_handler to handle the 4 dofts
      DoFHandler<2>                  sub_dof_handler(sub_triangulation);
      // Create a FE system for this element
      FESystem<2>                    sub_fe(FE_Q<2>(1),1);
      sub_dof_handler.distribute_dofs(sub_fe);

      // Create a mapping for this new element
      const MappingQ<2>      sub_mapping (1);
      QGauss<2>              sub_quadrature_formula(1);

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
    }
  }

  if (!approximatelyEqual(area,0.875,1e-10)) throw std::runtime_error("Test 3 - Wrong area for sub decomposed quad element");
  deallog << "Triangulation area is : " << area  << std::endl;
  std::cout << "Triangulation area is : " << area << std::endl;
}

int
main(int argc, char* argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    initlog();
    test1_square_subelement();
    test2_single_triangle_subelement();
    test3_triple_triangle_subelement();
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
