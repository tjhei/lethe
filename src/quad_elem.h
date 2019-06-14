//#include <deal.II/lac/trilinos_vector.h>

////NUMERICS
//#include <deal.II/numerics/data_out.h>

////DOFS
//#include <deal.II/dofs/dof_handler.h>
//#include <deal.II/dofs/dof_renumbering.h>
//#include <deal.II/dofs/dof_accessor.h>
//#include <deal.II/dofs/dof_tools.h>

////FE
//#include <deal.II/fe/fe_system.h>
//#include <deal.II/fe/fe_values.h>
//#include <deal.II/fe/fe_q.h>
//#include <deal.II/fe/mapping_q.h>

//// Distributed
//#include <deal.II/distributed/tria.h>

//#include "iblevelsetfunctions.h"

//#include <thread>         // std::this_thread::sleep_for
//#include <chrono>        // std::chrono::seconds



//using namespace dealii;

void quad_elem(std::vector<Point<2>> coor, std::vector<std::vector<double>> &cell_mat, std::vector<double> &sec_membre_elem)
{
  // Create a dummy empty triangulation
  Triangulation<2> triangulation (typename Triangulation<2>::MeshSmoothing
                                 (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));


  // Create triangulation points
  std::vector<Point<2> > triangulation_points(GeometryInfo<2>::vertices_per_cell);
  // Create 4 random points:
  triangulation_points[0]=coor[0];
  triangulation_points[1]=coor[1];
  triangulation_points[2]=coor[2];
  triangulation_points[3]=coor[3];

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
  QGauss<2>  quadrature_formula(8);

  Point<2> quadpt;

  // Integrate over this element, in this case we only integrate
  // over the quadrature to calculate the area
  FEValues<2> fe_values (fe, quadrature_formula, update_values | update_gradients |
                         update_quadrature_points | update_JxW_values);
  const unsigned int   n_q_points    = quadrature_formula.size();

  typename DoFHandler<2>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
            {
                cell_mat[i][j] += fe_values.shape_grad(i, q_index) * fe_values.shape_grad (j, q_index) * fe_values.JxW (q_index);
            }
        }
            for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
            {
                quadpt = fe_values.quadrature_point (q_index);
                sec_membre_elem[i] +=  fe_values.shape_value (i, q_index) *
                                          1 * 2 * M_PI_4 * M_PI_4 * cos(M_PI_4 * quadpt[0]) * cos(M_PI_4 * quadpt[1]) *
                                          fe_values.JxW (q_index);
                //std::cout << "valeur de shape fct "<< i << " au pt " << quadpt << " : " << fe_values.shape_value (i, q_index) << std::endl;
            }


  }
}
//  for (int i = 0; i < 4; ++i) {
//      std::cout << "Ligne " << i << " de la matrice : " << cell_mat[i][0] << ", " << cell_mat[i][1] << ", " << cell_mat[i][2] << ", " << cell_mat[i][3] << std::endl;
//  }
   std::cout << "RHS : " << sec_membre_elem[0] << ", " << sec_membre_elem[1] << ", " << sec_membre_elem[2] << ", " << sec_membre_elem[3] << std::endl;
  std::cout << "\n" << std::endl;

}
