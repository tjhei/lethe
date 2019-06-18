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

#include <thread>         // std::this_thread::sleep_for
#include <chrono>        // std::chrono::seconds



using namespace dealii;

void quad_elemf(std::vector<Point<2>> coor, FullMatrix<double> &cell_mat, std::vector<double> &sec_membre_elem)
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
                sec_membre_elem[i] += 0 ; /*  fe_values.shape_value (i, q_index) *
                                          1  * 2 * M_PI_4 * M_PI_4 * cos(M_PI_4 * quadpt[0]) * cos(M_PI_4 * quadpt[1]) *
                                          fe_values.JxW (q_index); */
                //std::cout << "valeur de shape fct "<< i << " au pt " << quadpt << " : " << fe_values.shape_value (i, q_index) << std::endl;
            }


  }
}
//  for (int i = 0; i < 4; ++i) {
//      std::cout << "Ligne " << i << " de la matrice : " << cell_mat[i][0] << ", " << cell_mat[i][1] << ", " << cell_mat[i][2] << ", " << cell_mat[i][3] << std::endl;
//  }
//   std::cout << "RHS : " << sec_membre_elem[0] << ", " << sec_membre_elem[1] << ", " << sec_membre_elem[2] << ", " << sec_membre_elem[3] << std::endl;
//  std::cout << "\n" << std::endl;

}





void quad_elems(double Tdirichlet, std::vector<Point<2>> coor, FullMatrix<double> &cell_mat, std::vector<double> &sec_membre_elem)
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
                if (i==j)
                  cell_mat[i][j] = 1;
                else {
                  cell_mat[i][j] = 0;
                }

          }
                  //quadpt = fe_values.quadrature_point (q_index);
                  sec_membre_elem[i] += Tdirichlet;
                  //std::cout << "valeur de shape fct "<< i << " au pt " << quadpt << " : " << fe_values.shape_value (i, q_index) << std::endl;


    }

    }




}






void quad_elem_mix(double Tdirichlet, std::vector<int> No_pts_solid, std::vector<int> corresp, std::vector<Point<2>> decomp_elem, FullMatrix<double> &cell_mat, std::vector<double> &cell_rhs)
{

    // Create a dummy empty triangulation
    Triangulation<2> triangulation (typename Triangulation<2>::MeshSmoothing
                                   (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));


    // Create triangulation points
    std::vector<Point<2> > triangulation_points(GeometryInfo<2>::vertices_per_cell);
    // Create 4 random points:
    triangulation_points[0]=decomp_elem[0];
    triangulation_points[1]=decomp_elem[1];
    triangulation_points[2]=decomp_elem[2];
    triangulation_points[3]=decomp_elem[3];

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

    double M[6][6] = {{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}};

    int vec_sol[6] = {1,1,1,1,1,1};
    vec_sol[No_pts_solid[0]] = -1;
    vec_sol[No_pts_solid[1]] = -1;


    typename DoFHandler<2>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      for (unsigned int i = 0; i < 2; ++i) {

          // The coefficient associated to the summits in the solid are set to 0 except for the one on the diagona

              M[No_pts_solid[i]][No_pts_solid[i]]=1;
              cell_rhs[No_pts_solid[i]] = Tdirichlet;
          }

          for (unsigned int i = 0; i < 4; ++i) {
              for (unsigned int j = 0; j < 4; ++j) {
                  for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
                  {
                      M[corresp[i]][corresp[j]] += fe_values.shape_grad(i, q_index) * fe_values.shape_grad (j, q_index) * fe_values.JxW (q_index);
                  }
                  //std::cout << " corresp : " << corresp[i] << ", " << corresp[j] << std::endl;



              }

          }



      for (unsigned int i = 0; i < 4; ++i) {
          for (unsigned int var = 0; var < 4; ++var) {
              cell_mat[i][var] = M[i][var];

          }std::cout <<  M[i][4] << " " << M[i][5] << std::endl;
          cell_rhs[i] += -Tdirichlet * (M[i][4] + M[i][5]);
      }




    }




}



