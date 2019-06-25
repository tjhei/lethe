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

//Addings

#include "iblevelsetfunctions.h"
#include "t_calc_interp.h"
#include "T_analytical.h"

#include <thread>
#include <chrono>
#include "enum_solid_fluid.h"

using namespace dealii;


void quad_elem_mix(double Tdirichlet, std::vector<In_fluid_or_in_solid> No_pts_solid, std::vector<int> corresp, std::vector<Point<2>> decomp_elem, FullMatrix<double> &cell_mat, std::vector<double> &cell_rhs)
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

    int num_pts_solid[4]={-1,-1,-1,-1};
    int a =0;
    for (int i = 0; i < 4; ++i) {
        if (No_pts_solid[i]==solid)
        {
            num_pts_solid[a]=i;
            a+=1;
        }
        if (a==2)
            break;
    }

    typename DoFHandler<2>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      for (unsigned int i = 0; i < 2; ++i) {

          // The coefficient associated to the summits in the solid are set to 0 except for the one on the diagona

              M[num_pts_solid[i]][num_pts_solid[i]]=1;
              cell_rhs[num_pts_solid[i]] = Tdirichlet;
          }

          for (unsigned int i = 0; i < 4; ++i) {
              for (unsigned int j = 0; j < 4; ++j) {
                  for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
                  {
                      M[corresp[i]][corresp[j]] += fe_values.shape_grad(i, q_index) * fe_values.shape_grad (j, q_index) * fe_values.JxW (q_index);
                  }
              }
          }


      for (unsigned int i = 0; i < 4; ++i) {
          for (unsigned int var = 0; var < 4; ++var) {
              cell_mat[i][var] = M[i][var];

          }
          cell_rhs[i] += -Tdirichlet * (M[i][4] + M[i][5]);
      }
    }
}

double quad_elem_L2(Point<2> center, double T1, double T2, double r1, double r2, std::vector<int> corresp, std::vector<In_fluid_or_in_solid> pts_statut, std::vector<Point<2>> decomp_elem, std::vector<double> sol_loc)
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
    QGauss<2>  quadrature_formula(4);

    // Integrate over this element, in this case we only integrate
    // over the quadrature to calculate the area
    FEValues<2> fe_values (fe, quadrature_formula, update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);
    const unsigned int   n_q_points    = quadrature_formula.size();

    double err=0;

    typename DoFHandler<2>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    std::vector<double> T_quad(4);
    int No_pts_solid[4] = {-1,-1,-1,-1};
    int a=0;
    for (int i=0;i<4;i++) {
        if (pts_statut[i]==solid)
        {No_pts_solid[a]=i; a+=1;}
    }

    T_quad[0]=sol_loc[No_pts_solid[0]]; //the vertices in the solid have a set T
    T_quad[1]=sol_loc[No_pts_solid[0]];
    T_quad[2]=sol_loc[corresp[2]];
    T_quad[3]=sol_loc[corresp[3]];

    for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      for (unsigned int q=0; q<n_q_points; ++q)
      {
          err+=std::pow(T_analytical(fe_values.quadrature_point (q), center, 1, 2, r1, r2)-T_calc_interp(T_quad, fe_values.quadrature_point (q)), 2)
                  *fe_values.JxW(q);
      }
    }
    return err;
}

