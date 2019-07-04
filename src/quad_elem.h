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
#include "ib_node_status.h"

using namespace dealii;


void condensate_quad(unsigned int dofs_per_cell, std::vector<node_status> status, std::vector<int> corresp, FullMatrix<double> mat6, FullMatrix<double> &mat4, std::vector<double> rhs6, std::vector<double> &rhs4)
{
    // The point of this function is to change the order of the lines in order to put the lines corresponding to the points in the fluid at the top of the matrix, which will make the condensation much easier
    int a[2];
    int c =0;
    for (int i = 0; i < 4; ++i) {
        if (status[i])
        {
            a[c]=i;
            c++;
        }
    }

    int num_loc[6] = {corresp[3], corresp[2], a[0], a[1], corresp[1], corresp[0]}; // new numerotation with shows the vertices in the fluid first

    FullMatrix<double> M6_renum(dofs_per_cell+2, dofs_per_cell+2);
    std::vector<double> rhs6_renum(dofs_per_cell+2);

    double acc = 1e-8;

    for (unsigned int i = 0; i < dofs_per_cell+2; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell+2; ++j) {
            M6_renum[i][j] = mat6[num_loc[i]][num_loc[j]];
        }
        rhs6_renum[i] = rhs6[num_loc[i]];
        if (std::abs(M6_renum[i][i])<acc) throw std::runtime_error("One of the diagonal terms of the 6x6 matrix is too close to 0");
    }
//    std::cout << "\n RHS 6 avant opé : " <<  rhs6_renum[0] << " " << rhs6_renum[1] << " " << rhs6_renum[2] << " " << rhs6_renum[3] << " " << rhs6_renum[4] << " " << rhs6_renum[5] << std::endl;

    int count =0;
    for (int k = dofs_per_cell /* dofs_per_cell starts at one, not 0 !! */ ; k >-1 ; --k) {
        for (int i = 0; i < count+1; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell+1-i; ++j) {
                M6_renum[k][j]=M6_renum[k][j]-M6_renum[dofs_per_cell+1-i][j] *
                        M6_renum[k][dofs_per_cell+1-i] / M6_renum[dofs_per_cell+1-i][dofs_per_cell+1-i];

//                std::cout << "\n coeff " << k << " " << j << " : " << M6_renum[k][j];

                // for example, if we have 4 dofs, we will have a 6x6 matrix
                //let's take the fifth line : M6_renum[4][0] += -M6_renum[5][0]*M6_renum[4][5]/M6_renum[5][5]
            }
//            std::cout << "\n " << k << " rh6 coeff " << -rhs6_renum[dofs_per_cell+1-i] *
//                         M6_renum[k][dofs_per_cell+1-i]/M6_renum[dofs_per_cell+1-i][dofs_per_cell+1-i] << std::endl;

            rhs6_renum[k]+=-rhs6_renum[dofs_per_cell+1-i] *
                    M6_renum[k][dofs_per_cell+1-i]/M6_renum[dofs_per_cell+1-i][dofs_per_cell+1-i];
        }
        count ++;
    }
//std::cout << "\n  mat 6x6" << std::endl;
//    for (int i = 0; i < 6; ++i) {
//        std::cout <<  M6_renum[i][0] << " " << M6_renum[i][1] << " " << M6_renum[i][2] << " " << M6_renum[i][3] << " " << M6_renum[i][4] << " " << M6_renum[i][5] << std::endl;
//    }
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < i+1; ++j) {
            mat4[num_loc[i]][num_loc[j]]=M6_renum[i][j];
        }
        rhs4[num_loc[i]]=rhs6_renum[i];
    }
//    std::cout << "\n RHS 6 post opé : " <<  rhs6_renum[0] << " " << rhs6_renum[1] << " " << rhs6_renum[2] << " " << rhs6_renum[3] << " " << rhs6_renum[4] << " " << rhs6_renum[5] << std::endl;

//    std::cout << "\n mat 4x4" << std::endl;
//        for (int i = 0; i < 4; ++i) {
//            std::cout <<  mat4[i][0] << " " << mat4[i][1] << " " << mat4[i][2] << " " << mat4[i][3] << std::endl;
//        }
//        std::cout << "\n rhs " << std::endl;
//        std::cout <<  rhs4[0] << " " << rhs4[1] << " " << rhs4[2] << " " << rhs4[3] << std::endl;
}


void quad_elem_mix(double Tdirichlet, std::vector<node_status> No_pts_solid, std::vector<int> corresp, std::vector<Point<2>> decomp_elem, FullMatrix<double> &cell_mat, std::vector<double> &cell_rhs)
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

    ///


    bool test_condensate = 1;


    ///


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




          FullMatrix<double> m4(4,4);
          std::vector<double> rhs4(4);

          ///
          if (test_condensate){



              std::vector<double> rhs6(6);

              std::fill(rhs4.begin(), rhs4.end(),0);
              m4=0;
              std::fill(rhs6.begin(), rhs6.end(),0);

              for (int j = 0; j < 6; ++j) {
                   for (int i = 4; i < 6; ++i){
                      if (i==j) M[i][i]=1;
                      else {
                          M[i][j]=0;
                      }
                  }
              }

//              std::cout << "\n \n \n MAT 6x6 originale " << std::endl;
//              for (int i = 0; i < 6; ++i) {
//                  std::cout <<  M[i][0] << " " << M[i][1] << " " << M[i][2] << " " << M[i][3] << " " << M[i][4] << " " << M[i][5] << std::endl;
//              }


              FullMatrix<double> m6(6,6);

//              std::cout << "\n \n mat 6x6 avant cond " << std::endl;
              for (int i = 0; i < 6; ++i) {
                  for (int j = 0; j < 6; ++j) {
                      m6(i,j)=M[i][j];
                  }
//                  std::cout << m6[i][0] << " " << m6[i][1] << " " << m6[i][2] << " " << m6[i][3] << " " << m6[i][4] << " " << m6[i][5] << std::endl;
              }

              for (int i = 0; i < 4; ++i) {
                  rhs6[i]=cell_rhs[i];
              }
              rhs6[4]=Tdirichlet;
              rhs6[5]=Tdirichlet;
//              std::cout << " \n \n cond" << std::endl;


              condensate_quad(4, No_pts_solid, corresp, m6, m4, rhs6,rhs4);

              for (unsigned int i = 0; i < 4; ++i) {
                for (unsigned int var = 0; var < 4; ++var) {
                  cell_mat[i][var] = m4[i][var];
                }
                cell_rhs[i] = rhs4[i];
              }
          }
      ///

      //      for (unsigned int i = 0; i < 4; ++i) {
      //          for (unsigned int var = 0; var < 4; ++var) {
      //              cell_mat[i][var] = M[i][var];
      //          }
      //          cell_rhs[i] += -Tdirichlet * (M[i][4] + M[i][5]);
      //      }
    }
}

double quad_elem_L2(Point<2> center, double T1, double T2, double r1, double r2, std::vector<int> corresp, std::vector<node_status> pts_statut, std::vector<Point<2>> decomp_elem, std::vector<double> sol_loc)
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



