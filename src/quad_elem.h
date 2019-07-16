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
#include <thread>
#include <chrono>
#include "ib_node_status.h"

using namespace dealii;

double T_calc_interp(std::vector<double> T, Point<2> pt_calc)
{
    // we use this to make the linear interpolation of the solution in order to dertermine the L2 error
    // T is the value of the solution on each vertex
    // pt_calc is the point at which you wish to calculate the value of the interpolated solution
    double x = pt_calc(0);
    double y = pt_calc(1);

    return T[0]*(1.0-x)*(1.-y)+T[1]*x*(1.-y)+T[2]*(1.-x)*y+T[3]*x*y;
}

double T_analytical_q(Point<2> pt, Point<2> center, double T1, double T2, double r1, double r2)
{
    double r_eff;
    Point<2> pt_eff;
    pt_eff(0) = pt(0)-center(0);
    pt_eff(1) = pt(1)-center(1);

    r_eff = sqrt(pt_eff.square());

    double A,B;
    A = (T2-T1)/log(r1/r2);
    B = A*log(r1) + T1;
    return -A*log(r_eff)+B;
}

void condensate(unsigned int nb_of_line, unsigned int new_nb, FullMatrix<double> &M, FullMatrix<double> &new_mat, std::vector<double> &rhs, std::vector<double> &new_rhs)
{
    int a;

    for (unsigned int i = nb_of_line-2; i >new_nb-1; --i) { // We begin at the second to last line, i is the number associated to line we're modifying

        for (unsigned int k = 0; k < nb_of_line-1-i ; ++k) { // How many times we modify the line
            a = nb_of_line-1-k;

            for (unsigned int j = 0; j < i+1; ++j) { // number associated to the column
                M(i,j) -= M(i,a)*M(a,j)/M(a,a);
            }

            rhs[i] -= rhs[a]*M(i,a)/M(a,a);
        }
    }

    // We modified the bottom of the matrix, now we have to reinject the coefficients
    // into the part of the matrix we want to return

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int k = nb_of_line-1; k > new_nb-1 ; --k) {
            for (unsigned int j = 0; j < k; ++j) {
                M(i,j)-=M(i,k)*M(k,j)/M(k,k);
            }
            rhs[i]-=rhs[k]*M(i,k)/M(k,k);
        }
    }

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int j = 0; j < new_nb; ++j) {
            new_mat[i][j]=M[i][j];
        }
        new_rhs[i]=rhs[i];
    }
}


void quad_elem_mix(double Tdirichlet, Vector<node_status> No_pts_solid, std::vector<int> corresp, Vector<Point<2>> decomp_elem, FullMatrix<double> &cell_mat, std::vector<double> &cell_rhs)
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

          FullMatrix<double> m4(4,4);
          std::vector<double> rhs4(4);
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

          FullMatrix<double> m6(6,6);

          for (int i = 0; i < 6; ++i) {
              for (int j = 0; j < 6; ++j) {
                  m6(i,j)=M[i][j];
              }
          }

          for (int i = 0; i < 4; ++i) {
              rhs6[i]=cell_rhs[i];
          }
          rhs6[4]=Tdirichlet;
          rhs6[5]=Tdirichlet;

          condensate(6,4,m6, cell_mat, rhs6, cell_rhs);
      }
}

double quad_elem_L2(Point<2> center, double T1, double T2, double r1, double r2, std::vector<int> corresp, Vector<node_status> pts_statut, Vector<Point<2>> decomp_elem, std::vector<double> sol_loc)
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
          err+=std::pow(T_analytical_q(fe_values.quadrature_point (q), center, 1, 2, r1, r2)-T_calc_interp(T_quad, fe_values.quadrature_point (q)), 2)
                  *fe_values.JxW(q);
      }
    }
    return err;
}



