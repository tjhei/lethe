#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/function.h>
#include "deal.II/base/tensor.h"
#include "deal.II/base/point.h"

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
#include "ib_node_status.h"
#include "ibcombiner.h"
#include "condensate.h"
#include "jacobian.h"

using namespace dealii;

class Heat_integration_circles: public Function<2>
{
public:
    Heat_integration_circles(int nb_poly_, std::vector<int> corresp_, std::vector<node_status> pts_statut_, std::vector<Point<2>> decomp_elem_) :
        Function<2>(1),
        nb_poly(nb_poly_),
        corresp(corresp_),
        pts_statut(pts_statut_),
        decomp_elem(decomp_elem_)

    {}

    Heat_integration_circles() :
        Function<2>(1)
    {}

    void T_integrate_IB(double Tdirichlet, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs);
    double T_L2_norm_IB(Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> T);

    void T_decomp_trg(double Tdirichlet, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs);
    double T_norm_L2_trg(Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> T);

    void T_decomp_quad(double Tdirichlet, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs);
    double T_norm_L2_quad(Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> sol_loc);

    void set_nb_poly(int nb_poly_){nb_poly = nb_poly_;}
    void set_corresp(std::vector<int> corresp_){corresp = corresp_;}
    void set_pts_status(std::vector<node_status> pts_statut_){pts_statut = pts_statut_;}
    void set_decomp(std::vector<Point<2>> decomp_elem_){decomp_elem = decomp_elem_;}

private:
    int nb_poly;
    std::vector<int> corresp;
    std::vector<node_status> pts_statut;
    std::vector<Point<2>> decomp_elem;

};

double T_analytical(Point<2> pt, Point<2> center, double T1, double T2, double r1, double r2)
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


// Functions for triangles //

void Heat_integration_circles::T_decomp_trg(double Tdirichlet, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs)
{
    // For a given element and the values of the distance function at its vertices, gives back the elementary matrix in the finite elements method
    // for the heat equation

    // determines the number of sub-elements

    // pts_status allows to know if a point is "solid" or "fluid" (with an enum)

    // corresp allows to know the equivalence between the numerotation among one subelement and the numerotation of the element

    // num_elem[i] gives the coordinates of the point associated to the number i in the numerotation of the element

    // decomp_elem is the decomposition in sub-elements (gives the coordinates in the right order for deal.ii)

    // nb_poly gives the number of sub-elements created

    // for more information check the documentation of nouv_triangles.h

        FullMatrix<double> M(6,6);
        M=0;

        Point<2> pt1, pt2, pt3;
        std::vector<int>           corresp_loc(3);
        int No_pts_solid[4] = {-1,-1,-1,-1};
        int a=0;
        for (unsigned int i=0;i<4;i++) {
            if (pts_statut[i]==solid)
            {No_pts_solid[a]=i; a+=1;}
        }

        for (int n = 0; n < nb_poly; ++n)
        {
            pt1= decomp_elem[3*n];
            pt2= decomp_elem[3*n+1];
            pt3= decomp_elem[3*n+2];

            corresp_loc[0] = corresp[3*n];
            corresp_loc[1] = corresp[3*n+1];
            corresp_loc[2] = corresp[3*n+2];

            double x1, x2, x3, y1, y2, y3;
            double jac;

            x1 = pt1(0);
            x2 = pt2(0);
            x3 = pt3(0);
            y1 = pt1(1);
            y2 = pt2(1);
            y3 = pt3(1);
            jac = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);

            double a11,a12, a13, a22,a33,a32;
            a11 = 1.0/(2.0*jac)*((y3-y2)*(y3-y1)+(x3-x2)*(x3-x1)+(y2-y1)*(y2-y3)+(x2-x3)*(x2-x1));
            a12 = 1.0/(2.0*jac)*((y3-y1)*(y2-y3)+(x3-x1)*(x2-x3));
            a13 = 1.0/(2.0*jac)*((y2-y1)*(y3-y2)+(x2-x1)*(x3-x2));
            a22 = 1.0/(2.0*jac)*((y3-y1)*(y3-y1)+(x3-x1)*(x3-x1));
            a33 = 1.0/(2.0*jac)*((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1));
            a32 = 1.0/(2.0*jac)*((y1-y2)*(y3-y1)+(x1-x3)*(x2-x1));

            double Melem[3][3] = {{a11, a12, a13},{a12, a22, a32}, {a13, a32, a33}};

            for (unsigned int k = 0; k < 3; ++k) {
                for (unsigned int l = 0; l < 3; ++l) {
                    M[corresp_loc[k]][corresp_loc[l]] += Melem[k][l];

                }
            }

        }

        unsigned int dofs_per_cell =4;
        Vector<double> rhs6(6);

        for (unsigned int i = 0; i < 4; ++i) {
            rhs6[i]=cell_rhs[i];
        }
        rhs6[4]=Tdirichlet;
        rhs6[5]=Tdirichlet;

        for (unsigned int i = 0; i < 4; ++i) {
            if (pts_statut[i])
            {
                for (unsigned int ii = 0; ii < 6; ++ii) {
                    M[i][ii]=0;
                    M[ii][i]=0;
                }
                M[i][i]=1;
                rhs6[i]=Tdirichlet;
            }
        }
        for (unsigned int i = 0; i <6; ++i) {
            M[4][i] = 0;
            M[5][i] = 0;
        }
        M[4][4]=1;
        M[5][5]=1;

        condensate(6, dofs_per_cell, M, cell_mat, rhs6, cell_rhs);
}


double interpolationtrg(int i, double x, double y)
{
    if (i==0){return 1-x-y;}
    else if (i==1){return x;}
    else {
        return y;
    }
}


double Heat_integration_circles::T_norm_L2_trg(Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> T)
{
    // evaluates the L2 norm of (T_analytical - T_calculated) where T_calculated is the solution found with finite elements and interpolated on the triangles

    double err = 0;

    double xi[4] = {1./3.,0.2,0.2,0.6};
    double eta[4] = {1./3.,0.2,0.6,0.2};
    double w[4] = {-0.28125, 0.2604166, 0.2604166, 0.2604166};
    double Tinterp;

    double Tloc[3];
    std::vector<Point<2>> trg(3);
    double jk;
    Point<2> pt;

    for (int n = 0; n < nb_poly; ++n)
    {
        Tloc[0] = T[corresp[3*n]];
        Tloc[1] = T[corresp[3*n+1]];
        Tloc[2] = T[corresp[3*n+2]];

        trg[0] = decomp_elem[3*n];
        trg[1] = decomp_elem[3*n+1];
        trg[2] = decomp_elem[3*n+2];

        jk = jacobian(0, 0, 0, trg);
        Point<2> pt_loc (0,0);

        for (int i = 0; i < 4; ++i) {

                pt(0)=xi[i];
                pt(1)=eta[i];

                pt_loc(0) = (1-pt(0)-pt(1))*trg[0](0)+pt(0)*trg[1](0)+pt(1)*trg[2](0);
                pt_loc(1) = (1-pt(0)-pt(1))*trg[0](1)+pt(0)*trg[1](1)+pt(1)*trg[2](1);

                Tinterp = interpolationtrg(0, xi[i],eta[i])*Tloc[0] + interpolationtrg(1, xi[i],eta[i])*Tloc[1] + interpolationtrg(2, xi[i],eta[i])*Tloc[2];
                err+=std::pow(T_analytical(pt_loc, center, T1, T2, r1, r2)-Tinterp,2)*jk*w[i];
        }
    }
    return err;
}

//          //

// Functions for quadrilaterals //


double T_calc_interp_quad(std::vector<double> T, Point<2> pt_calc)
{
    // we use this to make the linear interpolation of the solution in order to dertermine the L2 error
    // T is the value of the solution on each vertex
    // pt_calc is the point at which you wish to calculate the value of the interpolated solution
    double x = pt_calc(0);
    double y = pt_calc(1);

    return T[0]*(1.0-x)*(1.-y)+T[1]*x*(1.-y)+T[2]*(1.-x)*y+T[3]*x*y;
}


void Heat_integration_circles::T_decomp_quad(double Tdirichlet, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs)
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
    dof_handler.distribute_dofs(fe);
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
    unsigned int a =0;
    for (unsigned int i = 0; i < 4; ++i) {
        if (pts_statut[i]==solid)
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
          Vector<double> rhs4(4);
          Vector<double> rhs6(6);

          rhs4=0.;
          m4=0;
          rhs6=0.;

          for (int j = 0; j < 6; ++j) {
               for (int i = 4; i < 6; ++i){
                  if (i==j) M[i][i]=1;
                  else {
                      M[i][j]=0;
                  }
              }
          }

          FullMatrix<double> m6(6,6);

          for (unsigned int i = 0; i < 6; ++i) {
              for (unsigned int j = 0; j < 6; ++j) {
                  m6(i,j)=M[i][j];
              }
          }

          for (unsigned int i = 0; i < 4; ++i) {
              rhs6[i]=cell_rhs[i];
          }
          rhs6[4]=Tdirichlet;
          rhs6[5]=Tdirichlet;

          condensate(6,4,m6, cell_mat, rhs6, cell_rhs);
      }
}


double Heat_integration_circles::T_norm_L2_quad(Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> sol_loc)
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
    dof_handler.distribute_dofs(fe);
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
    unsigned int a=0;
    for (unsigned int i=0;i<4;i++) {
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
          err+=std::pow(T_analytical(fe_values.quadrature_point (q), center, T1, T2, r1, r2)-T_calc_interp_quad(T_quad, fe_values.quadrature_point (q)), 2)
                  *fe_values.JxW(q);
      }
    }
    return err;
}

//              //


// Function that integrates the heat equation for a boundary condition as T = Tdirichlet, depending on the kind of decomposition we made

void Heat_integration_circles::T_integrate_IB(double Tdirichlet, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs)
{
    if (nb_poly>0)
        T_decomp_trg(Tdirichlet, cell_mat, cell_rhs);
    else {
        T_decomp_quad(Tdirichlet, cell_mat, cell_rhs);
    }
}

// Function that evaluates the L2 norm of the error on the calculated solution to the Heat equation, given the values on the vertices of the decomposition
// and depending on the decomposition made

double Heat_integration_circles::T_L2_norm_IB(Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> T)
{
    if (nb_poly>0)
        return T_norm_L2_trg(center, T1, T2, r1, r2, T);
    else {
        return T_norm_L2_quad(center, T1, T2, r1, r2, T);
    }
}
