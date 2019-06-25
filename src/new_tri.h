#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/full_matrix.h>

#include "T_analytical.h"
#include "interpolationtrg.h"
#include "jacobian.h"
#include "enum_solid_fluid.h"
using namespace dealii;

void new_tri(double Tdirichlet, int nbtrg, std::vector<int> corresp, std::vector<Point<2>> decomp_elem, std::vector<In_fluid_or_in_solid> pts_statut, FullMatrix<double> &cell_mat, std::vector<double> &cell_rhs)
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

    double M[6][6] = {{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0}};

        Point<2> pt1, pt2, pt3;
        std::vector<int>           corresp_loc(3);
        int No_pts_solid[4] = {-1,-1,-1,-1};
        int a=0;
        for (int i=0;i<4;i++) {
            if (pts_statut[i]==solid)
            {No_pts_solid[a]=i; a+=1;}
        }

        for (int n = 0; n < nbtrg; ++n)
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

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    M[corresp_loc[k]][corresp_loc[l]] += Melem[k][l];

                }
            }

        }

        for (unsigned int j = 0; j < 4; ++j) {
            for (unsigned int var = 0; var < 4; ++var) {
                cell_mat[j][var] = M[j][var];
            }
            cell_rhs[j] += -Tdirichlet * (M[j][4] + M[j][5]);
        }

        int i =0;
        while (No_pts_solid[i]>=0) {

            // The coefficient associated to the vertices in the solid are set to 0 except for the one on the diagonal

                cell_mat[No_pts_solid[i]][No_pts_solid[i]]=1;
                cell_rhs[No_pts_solid[i]] = Tdirichlet;
                i++;
            }

}

double new_tri_L2(int nbtrg, std::vector<Point<2>> decomp_elem, std::vector<int> corresp, std::vector<In_fluid_or_in_solid> No_pts_solid, Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> T)
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

    for (int n = 0; n < nbtrg; ++n)
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
