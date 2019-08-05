#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/full_matrix.h>

#include "jacobian.h"
#include "ib_node_status.h"
using namespace dealii;

void condensate_trg(unsigned int nb_of_line, unsigned int new_nb, FullMatrix<double> &M, FullMatrix<double> &new_mat, Vector<double> &rhs, Vector<double> &new_rhs)
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

void T_decomp_trg(double Tdirichlet, int nbtrg, std::vector<int> corresp, std::vector<Point<2>> decomp_elem, std::vector<node_status> pts_statut, FullMatrix<double> &cell_mat, Vector<double> &cell_rhs)
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

        unsigned int dofs_per_cell =4;
        Vector<double> rhs6(6);

        for (int i = 0; i < 4; ++i) {
            rhs6[i]=cell_rhs[i];
        }
        rhs6[4]=Tdirichlet;
        rhs6[5]=Tdirichlet;

        for (int i = 0; i < 4; ++i) {
            if (pts_statut[i])
            {
                for (int ii = 0; ii < 6; ++ii) {
                    M[i][ii]=0;
                    M[ii][i]=0;
                }
                M[i][i]=1;
                rhs6[i]=Tdirichlet;
            }
        }
        for (int i = 0; i <6; ++i) {
            M[4][i] = 0;
            M[5][i] = 0;
        }
        M[4][4]=1;
        M[5][5]=1;

        condensate_trg(6, dofs_per_cell, M, cell_mat, rhs6, cell_rhs);
}

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

double interpolationtrg(int i, double x, double y)
{
    if (i==0){return 1-x-y;}
    else if (i==1){return x;}
    else {
        return y;
    }
}

double T_norm_l2_trg(int nbtrg, std::vector<Point<2>> decomp_elem, std::vector<int> corresp, std::vector<node_status> No_pts_solid, Point<2> center, double T1, double T2, double r1, double r2, std::vector<double> T)
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


