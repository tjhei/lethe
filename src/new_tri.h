#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/full_matrix.h>

using namespace dealii;

void new_tri(double Tdirichlet, int nbtrg, std::vector<int> corresp, std::vector<Point<2>> decomp_elem, std::vector<int> No_pts_solid, FullMatrix<double> &cell_mat, std::vector<double> &cell_rhs)
{
    // For a given element and the values of the distance function at its vertices, gives back the elementary matrix in the finite elements method
    // for the heat equation

    // determines the number of sub-elements

    // No_pts_solid allows to know which points are in the solid : it gives the numerotation associated to the vertices in the solid, has a length of four elements
    // you just have to see trough its elements and when No_pts_solid[i] < 0, it means there are no more vertices in the solid

    // corresp allows to know the equivalence between the numerotation among one subelement and the numerotation of the element

    // num_elem[i] gives the coordinates of the point associated to the number i in the numerotation of the element

    // decomp_elem is the decomposition in sub-elements (gives the coordinates in the right order for deal.ii)

    // nb_poly gives the number of sub-elements created

    // for more information check the documentation of nouv_triangles.h

    double M[6][6] = {{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0}};

        Point<2> pt1, pt2, pt3;
        std::vector<int>           corresp_loc(3);


        for (int n = 0; n < nbtrg; ++n)
        {
            pt1= decomp_elem[3*n];
            pt2= decomp_elem[3*n+1];
            pt3= decomp_elem[3*n+2];

            corresp_loc[0] = corresp[3*n];
            corresp_loc[1] = corresp[3*n+1];
            corresp_loc[2] = corresp[3*n+2];

//            std::cout << "corresp " << corresp_loc[0] << ", " << corresp_loc[1] << ", " << corresp_loc[2] << "\n" << std::endl;

            double x1, x2, x3, y1, y2, y3;
            double jac;

            x1 = pt1(0);
            x2 = pt2(0);
            x3 = pt3(0);
            y1 = pt1(1);
            y2 = pt2(1);
            y3 = pt3(1);
            //std::cout << "pts : " << pt1 << " , " << pt2 << " , " << pt3 << "\n" << std::endl;
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

                } /*std::cout << Melem[k][0] << ", "  << Melem[k][1] << ", " << Melem[k][2]   << std::endl;*/
            }
//            std::cout << "\n" << std::endl;
//            for (int i = 0; i < 6; ++i) {
//                std::cout << M[i][0] << ", " <<  M[i][1] << ", " << M[i][2] << ", " << M[i][3] << ", " << M[i][4] << ", " << M[i][5] << std::endl;
//            }
//            std::cout << "\n" << std::endl;
        }

        for (unsigned int j = 0; j < 4; ++j) {
            for (unsigned int var = 0; var < 4; ++var) {
                cell_mat[j][var] = M[j][var];
            }
            cell_rhs[j] += -Tdirichlet * (M[j][4] + M[j][5]);
            //std::cout << cell_rhs[j] << std::endl;
        }

        int i =0;
        while (No_pts_solid[i]>=0) {

            // The coefficient associated to the vertices in the solid are set to 0 except for the one on the diagonal

                cell_mat[No_pts_solid[i]][No_pts_solid[i]]=1;
                cell_rhs[No_pts_solid[i]] = Tdirichlet;
                i++;
            }
        //if (nb_poly>0) {for (int i = 0; i<6 ; i++ ) {std::cout << "ligne " << i << " = " << mat_elem2[i][0] << ", " << mat_elem2[i][1] << ", " << mat_elem2[i][2] << ", " << mat_elem2[i][3] << ", " << mat_elem2[i][4] << ", " << mat_elem2[i][5] << std::endl;}}
//        std::cout << "pts_solid : " << No_pts_solid[0] << ", " << No_pts_solid[1] << ", " << No_pts_solid[2] << ", " <<No_pts_solid[3] << std::endl;
//        std::cout << "\n" << std::endl;


}


