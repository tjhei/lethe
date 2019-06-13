#include "calcbk.h"
#include "derinterpquad.h"
//#include <vector>
//#include <deal.II/base/quadrature_lib.h>
//#include <deal.II/base/function.h>
//#include <deal.II/base/utilities.h>
//#include <deal.II/base/index_set.h>
//using namespace dealii;

double calcinteg(int trg_or_quad, int i, int j, std::vector<Point<2> > coor1)
{
    // calculates and returns the value of the scalar product between the gradient of psi_i and the gradient of psi_j, but by getting into the reference
      //            element, which means we multiply by the jacobian and by Bk and t(Bk)

    // the actual formula is  :  jac * grad(psi_j(ref)) * t(Bk) * Bk * grad(psi_i(ref))

    // trg_or_quad == 0 means the element is a triangle, anything else means it's a quadrilateral


    double Bk_tBk[2][2] = {{0,0},{0,0}};
    double S;
    if (trg_or_quad == 0)
    {
        // we have to calculate the inverted and transposed matrix of the transformation from the reference triangle to the considered element
        calcbk(trg_or_quad, 0,0, coor1, Bk_tBk);
        //std::cout << "calc bk = " << Bk_tBk[0][0] << " " << Bk_tBk[0][1] << "\n "<< Bk_tBk[1][0] << " " << Bk_tBk[1][1]<< std::endl;
        double gradinterp[3][2];

        gradinterp[0][0] = -1.0;
        gradinterp[0][1] = -1.0;

        gradinterp[1][0] = 1.0;
        gradinterp[1][1] = 0.0;

        gradinterp[2][0] = 0.0;
        gradinterp[2][1] = 1.0;


//         double x1, x2, x3, y1, y2 , y3;
//        x1 = coor1[0][0];
//        x2 = coor1[1][0];
//        x3 = coor1[2][0];
//        y1 = coor1[0][1];
//        y2 = coor1[1][1];
//        y3 = coor1[2][1];
//        double jac = jacobian(trg_or_quad, 0, 0 , coor1);
//        double t;
//                double M11, M22, M12;
//                M11 = (y3-y1)*(y3-y1)+(x1-x3)*(x1-x3);
//                M12 = (y1-y2)*(y3-y1)+(x2-x1)*(x1-x3);
//                M22 = (y1-y2)*(y1-y2)+(x2-x1)*(x2-x1);
//                double B[2][2] = {{M11/(jac*jac), M12/(jac*jac)},{M12/(jac*jac), M22/(jac*jac)}};

//                for (int ii = 0; ii < 2; ++ii) {
//                    for (int jj = 0; jj < 2; ++jj) {
//                        std::cout << "egal ? " << B[ii][jj] <<", " << Bk_tBk[ii][jj] << std::endl;
//                    }

//                }
//        t = jac/2*(gradinterp[i][0]*(B[0][0]*gradinterp[j][0]+B[0][1]*gradinterp[j][1])
//                + gradinterp[i][1]*(B[1][0]*gradinterp[j][0]+B[1][1]*gradinterp[j][1]));




        S = jacobian(trg_or_quad, 0.0, 0.0, coor1)* (gradinterp[i][0]*(Bk_tBk[0][0]*gradinterp[j][0] + Bk_tBk[0][1]*gradinterp[j][1])+
                gradinterp[i][1]*(Bk_tBk[1][0]*gradinterp[j][0] + Bk_tBk[1][1]*gradinterp[j][1]))/2;

        //bool a = t==S;
        //std::cout << t <<", " << S << std::endl;
        std::cout << "S = " << S << std::endl;
    }

    else
    {
        S = 0;
        std::vector<double> gradi(2), gradj(2);
        double w[5], x[5];
        x[0] = -0.906179845938664;
        x[1] = -0.538469310105683;
        x[2] = 0.0;
        x[3] = 0.538469310105683;
        x[4] = 0.906179845938664;

        w[0] = 0.236926885056189;
        w[1] = 0.478628670499365;
        w[2] = 0.568888889888889;
        w[3] = 0.478628670499365;
        w[4] = 0.236926885056189;

        // since in a square this expression is a 8th degree polynomial, we have to apply qudrature formulas (Gauss), hence the w and x vectors

        for (int ii=0; ii<5; ii++) {
            for (int jj = 0;  jj< 5; jj++) {
                calcbk(trg_or_quad, x[ii], x[jj], coor1, Bk_tBk);
                derinterpquad(i, x[ii], x[jj], gradi);
                derinterpquad(j, x[ii], x[jj], gradj);

                //std::cout << "jac = " << jacobian(trg_or_quad, x[ii], x[jj], coor1) << " en x[i], y[i] = " << x[ii] << ", " << x[jj]  << std::endl;
               // std::cout << Bk_tBk[0][0] <<", " << Bk_tBk[0][1]<<", " << Bk_tBk[1][0]<<", " << Bk_tBk[1][1] << std::endl;

                S+= w[ii]*w[jj]*jacobian(trg_or_quad, x[ii], x[jj], coor1)*(  gradi[0]*(Bk_tBk[0][0]*gradj[0] + Bk_tBk[0][1]*gradj[1])+
                        gradi[1]*(Bk_tBk[1][0]*gradj[0] + Bk_tBk[1][1]*gradj[1])   );
            }
        }
        //std::cout << "\n" << std::endl;

    }
    return S;
}
