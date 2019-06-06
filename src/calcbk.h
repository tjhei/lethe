#include "jacobian.h"
void calcbk(int trg_or_quad, double x, double y, std::vector<Point<2> > coor, double Bk_transpBk[2][2])
{
    // les coordonnées arrivent sous la forme normale de deal.II
    // on les réorganise ensuite

    double x1, x2, x3, y1, y2, y3;
    x1 = coor[3][0];
    x2 = coor[2][0];
    x3 = coor[0][0];
    y1 = coor[3][1];
    y2 = coor[2][1];
    y3 = coor[0][1];
    double Bk[2][2];
    double transpBk[2][2];
    double jac = jacobian(trg_or_quad, x, y, coor);

    if (trg_or_quad==0)
    {
        Bk[0][0] = (y3-y1)/jac;
        Bk[0][1] = (x3-x1)/jac;
        Bk[1][0] = (y2-y1)/jac;
        Bk[1][1] = (x2-y1)/jac;

        transpBk[0][0] = (y3-y1)/jac;
        transpBk[1][0] = (x3-x1)/jac;
        transpBk[0][1] = (y2-y1)/jac;
        transpBk[1][1] = (x2-y1)/jac;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; j++) {
                Bk_transpBk[i][j] += Bk[i][0]*transpBk[0][j] + Bk[i][1]*transpBk[1][j];
            }
        }
    }

    else
    {
        double x4, y4, B11, B12, B21, B22, C11, C21;

        x4 = coor[1][0];
        y4 = coor[1][1];

        B11 = (x1-x2-x3+x4)/4;
        B12 = (x1+x2-x3-x4)/4;
        B21 = (y1-y2-y3+y4)/4;
        B22 = (y1+y2-y3-y4)/4;

        C11 = (x1-x2+x3-x4)/4;

        C21 = (y1-y2+y3-y4)/4;

        Bk[0][0] = (B22+x*C21)/jac;
        Bk[0][1] = -(B21+y*C21)/jac;
        Bk[1][0] = -(B12+x*C11)/jac;
        Bk[1][1] = (B11+y*C11)/jac;

        transpBk[0][0] = (B22+x*C21)/jac;
        transpBk[1][0] = -(B21+y*C21)/jac;
        transpBk[0][1] = -(B12+x*C11)/jac;
        transpBk[1][1] = (B11+y*C11)/jac;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; j++) {
                Bk_transpBk[i][j] += Bk[i][0]*transpBk[0][j] + Bk[i][1]*transpBk[1][j];
            }
        }
    }
}
