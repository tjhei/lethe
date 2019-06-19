#include <deal.II/base/point.h>

#ifndef T_ANALYTICAL_H
#define T_ANALYTICAL_H

using namespace dealii;

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

#endif //T_ANALYTICAL_H
