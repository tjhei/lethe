#include <deal.II/base/point.h>
#include <vector>
#include "interpolationquad.h"
using namespace dealii;

#ifndef T_CALC_INTERP_H
#define T_CALC_INTERP_H


double T_calc_interp(std::vector<double> T, Point<2> pt_calc)
{
    // we use this to make the linear interpolation of the solution in order to dertermine the L2 error
    // T is the value of the solution on each vertex
    // pt_calc is the point at which you wish to calculate the value of the interpolated solution
    double x = pt_calc(0);
    double y = pt_calc(1);

    return T[0]*(1.0-x)*(1.-y)+T[1]*x*(1.-y)+T[2]*(1.-x)*y+T[3]*x*y;
}

#endif // T_CALC_INTERP_H
