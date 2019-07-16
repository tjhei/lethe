#include "deal.II/lac/vector.h"
#include "deal.II/base/point.h"

using namespace dealii;

#ifndef JACOBIAN_H
#define JACOBIAN_H

double jacobian(int trg_or_quad, double x, double y, Vector<Point<2>> coor)
{
    /* calculates the jacobian of the polygon of coordinates given by coor
    at the point of coordinates (x, y), depending on if it's a triangle or a qudrilateral */

    if (trg_or_quad == 0) // 0 means it's a triangle
    {

        return (coor[2][1]-coor[0][1])*(coor[1][0]-coor[0][0])-
                (coor[1][1]-coor[0][1])*(coor[2][0]-coor[0][0]);

    }

    else // if (trg_or_quad == 1), 1 means it's a quadrilateral
    {
        /* Same problem as in nouvtriangles.h, we have got to change the order of the points so that
        we can calculate correctly the jacobian of the transformation from the square [-1, 1]x[-1, 1]
        To understand better why we have to make this you can get more explanations (in french tho) by
        by searching "Les éléments finis : de la théorie à la pratique" on any web browser, and
        you'll see the details of the calculations*/

        double x1, y1, x2, y2, x3, y3, x4, y4, B11, B12, B21, B22, C11, C21;
        x1 = coor[3][0];
        x2 = coor[2][0];
        x3 = coor[0][0];
        y1 = coor[3][1];
        y2 = coor[2][1];
        y3 = coor[0][1];
        x4 = coor[1][0];
        y4 = coor[1][1];

        B11 = (x1-x2-x3+x4)/4;
        B12 = (x1+x2-x3-x4)/4;
        B21 = (y1-y2-y3+y4)/4;
        B22 = (y1+y2-y3-y4)/4;

        C11 = (x1-x2+x3-x4)/4;

        C21 = (y1-y2+y3-y4)/4;
        // std::cout << B11 << ", " << B12 << ", " << B21 << ", " << B22 << std::endl;
        return ((B11 + y * C11)*(B22 + x * C21) - (B21 + y * C21)*(B12 + x * C11)) ;
    }
}
#endif
