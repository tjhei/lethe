#include "interpolationquad.h"
void quadboundary(int i, std::vector<Point<2> > &decomp_elem, std::vector<Point<2> > coor_elem, std::vector<double> val_f)
{
    double x1, x2, y1, y2;
    x1=0;
    x2=0;
    y1=0;
    y2=0;

    // We calculate the coordinates of the intersections between the element and the boundary by interpolating the distance function with the Lagrange functions
    // We first determine the coordinates of this point in the reference square [-1, 1]x[-1, 1]
    if (i==1)
    {
        x1 = -1.0;
        y1 = (val_f[1]+val_f[2])/(val_f[2]-val_f[1]);
        x2 = 1.0;
        y2 = (val_f[0]+val_f[3])/(val_f[3]-val_f[0]);
    }

    else if (i==3)
    {
        x1 = 1.0;
        y1 = (val_f[0]+val_f[3])/(val_f[3]-val_f[0]);
        x2 = -1.0;
        y2 = (val_f[1]+val_f[2])/(val_f[2]-val_f[1]);
    }

    else if (i==2)
    {
        x1 = (val_f[3]+val_f[2])/(val_f[2]-val_f[3]);
        y1 = -1.0;
        x2 = (val_f[0]+val_f[1])/(val_f[1]-val_f[0]);
        y2 = 1.0;
    }

    else if (i==4)
    {
        x1 = (val_f[0]+val_f[1])/(val_f[1]-val_f[0]);
        y1 = 1.0;
        x2 = (val_f[3]+val_f[2])/(val_f[2]-val_f[3]);
        y2 = -1.0;
    }

    /* std::cout << "point 1 " << x1 << ", " << y1 << " point 2 " << x2 << ", " << y2 << std::endl; */
    // we then apply the transformation to get the coordinates in the element we're considering

    double xx1 = coor_elem[0][0] ;
    double xx2 = coor_elem[1][0] ;
    double xx3 = coor_elem[2][0] ;
    double xx4 = coor_elem[3][0] ;

    double yy1 = coor_elem[0][1] ;
    double yy2 = coor_elem[1][1] ;
    double yy3 = coor_elem[2][1] ;
    double yy4 = coor_elem[3][1] ;

    double x11 = (1+x1)*(1+y1)*xx1/4 + (1-x1)*(1+y1)*xx2/4+(1-x1)*(1-y1)*xx3/4+(1+x1)*(1-y1)*xx4/4 ;
    double y11 = ((1+x1)*(1+y1) * yy1 + (1-x1)*(1+y1) * yy2 + (1-x1)*(1-y1) * yy3 + (1+x1)*(1-y1) * yy4)/4 ;

    double x22 = (1+x2)*(1+y2)*xx1/4 + (1-x2)*(1+y2)*xx2/4+(1-x2)*(1-y2)*xx3/4+(1+x2)*(1-y2)*xx4/4 ;
    double y22 = ((1+x2)*(1+y2) * yy1 + (1-x2)*(1+y2) * yy2 + (1-x2)*(1-y2) * yy3 + (1+x2)*(1-y2) * yy4)/4 ;


    Point<2> pt1, pt2;
    pt1[0] = x11;
    pt1[1] = y11;

    pt2[0] = x22;
    pt2[1] = y22;

    /*decomp_elem[0]=coor_elem[i];
    decomp_elem[1]=coor_elem[(i+1) % 4];
    decomp_elem[2]=pt1;
    decomp_elem[3]=pt2;
    Doing this the returned quadrilateral wouldn't have its summits in the right order*/

    /// i varie entre 1 et 4 !!!!!! d'où le fait qu'on prenne i-1 % 4 et i % 4
    decomp_elem[3]=coor_elem[(i-1) % 4];
    decomp_elem[2]=coor_elem[(i) % 4];
    decomp_elem[0]=pt1;
    decomp_elem[1]=pt2;
    /*std::cout <<"coor elem " << coor_elem[0] << ", " << coor_elem[1] << ", " <<coor_elem[2] << ", " <<coor_elem[3] << std::endl;
    std::cout <<"fct dist " << val_f[0] << ", " << val_f[1] << ", " <<val_f[2] << ", " <<val_f[3] << std::endl;
    std::cout <<"Nouvel élément " << decomp_elem[0] << ", " << decomp_elem[1] << ", " << decomp_elem[2] << ", " << decomp_elem[3] << "\n \n" << std::endl;
    */
}
