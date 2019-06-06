void trgboundary(int b, std::vector<Point<2> > &boundary_pts, std::vector<Point<2> > coor_elem, std::vector<double> val_f)
{
    double x1, x2, y1, y2;
    x1=0;
    x2=0;
    y1=0;
    y2=0;
    // We calculate the coordinates of the intersections between the element and the boundary by interpolating the distance function with the Lagrange functions
    // We first determine the coordinates of this point in the reference square [-1, 1]x[-1, 1]

    /*std::cout << "val_f = " << val_f[0] << ", " << val_f[1] << ", " << val_f[2] << ", " << val_f[3] << std::endl;
    std::cout << "b = " << b << std::endl; */

    if (b==0)
    {
        x1 = (val_f[0]+val_f[1])/(val_f[1]-val_f[0]);
        y1 = 1;
        x2 = 1;
        y2 = (val_f[0]+val_f[3])/(val_f[3]-val_f[0]);
    }

    else if (b==2)
    {
        x1 = (val_f[2]+val_f[3])/(val_f[2]-val_f[3]);
        y1 = -1;
        x2 = -1;
        y2 = (val_f[1]+val_f[2])/(val_f[2]-val_f[1]);
    }

    else if (b==1)
    {
        x1 = -1;
        y1 = (val_f[2]+val_f[1])/(val_f[1]-val_f[2]);
        x2 = (val_f[0]+val_f[1])/(val_f[1]-val_f[0]);
        y2 = 1;
    }

    else // b == 3
    {
        x1 = 1;
        y1 = (val_f[0]+val_f[3])/(val_f[3]-val_f[0]);
        x2 = (val_f[3]+val_f[2])/(val_f[2]-val_f[3]);
        y2 = -1;
    }

    // we then apply the transformation to get the coordinates in the element we're considering
    Point<2> pt1(1), pt2(1);

    for (int j = 0; j < 4; ++j)
    {
        double L1 = interpolationquad(j, x1, y1);
        double L2 = interpolationquad(j, x2, y2);


        pt1[0]+= L1 * coor_elem[j][0];
        pt1[1]+= L1 * coor_elem[j][1];

        pt2[0]+= L2 * coor_elem[j][0];
        pt2[1]+= L2 * coor_elem[j][1];
    }
    boundary_pts[0]=pt1;
    boundary_pts[1]=pt2;
    /* std::cout << "triongle(s)" << std::endl; */
    // the points of intersection are returned in the vector boundary_pts, and are in the right order if there is only one summit in the fluid
    // if there are 3 summits in the fluid, you have to change the order so that the triangles that will be created can be described in the trigonometrical order
}

