#include "jacobian.h"
double area(int nb, std::vector<Point<2> > decomp_elem, std::vector<double> dist, std::vector<Point<2> > coor)
{
    double area_t= 0.0;
    /*for (int i = 0; i < 9; ++i) {
        std::cout <<"décomp " << i <<" = " << decomp_elem[i] << std::endl;
    } */
    // calculates the area of the fluid part of a given element
    /* for (int i = 0; i < 4; ++i) {
            std::cout <<"coor " << i <<" = " << coor[i] << std::endl;
        } */
    if (nb > 0)
    {
        std::vector<Point<2> > trg(3);
        for (int i = 0 ; i < nb ; i++)
        {
            trg[0]= decomp_elem[3*i];
            trg[1]= decomp_elem[3*i+1];
            trg[2]= decomp_elem[3*i+2];
            std::cout << "aire du " << i+1 <<"e triangle : " << jacobian(0, 0.0, 0.0, trg)/2 << std::endl;
            area_t += jacobian(0, 0.0, 0.0, trg)/2;
        }
    }

    else if (nb == 0)
    {
        for (int i = 0 ; i < 4 ; i++)
        {
            if (dist[i]>0){area_t += 4.0*jacobian(1, 0.0, 0.0, coor); break;}
        }
    }

    else //if (nb == -1)
    {
        std::vector<Point<2> >  quad(4);
        quad[0] = decomp_elem[0];
        quad[1] = decomp_elem[1];
        quad[2] = decomp_elem[2];
        quad[3] = decomp_elem[3];
        std::cout << "aire du quad créé : " << 4.0*jacobian(1, 0.0, 0.0, quad) << std::endl;
        area_t += 4.0*jacobian(1, 0.0, 0.0, quad);
    }

    return area_t;
}
