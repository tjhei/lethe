#include "quadboundary.h"
#include "trgboundary.h"


void nouvtriangles(std::vector<Point<2> > &decomp_elem, int* nb_poly, std::vector<Point<2> > coor_elem1, std::vector<double> val_f1)
{
    /* decomp_elem is a vector in which we shall store the coordinates of the sub_elements created with this function
     nb_poly will allow us to indicate the number of polygons returned in decomp_elem and to specify if we return triangles or a quadrilateral
     coor_elem are the coordinates of each summit of the element considered
     val_f gives the value of the distance function for each summit of the element

     This code is made to work with the following numerotation for the summits of the element

          1----0
          |    |
          2----3

     However the numerotation used in deal.II is as follows

          2----3
          |    |
          0----1

     Thus we shall make a local numerotation for the summits, and create a local vector for the distance function matching with it. */

    std::vector<Point<2> > coor_elem(4);
    std::vector<double> val_f(4);

    coor_elem[0] = coor_elem1[3];
    coor_elem[1] = coor_elem1[2];
    coor_elem[2] = coor_elem1[0];
    coor_elem[3] = coor_elem1[1];

    val_f[0] = val_f1[3];
    val_f[1] = val_f1[2];
    val_f[2] = val_f1[0];
    val_f[3] = val_f1[1];

    // We will start by finding if there are any changes in the sign of the distance function among the the summits

    std::vector<int> a(4);
    int sum_a =0;
    int in = 0; // it will allow us to spot the hypothetical cases where the boundary is one of the sides of the element considered
    for (int i = 0; i < 4 ; i++)
    {
        a[i]=1;
        if (val_f[i]<0) {a[i]=-1;}

        else if (val_f[i]==0)
        {
            a[i]=0;
            if (val_f[(i+1) % 4]  == 0 || val_f[(i+3) % 4] ==0) {in = 1;}
        }
        sum_a += a[i];
    }

    // sum_a indicates if there are more summits in the fluid than in the solid (it is then strictly positive)

    if (sum_a==0) // there are as many summits in the fluid as in the solid
    {

         // we have to look for the cases where the boundary fluid-solid passes through summits
        for (int i = 0; i < 4 ; i++)
        {

            if (a[i]>0)
            {
                if (a[(i+1) % 4]==0) // we are in the case where the boundary goes through 2 summits
                {
                    decomp_elem[0]=coor_elem[i];
                    decomp_elem[1]=coor_elem[(i+1) % 4];
                    decomp_elem[2]=coor_elem[(i+3) % 4];
                    *nb_poly = 1;
                    break;
                }

                else if ((i==0) && (a[3]>0)) // this case exists just to have always the same order in the summits of the sub-elements
                {
                    quadboundary(4, decomp_elem, coor_elem, val_f);
                    *nb_poly = -1;
                    break;
                }
                else if (i==0 && a[3]<0) {quadboundary(1, decomp_elem, coor_elem, val_f);
                *nb_poly = -1;
                break;}
                else if (i==1) {quadboundary(2, decomp_elem, coor_elem, val_f);
                *nb_poly = -1;
                break;}
                else if (i==2) {quadboundary(3, decomp_elem, coor_elem, val_f);
                *nb_poly = -1;
                break;}
                else if (i==3) {quadboundary(4, decomp_elem, coor_elem, val_f);
                *nb_poly = -1;
                break;}
            }
        }
    }


    else if (sum_a<0 && sum_a>-3) // case where there is one or less summit in the fluid part
    {
        if (in==0) // if in = 1, it means that the boundary is merging with one side
        {
            int b=0;
            for (int i = 0; i < 4; ++i)
            {
                if (a[i]==1){b=i; break;}
            }

            std::vector<Point<2> >      boundary_pts(2);
            trgboundary(b, boundary_pts, coor_elem, val_f);

            //std::cout << "Boundary Points (" << boundary_pts[0] << "), (" << boundary_pts[1] << ")" << std::endl;
            decomp_elem[0]=coor_elem[b];
            decomp_elem[1]=boundary_pts[0];
            decomp_elem[2]=boundary_pts[1];
            *nb_poly = 1;
        }

        else if (in==1){*nb_poly = 0;}

    }


    else if (sum_a>=3 || sum_a<=-3){*nb_poly = 0;}


    else if (sum_a>0 && sum_a<3) // there is up to one summit in the solid zone
    {
        int b=0;
        for (int i = 0; i < 4; ++i)
        {
            if (a[i]==-1){b=i; break;}
        }

        if (sum_a==2)
        {
            if (in==0)
            {
                std::vector<Point<2> >      boundary_pts(2);
                trgboundary(b, boundary_pts, coor_elem, val_f);
                //std::cout <<"Boundary Points (" << boundary_pts[0] << "), (" << boundary_pts[1] << ")" << std::endl;
                decomp_elem [0] = coor_elem[(b+1)%4];
                decomp_elem [1] = coor_elem[(b+2)%4];
                decomp_elem [2] = boundary_pts[0];

                decomp_elem [3] = coor_elem[(b+2)%4];
                decomp_elem [4] = boundary_pts[1];
                decomp_elem [5] = boundary_pts[0];

                decomp_elem [6] = coor_elem[(b+2)%4];
                decomp_elem [7] = coor_elem[(b+3)%4];
                decomp_elem [8] = boundary_pts[1];

                *nb_poly = 3;
            }
            else if (in==1){*nb_poly = 0;}

        }

        else if (sum_a==1) // corresponds to the case where there is one summit in the solid and one summit crossed by the boundary
        {
            std::vector<Point<2> >      boundary_pts(2);
            trgboundary(b, boundary_pts, coor_elem, val_f);

            if (val_f[(b+1) % 4]==0)
            {
                decomp_elem[0] = coor_elem[(b+2) % 4];
                decomp_elem[1] = coor_elem[(b+3) % 4];
                decomp_elem[2] = boundary_pts[0];
                decomp_elem[3] = boundary_pts[1];
            }

            else if (val_f[(b+3) % 4]==0)
            {
                decomp_elem[0] = coor_elem[(b+1) % 4];
                decomp_elem[1] = coor_elem[(b+2) % 4];
                decomp_elem[2] = boundary_pts[0];
                decomp_elem[3] = boundary_pts[1];
            }

            *nb_poly = -1;
        }
    }
    /*if (*nb_poly > 0) {for (int i = 0; i < *nb_poly; ++i) {
        std::cout <<"Sub_elem " << i+1 <<" = " << decomp_elem[3*i] << "\n" << decomp_elem[3*i+1] << "\n" << decomp_elem[3*i+2] << "\n" <<std::endl;
    } } */
}
