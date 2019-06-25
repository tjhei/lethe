#include "quadboundary.h"
#include "trgboundary.h"
#include "deal.II/base/point.h"
#include <vector>

#include "enum_solid_fluid.h"
#ifndef NOUVTRIANGLES_H
#define NOUVTRIANGLES_H


using namespace dealii;


void nouvtriangles(std::vector<int> &corresp, std::vector<In_fluid_or_in_solid> &No_pts_solid, std::vector<Point<2> > &num_elem, std::vector<Point<2> > &decomp_elem, int* nb_poly, std::vector<Point<2> > coor_elem1, std::vector<double> val_f1)
{
    /* *** decomp_elem is a vector in which we shall store the coordinates of the sub_elements created with this function

     *** nb_poly will allow us to indicate the number of sub_elements created in decomp_elem and to specify if we return triangles or a quadrilateral
     nb_ poly is pretty simple :
        - positive if the sub_elements are triangles, then it indicates the number created
        - 0 if the element considered is totally in the fluid or in the solid
        - -1 if we create a qudrilateral

     *** corresp : returns the equivalence between the local numerotation in the sub-elements and the numerotation in the initial element, its length is 9 so that we don,t have to create a dynamic array
     so for ex. if we create only one triangle, we shall fill only the 3 first elements of corresp and then we will put -1 to all the others elements of corresp
     Example :

     2-5--3     Let this rectangle be the element studied. For this example, the vertices 0, 1 and 3 are in the solid, 4 and 5 are the points of intersection between the sides of the rectangle and the boundary
     |/   |
     4    |     The vertices of the sub-element created would be 2, 4 and 5. Thus, corresp would be {2, 4, 5, -1, -1,  ..., -1}
     |    |     Here nb_poly = 1. Decom_elem gives the coordinates of 2, 4 and 5 (in this order !).
     0----1

     *** No_pts_solid allows you to know if the i-th vertex is in the fluid or in the solid.

     *** num_elem[i] gives the coordinates of the point associated to the number i in the numerotation of the element. In this example, num_elem[0] would be the coordinates of the point 0, and thus would be equal to coor_elem1[0].

     *** You also have to give :
     - coor_elem1 : the coordinates of each vertex of the element considered
     - val_f1 : the value of the distance function for each vertex of the element

    IMPORTANT :
     This code is made to work with the following numerotation for the vertices of the element

          1----0
          |    |
          2----3

     However the numerotation used in deal.II is as follows

          2----3
          |    |
          0----1

     Thus we shall make a local numerotation for the vertices, and create a local vector for the distance function matching with it.
     FOR THAT WE JUST HAVE TO USE THE VECTOR "vec_change_coor"

*/

    std::vector<Point<2> > coor_elem(4);
    std::vector<double> val_f(4);

    //////int inv_vec_change_coor[4] {2, 3, 1, 0}; // to change coor from ref coor to deal.II coor
    const int vec_change_coor[4] {3, 2, 0, 1}; // to change coor from deal.II coor to ref coor


    for (int jj = 0; jj < 4; ++jj) { // changing the coordinates
        coor_elem[jj]  = coor_elem1[vec_change_coor[jj]];
        val_f[jj] = val_f1[vec_change_coor[jj]];
        //std::cout << coor_elem[jj] << std::endl;
    }

    double accuracy = 0.0001;

    // We will start by finding if there are any changes in the sign of the distance function among the the summits

    std::vector<int> a(4);
    int sum_a =0;
    int in = 0; // it will allow us to spot the hypothetical cases where the boundary is one of the sides of the element considered
    for (int i = 0; i < 4 ; i++)
    {
        a[i]=1;
        No_pts_solid[vec_change_coor[i]] = fluid;
        if (val_f[i]<0) {a[i]=-1; No_pts_solid[vec_change_coor[i]] = solid;}

        else if ((val_f[i]<accuracy) && (val_f[i]>-accuracy))
        {
            a[i]=0;
            if (((val_f[(i+1) % 4] <accuracy) && (val_f[i]>-accuracy)) || ((val_f[(i+3) % 4]<accuracy)&&(val_f[(i+3) % 4]>-accuracy))) {in = 1;}
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

                    corresp = {vec_change_coor[i], vec_change_coor[(i+1)%4], vec_change_coor[(i-1)%4], -1, -1, -1, -1, -1, -1};
                    num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], coor_elem[(i+1) % 4], coor_elem[(i+3) % 4]}; //// il va falloir traiter ce cas autrement ...
                     ////////
                    ///    ///
                    /// // ///
                    /// // ///
                    ///    ///
                     ////////
                    *nb_poly = 1;
                    break;
                }

                else if ((i==0) && (a[3]>0)) // this case exists just to have always the same order in the summits of the sub-elements
                {
                    quadboundary(4, decomp_elem, coor_elem, val_f);

                    corresp = {4, 5, vec_change_coor[3], vec_change_coor[0],-1,-1,-1,-1,-1};
                    num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], decomp_elem[0], decomp_elem[1]};

                    *nb_poly = -1;
                    break;
                }

                else
                {
                    quadboundary(i+1, decomp_elem, coor_elem, val_f);

                    corresp = {4, 5, vec_change_coor[(i+1) % 4], vec_change_coor[i],-1,-1,-1,-1,-1};
                    num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], decomp_elem[0], decomp_elem[1]};

                    *nb_poly = -1;
                    break;
                }
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
            //std::cout <<  vec_change_coor[b] << std::endl;
            num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], boundary_pts[0], boundary_pts[1]};
            corresp = {vec_change_coor[b], 4, 5, -1,-1,-1,-1,-1};

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

                decomp_elem [6] = coor_elem[(b+3)%4];
                decomp_elem [7] = boundary_pts[1];
                decomp_elem [8] = coor_elem[(b+2)%4];

                corresp = {vec_change_coor[(b+1)%4], vec_change_coor[(b+2)%4], 5, vec_change_coor[(b+2)%4], 4, 5, vec_change_coor[(b+3)%4],4 , vec_change_coor[(b+2)%4] };
                num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], boundary_pts[1], boundary_pts[0]}; // boundary points are sorte differently because of the way the function trgboundary works

                *nb_poly = 3;
            }
            else if (in==1){*nb_poly = 0;}

        }

        else if (sum_a==1) // corresponds to the case where there is one summit in the solid and one summit crossed by the boundary
        {
            std::vector<Point<2> >      boundary_pts(2);
            trgboundary(b, boundary_pts, coor_elem, val_f);

            if ((val_f[(b+1) % 4]<accuracy) && (val_f[(b+1) % 4]>-accuracy))
            {
                decomp_elem[0] = coor_elem[(b+2) % 4];
                decomp_elem[1] = coor_elem[(b+3) % 4];
                decomp_elem[2] = boundary_pts[0];
                decomp_elem[3] = boundary_pts[1];

                corresp = {vec_change_coor[(b+2) % 4], vec_change_coor[(b+3) % 4], vec_change_coor[(b+1) % 4], 4,-1,-1,-1,-1,-1};
                num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], decomp_elem[3], decomp_elem[2]};

            }

            else if ((val_f[(b+3) % 4]<accuracy) && (val_f[(b+3) % 4]>-accuracy))
            {
                decomp_elem[0] = coor_elem[(b+1) % 4];
                decomp_elem[1] = coor_elem[(b+2) % 4];
                decomp_elem[2] = boundary_pts[0];
                decomp_elem[3] = boundary_pts[1];

                corresp = {vec_change_coor[(b+1) % 4], vec_change_coor[(b+2) % 4], 5, vec_change_coor[(b+3) % 4], -1,-1,-1,-1,-1};
                num_elem = {coor_elem1[0], coor_elem1[1], coor_elem1[2], coor_elem1[3], decomp_elem[3], decomp_elem[2]};

            }





            *nb_poly = -1;
        }
    }
    /*if (*nb_poly > 0) {for (int i = 0; i < *nb_poly; ++i) {
        std::cout <<"Sub_elem " << i+1 <<" = " << decomp_elem[3*i] << "\n" << decomp_elem[3*i+1] << "\n" << decomp_elem[3*i+2] << "\n" <<std::endl;
    } } */
}
#endif //NOUVTRIANGLES_H
