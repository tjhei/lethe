#include "calcinteg.h"


void integlocal(double Tdirichlet, double mat[4][4], double second_membre[4], std::vector<Point<2> > coor, std::vector<double> dist)
{
    // For a given element and the values of the distance function at its summit, gives back the elementary matrix in the finite elements method
    // for the heat equation

    double mat_elem2[6][6] = {{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0}};

    std::vector<int> corresp;
    std::vector<int> No_pts_solid;
    std::vector<Point<2> > num_elem;
    int nb_poly;
    std::vector<Point<2> > decomp_elem(9);

    nouvtriangles(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, coor, dist);

    // if (nb_poly>0) {std::cout << "nb_poly" << nb_poly << std::endl;}
    //double vec_change_coor[4] {3, 2, 0, 1}

    //if a quadrilateral is created :
    if (nb_poly<0)
    {

        // we calculate the values of the integral in the sub_element



        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                if ((corresp[i] == 4 || corresp[i] == 5) && (i!=j))
                {
                    mat_elem2[corresp[i]][corresp[j]]= 0;
                }

                else
                {
                    mat_elem2[corresp[i]][corresp[j]]=calcinteg(1, i, j, decomp_elem);
                }
            }

        }

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat[i][j] = mat_elem2[i][j];
            }
        }

        int i = 0;
        while (No_pts_solid[i] >= 0)
        {
            mat_elem2[No_pts_solid[i]][No_pts_solid[i]] = 1;
            second_membre[No_pts_solid[i]] = Tdirichlet;
            i++;
        }

        /*
          for (int i = 0; i<6 ; i++ ) {std::cout << "ligne " << i << " = " << mat_elem2[i][0] << ", " << mat_elem2[i][1] << ", " << mat_elem2[i][2] << ", " << mat_elem2[i][3] << ", " << mat_elem2[i][4] << ", " << mat_elem2[i][5] << std::endl;
        }

        std::cout << "\n" << std::endl;
        */

        // we will condense the 6 by 6 matrix to make the new dofs disappear, since their value is fixed on the boundary

        for (int i = 0; i < 4; ++i) {
            if ((dist[i])>0){ second_membre [i] = - Tdirichlet * (mat_elem2[i][4] + mat_elem2[i][5]);}
        }

    }








    // if triangles are created :
    if (nb_poly >0)
    {
        std::vector<Point<2> >     trg(3);
        std::vector<int>           corresp_loc(3);

        for (int n = 0; n < nb_poly; ++n)
        {
            trg[0]= decomp_elem[3*n];
            trg[1]= decomp_elem[3*n+1];
            trg[2]= decomp_elem[3*n+2];

            corresp_loc[0] = corresp[3*n];
            corresp_loc[1] = corresp[3*n+1];
            corresp_loc[2] = corresp[3*n+2];

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {

                    if ((corresp_loc[i] == 4 || corresp_loc[i] == 5) && (i!=j))
                    {
                        mat_elem2[corresp_loc[i]][corresp_loc[j]]+= 0;
                    }


                    else
                    {
                        mat_elem2[corresp_loc[i]][corresp_loc[j]] += calcinteg(0, i, j, trg);
                    }
                }
            }




        }

        int i = 0;
        while (No_pts_solid[i] >= 0)
        {
            mat_elem2[No_pts_solid[i]][No_pts_solid[i]] = 1;
            second_membre[No_pts_solid[i]] = Tdirichlet;
            i++;
        }

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat[i][j] = mat_elem2[i][j];
            }
        }

        /* for (int i = 0; i<6 ; i++ ) {std::cout << "ligne " << i << " = " << mat_elem2[i][0] << ", " << mat_elem2[i][1] << ", " << mat_elem2[i][2] << ", " << mat_elem2[i][3] << ", " << mat_elem2[i][4] << ", " << mat_elem2[i][5] << std::endl;
      }

      std::cout << "\n" << std::endl; */

    }







    // if no sub-element is created
    if (nb_poly == 0)
    {
        if (dist[0]>0)
        {
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    mat[i][j] = calcinteg(1, i, j, coor);
                }
            }
        }

        else
        {
            for (int i = 0; i < 4; ++i)
            {
                mat[i][i]=1;
                second_membre[i] = Tdirichlet;
            }
        }

    }

}

