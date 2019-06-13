#include "calcinteg.h"
#include "nouvtriangles.h"
/* #include <vector>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
*/
using namespace dealii;

void integlocal(double Tdirichlet, double mat[4][4], std::vector<double> &second_membre, std::vector<Point<2> > coor, std::vector<double> dist)
{
    // For a given element and the values of the distance function at its summit, gives back the elementary matrix in the finite elements method
    // for the heat equation

    // VERY IMPORTANT : AS USUAL, THE NUMEROTATION USED BY DEAL.ii DOESN'T MATCH WITH WHAT I'VE DONE SO I HAD TO CREATE NEW ELEMENTS WITH THE GOOD NUMEROTATION
    // IN ORDER NOT TO WRITE EVERYTHING ALL OVER AGAIN

    // THE THING IS IT IS VERY SIMPLE TO PASS FROM ONE NUMEROTATION TO ANOTHER : YOU JUST HAVE TO APPLY THE VECTOR "vec_change_coor" TO THE INDEXES OF THE SYSTEM
    // THIS IS ONLY FOR THE QUADRILATERALS THOUGH. FOR THOSE, WE START BY TAKING THE COORDINATES AND APPLYING THE VECTOR IN ORDER TO HAVE THE RIGHT ORDER IN THE SUMMITS
    // AT THE END, WE HAVE TO APPLY THE VECTOR AGAIN WHEN STORING THE VALUES IN "mat" AND "second_membre" IN ORDER TO PUT THEM AT THE RIGHT PLACE FOR DEAL.ii


    double mat_elem2[6][6] = {{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0}};

    std::vector<int> corresp;
    std::vector<int> No_pts_solid;
    std::vector<Point<2> > num_elem;
    int nb_poly;
    std::vector<Point<2> > decomp_elem(9);

    nouvtriangles(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, coor, dist); // determines the number of sub-elements

    // No_pts_solid allows to know which points are in the solid : it gives the numerotation associated to the summits in the solid, has a length of four elements
    // you just have to see trough its elements and when No_pts_solid[i] < 0, it means there are no more summits in the solid

    // corresp allows to know the equivalence between the numerotation among one subelement and the numerotation of the element

    // num_elem[i] gives the coordinates of the point associated to the number i in the numerotation of the element

    // decomp_elem is the decomposition in sub-elements (gives the coordinates in the right order for deal.ii)

    // nb_poly gives the number of sub-elements created






    int vec_change_coor[4] {3, 2, 0, 1}; // VERY IMPORTANT, TAKE A LOOK AT THE DESCRITPION OF THE FUNCTION





    //if a quadrilateral is created :
    if (nb_poly<0)
    {
        double w[5], x[5];
        x[0] = -0.906179845938664;
        x[1] = -0.538469310105683;
        x[2] = 0.0;
        x[3] = 0.538469310105683;
        x[4] = 0.906179845938664;

        w[0] = 0.236926885056189;
        w[1] = 0.478628670499365;
        w[2] = 0.568888889888889;
        w[3] = 0.478628670499365;
        w[4] = 0.236926885056189;

        // we calculate the values of the integral in the sub_element


        double melem[4][4] = {{0, 0, 0, 0},{0, 0, 0, 0},{0, 0, 0, 0},{0, 0, 0, 0}};
        double jac =0;
        double x1,x2,x3,x4,y1,y2,y3,y4,B11,B12,B21,B22,C11,C21;

        x1 = decomp_elem[3][0];
        x2 = decomp_elem[2][0];
        x3 = decomp_elem[0][0];
        y1 = decomp_elem[3][1];
        y2 = decomp_elem[2][1];
        y3 = decomp_elem[0][1];
        x4 = decomp_elem[1][0];
        y4 = decomp_elem[1][1];

        B11 = (x1-x2-x3+x4)/4;
        B12 = (x1+x2-x3-x4)/4;
        B21 = (y1-y2-y3+y4)/4;
        B22 = (y1+y2-y3-y4)/4;

        C11 = (x1-x2+x3-x4)/4;

        C21 = (y1-y2+y3-y4)/4;
        double M11, M12, M22;
        //double M[2][2];

        double val;
        std::vector<double> gradi(2), gradj(2);
        double ai,bi, aj,bj;

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {

                val = 0;

                for (int ii = 0; ii < 5; ++ii) //quadrature formula for a degree 8 polynomial ( jk degree 2, Bk degree 2 and t(Bk) too, each grad is of degree 1)
                {
                    for (int jj = 0; jj < 5; ++jj)
                    {
                        jac = jacobian(1, x[ii],x[jj], decomp_elem);
                        M11 = ((B22+x[ii]*C21)*(B22+x[ii]*C21)+(B12+x[ii]*C11)*(B21+x[ii]*C11))/(jac*jac);
                        M12 = (-((B22+x[ii]*C21)*(B21+x[jj]*C21)+(B11+x[jj]*C11)*(B12+x[ii]*C11)))/(jac*jac);
                        M22 = ((B21+x[jj]*C21)*(B21+x[jj]*C21)+(B11+x[jj]*C11)*(B11+x[jj]*C11))/(jac*jac);

                        derinterpquad(i, x[ii], x[jj], gradi);
                        ai=gradi[0];
                        bi=gradi[1];

                        derinterpquad(j, x[ii], x[jj], gradj);
                        aj=gradj[0];
                        bj=gradj[1];

                        val += jac*(ai*(M11*aj+M12*bj) + bi*(M12*aj+M22*bj))*w[ii]*w[jj];
                    }
                }

                melem[vec_change_coor[i]][vec_change_coor[j]] = val;


                if ((corresp[i] == 4 || corresp[i] == 5) && (i!=j))
                {
                    mat_elem2[corresp[i]][corresp[j]]= 0;
                }

                else
                {
                    //mat_elem2[corresp[i]][corresp[j]]=calcinteg(1, i, j, decomp_elem);
                    mat_elem2[corresp[i]][corresp[j]] = melem[i][j];
                }
                mat[i][j] = 0;
            }

        }

        int i = 0;
        while (No_pts_solid[i] >= 0)
        {
            mat_elem2[No_pts_solid[i]][No_pts_solid[i]] = 1;
            second_membre[vec_change_coor[No_pts_solid[i]]] += Tdirichlet; // not sure about this, i'll have to have a look at nouvtriangles again //////////////////////
            ////////////////////////////////////////////////////////////////////////////
            /////////////////
            /// //////////////
            /// ///////////////
            i++;
        }

        /*
          for (int i = 0; i<6 ; i++ ) {std::cout << "ligne " << i << " = " << mat_elem2[i][0] << ", " << mat_elem2[i][1] << ", " << mat_elem2[i][2] << ", " << mat_elem2[i][3] << ", " << mat_elem2[i][4] << ", " << mat_elem2[i][5] << std::endl;
        }

        std::cout << "\n" << std::endl;
        */

        // we will condense the 6 by 6 matrix to make the new dofs disappear, since their value is fixed on the boundary

        for (int i = 0; i < 4; ++i) {
            if ((dist[i])>0){ second_membre [vec_change_coor[i]] += - Tdirichlet * (mat_elem2[i][4] + mat_elem2[i][5]);}
        }

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat[vec_change_coor[i]][vec_change_coor[j]] = mat_elem2[i][j];
            }
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

            double Melem[3][3] = {{0, 0, 0},{0, 0, 0},{0, 0, 0}};

            double x1, x2, x3, y1, y2, y3;
            double jac;

            x1 = trg[0][0];
            x2 = trg[1][0];
            x3 = trg[2][0];
            y1 = trg[0][1];
            y2 = trg[1][1];
            y3 = trg[2][1];

            jac = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);

                    double M11, M22, M12;
                    M11 = (y3-y1)*(y3-y1)+(x1-x3)*(x1-x3);
                    M12 = (y1-y2)*(y3-y1)+(x2-x1)*(x1-x3);
                    M22 = (y1-y2)*(y1-y2)+(x2-x1)*(x2-x1);

                    double a11,a22,a33,a32;
                    a11 = 1.0/(2*jac)*(M11+2*M12+M22);
                    a22 = 1.0/(2*jac)*(M11+M12);
                    a33 = 1.0/(2*jac)*(M22+M12);
                    a32 = 1.0/(2*jac)*M12;

                    double Mr[3][3] = {{a11, -a22, -a33},{-a22, a22, a32}, {-a33, a32, a33}};

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {

                    if ((corresp_loc[i] == 4 || corresp_loc[i] == 5) && (i!=j))
                    {

                        mat_elem2[corresp_loc[i]][corresp_loc[j]]= 0;
                        //double a = calcinteg(0, i, j, trg);
                        //Melem[i][j] = a;
                    }


                    else
                    {
                       //double a = calcinteg(0, i, j, trg);
                         //  Melem[i][j] = a;
                        /// NON /////////////////////IL Y A UN PROBLÈME DANS L'ASSIGNATION DES VALEURS DANS LA MATRICE QUAND ON UTILISE CALCINTEG, ÇA ME SOULE DE CHERCHER DONC FAUDRA S'Y ATTAQUER UN JOUR OU SUPPRIMER CETTE PARTIE DU CODE
                        /// LE PB VIENT VISIBLEMENT DE LA VALEUR CALCULÉE PAR CALCINTEG JSP PQ JPP DE CETTE FONCTION
                        mat_elem2[corresp_loc[i]][corresp_loc[j]] += Mr[i][j];

                        /// A REPRENDRE UN JOUR SI J'AI LE TEMPS CE QUI EST HAUTEMENT IMPROBABLE

                        //if (nb_poly==1)  {std::cout << "calcinteg : " << a << /* calcinteg(0, i, j, trg) << */ std::endl;}

                    }
                    //std::cout << "val m elem " << Melem[i][j] << ", " << Mr[i][j] << std::endl;
                 }
                /// UTILISER Mr EST SUFFISANT MAIS JE COMPRENDS PAS PK ÇA FONCTIONNE PAS
            }




        }

        int i = 0;
        while (No_pts_solid[i] >= 0)
        {
            mat_elem2[No_pts_solid[i]][No_pts_solid[i]] = 1;
            second_membre[No_pts_solid[i]] += Tdirichlet;
            i++;
        }

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat[i][j] = mat_elem2[i][j];
            }
        }

        //if (nb_poly>0) {for (int i = 0; i<6 ; i++ ) {std::cout << "ligne " << i << " = " << mat_elem2[i][0] << ", " << mat_elem2[i][1] << ", " << mat_elem2[i][2] << ", " << mat_elem2[i][3] << ", " << mat_elem2[i][4] << ", " << mat_elem2[i][5] << std::endl;}}
//        std::cout << "pts_solid : " << No_pts_solid[0] << ", " << No_pts_solid[1] << ", " << No_pts_solid[2] << ", " <<No_pts_solid[3] << std::endl;
//        std::cout << "\n" << std::endl;

    }









    // if no sub-element is created
    if (nb_poly == 0)
    {
        if (dist[0]>0)
        {
//            double w[5], x[5];
//            x[0] = -0.906179845938664;
//            x[1] = -0.538469310105683;
//            x[2] = 0.0;
//            x[3] = 0.538469310105683;
//            x[4] = 0.906179845938664;

//            w[0] = 0.236926885056189;
//            w[1] = 0.478628670499365;
//            w[2] = 0.568888889888889;
//            w[3] = 0.478628670499365;
//            w[4] = 0.236926885056189;

//            // we calculate the values of the integral in the sub_element


//            double melem[4][4] = {{0, 0, 0, 0},{0, 0, 0, 0},{0, 0, 0, 0},{0, 0, 0, 0}};
//            double jac =0;
//            double x1,x2,x3,x4,y1,y2,y3,y4,B11,B12,B21,B22,C11,C21;

//            x1 = coor[3][0];
//            x2 = coor[2][0];
//            x3 = coor[0][0];
//            y1 = coor[3][1];
//            y2 = coor[2][1];
//            y3 = coor[0][1];
//            x4 = coor[1][0];
//            y4 = coor[1][1];

//            B11 = (x1-x2-x3+x4)/4;
//            B12 = (x1+x2-x3-x4)/4;
//            B21 = (y1-y2-y3+y4)/4;
//            B22 = (y1+y2-y3-y4)/4;

//            C11 = (x1-x2+x3-x4)/4;

//            C21 = (y1-y2+y3-y4)/4;
//            double M11, M12, M22;
//            //double M[2][2];

//            double val;
//            std::vector<double> gradi(2), gradj(2);
//            double ai,bi, aj,bj;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
//                    val = 0;

//                    for (int ii = 0; ii < 5; ++ii) //quadrature formula for a degree 8 polynomial ( jk degree 2, Bk degree 2 and t(Bk) too, each grad is of degree 1)
//                    {
//                        for (int jj = 0; jj < 5; ++jj)
//                        {
//                            jac = jacobian(1, x[ii],x[jj], coor);
//                            M11 = ((B22+x[ii]*C21)*(B22+x[ii]*C21)+(B12+x[ii]*C11)*(B21+x[ii]*C11))/(jac*jac);
//                            M12 = (-((B22+x[ii]*C21)*(B21+x[jj]*C21)+(B11+x[jj]*C11)*(B12+x[ii]*C11)))/(jac*jac);
//                            M22 = ((B21+x[jj]*C21)*(B21+x[jj]*C21)+(B11+x[jj]*C11)*(B11+x[jj]*C11))/(jac*jac);

//                            derinterpquad(i, x[ii], x[jj], gradi);
//                            ai=gradi[0];
//                            bi=gradi[1];

//                            derinterpquad(j, x[ii], x[jj], gradj);
//                            aj=gradj[0];
//                            bj=gradj[1];

//                            val += jac*(ai*(M11*aj+M12*bj) + bi*(M12*aj+M22*bj))*w[ii]*w[jj];
//                        }
//                    }

//                    mat[vec_change_coor[i]][vec_change_coor[j]] = val;

                    mat[vec_change_coor[i]][vec_change_coor[j]] = calcinteg(1, i, j, coor);
                }
            }
        }

        else
        {
            for (int i = 0; i < 4; ++i)
            {
                mat[i][i]=1;
                second_membre[i] += Tdirichlet;
            }
        }

    }

}

//Coor elem : 1.75 1.75, 2 1.75, 1.75 2, 2 2
// Val f dist : 1.32375, 1.4939, 1.51864, 1.67643

//Ligne 0 de la matrice : 0.666667, -0.166667, -0.333333, -0.166667
//Ligne 1 de la matrice : -0.166667, 0.666667, -0.166667, -0.333333
//Ligne 2 de la matrice : -0.333333, -0.166667, 0.666667, -0.166667
//Ligne 3 de la matrice : -0.166667, -0.333333, -0.166667, 0.666667

