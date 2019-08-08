#include <vector>
#include "deal.II/lac/full_matrix.h"
#include "deal.II/base/tensor.h"
#include "deal.II/lac/vector.h"

#include "ib_node_status.h"
#include "nouvtriangles.h"
#include "trg_tools_class.h"

using namespace dealii;

// We here check if we get the right evaluation for the elementary matrix and elementary rhs
// with GLS_residual, which will then be removed because no more used in the solver.
// Yet the code used in this file is pretty much the same as the function implemented in
// the class of the solver in ../../prototypes/IBSteadyNS/IBSteadyNS.cc


template<int dim>
void GLS_residual_trg_adapted(  TRG_tools<dim>  trg_,

                        FullMatrix<double> &local_mat,
                        Vector<double> &local_rhs,

                        double viscosity_, double tau)

// This fucntions is used here because it is pretty much identical to what we use in the solver
// It has been adapted to evaluate the value of the coefficients of the elementary matrix and rhs at one point of the quadrature used in the solver
// local_mat and local_rhs are the local matrix and right hand side we are going to fill
// viscosity is a parameter of the problem (the viscosity obviously)

{
    local_mat = 0;
    local_rhs = 0;



    unsigned int dofs_per_trg = (dim+1)*(dim+1);
    // dofs_per_trg is the number of dof per vertex multiplied by 3 (in 2D it should be 3, 3D 4)

    if (dim==2)
    {


        // Vectors for the shapes functions //

        std::vector<double>           div_phi_u_(dofs_per_trg);
        std::vector<Tensor<1,dim>>    phi_u(dofs_per_trg);
        std::vector<Tensor<2,dim>>    grad_phi_u(dofs_per_trg);
        std::vector<double>           phi_p(dofs_per_trg);
        std::vector<Tensor<1,dim>>    grad_phi_p(dofs_per_trg);

        // the part for the force vector is not implemented yet

        // quadrature points + weight for a triangle : Hammer quadrature
        // Building the vector of quadrature points and vector associated weights //

        unsigned int n_pt_quad = 1;

        std::vector<Point<dim>>             quad_pt(n_pt_quad);
        std::vector<double>                 weight(n_pt_quad);
//        get_Quadrature_trg(quad_pt, weight);

        quad_pt[0][0]   = 0.2;
        quad_pt[0][1]   = 0.2;
        weight[0]       = 1;

        // Passage matrix from element coordinates to ref element coordinates, it is necessary to calculate the derivates

        Tensor<2, dim>       pass_mat; // we express the coordinates of the ref element depending on the coordinates of the actual element
        trg_.matrix_pass_elem_to_ref(pass_mat);

        // Values and gradients interpolated at the quadrature points shall be stored in the following vectors

        Tensor<1,dim>           interpolated_v;
        double                  interpolated_p = 0;
        Tensor<1,dim>           interpolated_grad_p;
        Tensor<2,dim>           interpolated_grad_v;

        // jacobian is a constant in a triangle
        double jac = trg_.jacob() ;

        for (unsigned int q=0; q<n_pt_quad; ++q)
        {
            interpolated_v =0;

            double JxW = weight[q]*jac ;

            // Get the values of the variables at the quadrature point //

            interpolated_p = trg_.interpolate_pressure(quad_pt[q]);
            trg_.interpolate_velocity(quad_pt[q], interpolated_v);
            trg_.interpolate_grad_pressure(interpolated_grad_p);
            trg_.interpolate_grad_velocity(interpolated_grad_v);


            // Get the values of the shape functions and their gradients at the quadrature points //

            // phi_u is such as [[phi_u_0,0], [0, phi_v_0] , [0,0], [phi_u_1,0], ...]
            // phi_p is such as [0, 0, phi_p_0, 0, ...]
            // div_phi_u is such as [d(phi_u_0)/d(xi), d(phi_v_0)/d(eta) , 0, d(phi_u_1)/d(xi), ...] (xi, eta) being the system of coordinates used in the ref element
            // grad_phi_u is such as [[[grad_phi_u_0],[0, 0]], [[0, 0], [grad_phi_v_0]], [[0, 0], [0, 0]], [[grad_phi_u_1],[0, 0]], ...]
            // grad_phi_p is such as [[0, 0], [0, 0], [grad_phi_p_0], [0, 0], ...]

            trg_.build_phi_p(quad_pt[q], phi_p);
            trg_.build_phi_u(quad_pt[q], phi_u);
            trg_.build_div_phi_u(pass_mat, div_phi_u_);
            trg_.build_grad_phi_p(pass_mat, grad_phi_p);
            trg_.build_grad_phi_u(pass_mat, grad_phi_u);


            // Calculate and put in a local matrix and local rhs which will be returned
            for (unsigned int i=0; i<dofs_per_trg; ++i)
            {

                // matrix terms

                for (unsigned int j=0; j<dofs_per_trg; ++j)
                {
                    local_mat(i, j) += (     viscosity_ * trace(grad_phi_u[j] * grad_phi_u[i])                      // TERME OK

                                             + phi_u[i] * interpolated_grad_v * phi_u[j]                            // TERME OK

                                             + (grad_phi_u[j] * interpolated_v) * phi_u[i]                          // TERME OK

                                             - (div_phi_u_[i])*phi_p[j]                                             // TERME OK

                                             + phi_p[i]*(div_phi_u_[j])                                             // TERME OK

                                             ) * JxW ;

                    // PSPG GLS term //

                    local_mat(i, j) += tau* (     phi_u[j]* (grad_phi_p[i]*interpolated_grad_v)                     // TERME OK
                                               + grad_phi_u[j] * interpolated_v * grad_phi_p[i]                     // TERME OK
                                               + grad_phi_p[j]*grad_phi_p[i]                                        // TERME OK
                                               )  * JxW;

                    // SUPG term //

                    local_mat(i, j) += tau*( // convection and velocity terms
                                      (interpolated_grad_v * phi_u[j]) * (grad_phi_u[i] * interpolated_v)           // TERME OK
                                    + (grad_phi_u[i] * interpolated_v) * (grad_phi_u[j] * interpolated_v)           // TERME OK
                                    +  phi_u[j] * ((interpolated_grad_v * interpolated_v) * grad_phi_u[i])          // TERME OK
                                    )* JxW

                                    +  tau* // pressure terms
                                  (  grad_phi_p[j] * (grad_phi_u[i] * interpolated_v)                               // TERME OK
                                    +  phi_u[j]*(interpolated_grad_p *grad_phi_u[i])                                // TERME OK
                                    )* JxW;
                }



                // Evaluate the rhs, with corrective terms

                double present_velocity_divergence =  trace(interpolated_grad_v);

                local_rhs(i) += ( - viscosity_*trace(interpolated_grad_v*grad_phi_u[i])
                                  - interpolated_grad_v * interpolated_v * phi_u[i]
                                  + interpolated_p * div_phi_u_[i]
                                  - present_velocity_divergence*phi_p[i]
                                ) * JxW;


                // PSPG GLS term for the rhs //
                local_rhs(i) +=  tau*(  - interpolated_grad_v * interpolated_v* grad_phi_p[i]
                                        - interpolated_grad_p * grad_phi_p[i]
                                     )  * JxW;

                // SUPG term for the rhs //
                local_rhs(i) += tau*(   - interpolated_grad_v * interpolated_v * (grad_phi_u[i] * interpolated_v )
                                        - interpolated_grad_p * (grad_phi_u[i] * interpolated_v)
                                    )   * JxW;

            }
        }

    }
}


void test_residual()
{
    // building a "cell" and the distance to the boundary associated to each vertex
    std::vector<Point<2>>   coor(4);
    std::vector<double>     dist(4);

    // dist will be chosen so that the element is decomposed into one triangle (3 vertices in the solid, one in the fluid)

    dist[0] = 1.;
    dist[1] = -1.;
    dist[2] = -1.;
    dist[3] = -1.5;

    coor[0][0] = 0;
    coor[0][1] = 0;
    coor[1][0] = 2;
    coor[1][1] = 0;
    coor[2][0] = 0;
    coor[2][1] = 2;
    coor[3][0] = 2;
    coor[3][1] = 2;

    // the decomposed triangle should be formed by (0, 0), (1, 0), (0, 1), the ref triangle

    // Initializing tools for the conformal decomposition
    std::vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
    int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
    std::vector<Point<2> >               num_elem(6);
    std::vector<int>                     corresp(9);
    std::vector<node_status>             pts_statut(4);

    // initializing a vector of velocity and vector of pressure on the nodes of the triangle built with the decomposition function
    std::vector<Tensor<1,2>>             velocity_node(3);
    std::vector<double>                  pressure_node(3);

    // Matrix and RHS sides;
    unsigned int dofs_per_trg = 3*3;
    FullMatrix<double>   cell_mat (dofs_per_trg, dofs_per_trg);
    Vector<double>       cell_rhs (dofs_per_trg);

    // We decompose the element
    decomposition(corresp, pts_statut, num_elem, decomp_elem, &nb_poly, coor, dist);

    // setting values to v and p on the vertices of the triangle
    velocity_node[0][0] = 0.1;
    velocity_node[0][1] = 0.1;
    velocity_node[1][0] = 0.2;
    velocity_node[1][1] = -0.2;
    velocity_node[2][0] = -0.1;
    velocity_node[2][1] = 0.2;

    pressure_node[0] = 0.1;
    pressure_node[1] = 0.5;
    pressure_node[2] = 0.75;

    // Initializing a TRG_tools object
    TRG_tools<2>    trg;
    trg.set_coor_trg(decomp_elem);
    trg.set_dofs_per_node(3);
    trg.set_P_on_vertices(pressure_node);
    trg.set_V_on_vertices(velocity_node);

    Tensor<1, 2>        interpolated_v;
    double              interpolated_p;
    Tensor<1, 2>        interpolated_grad_p;
    Tensor<2, 2>        interpolated_grad_v;

    // we will check the calculus at the point (0.2, 0.2), with a given weight matching with one of the quadrature points we use in the solver
    double weight = 1;
    Point<2>         Pt_quad (0.2, 0.2);

    // get the interpolated vectors we need at the point of evaluation
    interpolated_p = trg.interpolate_pressure(Pt_quad);
    trg.interpolate_velocity(Pt_quad, interpolated_v);
    trg.interpolate_grad_pressure(interpolated_grad_p);
    trg.interpolate_grad_velocity(interpolated_grad_v);

    // Creating the viscosity which is necesary for the function
    double viscosity =1;
    double tau =1;

    GLS_residual_trg_adapted(trg, cell_mat, cell_rhs, viscosity, tau); // reprendre ce qui est dans le script pour faire le calcul en un seul pt



    // reprendre la feuille pour évaluer à la main les termes de la matrice au point de quad

        FullMatrix<double>  Check_mat(9,9);
        Vector<double>      Check_rhs(9);

        Check_mat =0;
        Check_rhs =0;

        double phi0 = trg.shape_function(0, Pt_quad);        //we will store in phi(j) the value of the shape function associated to the vertex j at the point of quad
        double phi1 = trg.shape_function(1, Pt_quad);
        double phi2 = trg.shape_function(2, Pt_quad);

        // same for the gradients of the shape functions
        Tensor<1, 2>        grad_phi_0;
        grad_phi_0[0] =-1;
        grad_phi_0[1] =-1;
        Tensor<1, 2>        grad_phi_1;
        grad_phi_1[0] =1;
        grad_phi_1[1] =0;
        Tensor<1, 2>        grad_phi_2;
        grad_phi_2[0] =0;
        grad_phi_2[1] =1;

        std::vector<double>             phi(3);
        std::vector<Tensor<1, 2>>       grad_phi(3);

        phi[0]=phi0;
        phi[1]=phi1;
        phi[2]=phi2;
        grad_phi[0]=grad_phi_0;
        grad_phi[1]=grad_phi_1;
        grad_phi[2]=grad_phi_2;

        for (int i = 0; i < 9; ++i) {

            // evaluating the matrix

            for (int j = 0; j < 9; ++j) {
                if (j%3!=2)
                {
                    if (i%3!=2)
                    {
                        Check_mat(i,j) += viscosity * grad_phi[i/3][j%3] * grad_phi[j/3][i%3]*weight;
                        Check_mat(i,j) += phi[j/3] * interpolated_grad_v[i%3][j%3] * phi[i/3]*weight;
                        Check_mat(i,j) += (interpolated_v[0]*grad_phi[j/3][0] + interpolated_v[1]*grad_phi[j/3][1])* phi[i/3] * (i%3==j%3)*weight;
                    }
                    else {
                        Check_mat(i,j) += phi[i/3]*grad_phi[j/3][j%3]*weight;
                    }
                }
                else {
                        if (i%3!=2)
                    {
                        Check_mat(i,j) += - phi[j/3] * grad_phi[i/3][i%3]*weight;
                    }


                }

                // PSPG part
                if (j%3!=2)
                {
                    if (i%3==2){
                        Check_mat(i,j) += tau*phi[j/3] * (interpolated_grad_v[0][j%3] * grad_phi[i/3][0] + interpolated_grad_v[1][j%3] * grad_phi[i/3][1])*weight;
                        Check_mat(i,j) += tau*grad_phi[i/3][j%3]*(interpolated_v[0]*grad_phi[j/3][0] + interpolated_v[1]*grad_phi[j/3][1]) * weight;
                    }
                }
                else {
                    if (i%3==2)
                    {
                        Check_mat(i,j) += tau*grad_phi[i/3]*grad_phi[j/3]*weight;
                    }

                }

                //SUPG part
                if (j%3!=2)
                {
                    if (i%3!=2){
                        Check_mat(i,j) += tau*phi[j/3]*interpolated_grad_v[i%3][j%3]*(interpolated_v[0]*grad_phi[i/3][0] + interpolated_v[1]*grad_phi[i/3][1]) * weight;
                        Check_mat(i,j) += tau*(interpolated_v[0]*grad_phi[j/3][0]+interpolated_v[1]*grad_phi[j/3][1])*(interpolated_v[0]*grad_phi[i/3][0]+interpolated_v[1]*grad_phi[i/3][1])*weight*(i%3==j%3);
                        Check_mat(i,j) += tau*phi[j/3]*grad_phi[i/3][j%3]*(interpolated_v[0]*interpolated_grad_v[i%3][0]+interpolated_v[1]*interpolated_grad_v[i%3][1])*weight;
                    }
                }
                // pressure terms
                if (j%3==2) {
                    if (i%3!=2)
                    {
                        Check_mat(i,j) += tau * grad_phi[j/3][i%3] * (interpolated_v[0]*grad_phi[i/3][0]+interpolated_v[1]*grad_phi[i/3][1]) * weight;
                    }
                }
                else {
                    if (i%3!=2)
                    {
                        Check_mat(i,j) += tau * phi[j/3]*interpolated_grad_p[i%3]*grad_phi[i/3][j%3]*weight;
                    }

                }
            }

            // evaluating the rhs
            // will be done later

            if (i%3!=2)
            {
                Check_rhs(i) += 0;
            }
        }



        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (std::abs(Check_mat(i,j) - cell_mat(i,j))>1e-10)
                    std::cout << "failed to build the coef "  <<  i << ","<< j << " of the matrix, error : " << std::abs(Check_mat(i,j) - cell_mat(i,j)) << std::endl;
            }
//            if (std::abs(Check_rhs[(i)] - cell_rhs[(i)])>1e-10)
//                std::cout << "failed to build the coef " << i << " of the rhs, error : " << std::abs(Check_rhs[(i)] - cell_rhs[(i)]) << std::endl;
        }

}


int main()
{
    test_residual();
    return 1;
}
