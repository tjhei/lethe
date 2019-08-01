//BASE
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>


//LAC
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_solver.h>
// Trilinos includes
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>


//GRID
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>


//DOFS
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

//FE
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

//Numerics
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

// Distributed
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>

// Added
#include "trg_tools.h"


// Finally, this is as in previous programs:
using namespace dealii;


template<int dim>
void GLS_residual_trg(  std::vector<Point<dim>>          decomp_trg,

                        std::vector<Tensor<1,dim>>       veloc_trg,
                        std::vector<double>              press_trg,

                        Tensor<1, dim>  force,

                        FullMatrix<double> &local_mat,
                        Vector<double> &local_rhs,

                        double viscosity_)

// decomp_trg gives the coordinates of the vertices of the triangle considered
// the 4 following arguments are the values on these vertices of respectively the components of the velocity, the pressure, and the gradients of the velocity and the pressure

// local_mat and local_rhs are the local matrix and right hand side we are going to fill
// the condensation is not made here
// viscosity is a parameter of the problem (the viscosity obviously)
// tau is the correction coefficient
{
    local_mat = 0;
    local_rhs = 0;

    double h; // size of the element

    unsigned int dofs_per_trg = (dim+1)*(dim+1);
    // dofs_per_trg is the number of dof per vertex multiplied by 3 (in 2D it should be 3, 3D 4)

    if (dim==2)
    {
        h = size_el(decomp_trg) ;

        // Passage matrix from element coordinates to ref element coordinates, it is necessary to calculate the derivates

        Tensor<2, dim>       pass_mat; // we express the coordinates of the ref element depending on the coordinates of the actual element

        for (int i = 0; i < dim; ++i) {
            for (int var = 0; var < dim; ++var) {
                pass_mat[i][var] = partial_coor_ref_2D(i,var, decomp_trg);
            }
        }

        int dofs_per_node = dim+1;

        // Vectors for the shapes functions //

        std::vector<double>           div_phi_u_(dofs_per_trg);
        std::vector<Tensor<1,dim>>    phi_u(dofs_per_trg);
        std::vector<Tensor<2,dim>>    grad_phi_u(dofs_per_trg);
        std::vector<double>           phi_p(dofs_per_trg);
        std::vector<Tensor<1,dim>>    grad_phi_p(dofs_per_trg);

        // the part for the force vector is not implemented yet

        // quadrature points + weight for a triangle : Hammer quadrature

        unsigned int n_pt_quad = 4;
        std::vector<Point<dim>>             quad_pt(n_pt_quad);
        std::vector<double>                 weight(n_pt_quad);

        // Building the vector of quadrature points and vector associated weights //

        Point<dim> pt0(1./3., 1./3.);   Point<dim> pt1(1./5., 1./5.);   Point<dim> pt2(1./5., 3./5.);   Point<dim> pt3(3./5., 1./5.);
        quad_pt[0] = pt0;               quad_pt[1] = pt1;               quad_pt[2] = pt2;               quad_pt[3] = pt3;
        weight[0] = -0.281250;          weight[1] = 0.264167;           weight[2] = 0.264167;           weight[3] = 0.264167;

        // Values and gradients interpolated at the quadrature points shall be stored in the following vectors

        Tensor<1,dim>           interpolated_v;
        double                  interpolated_p = 0;
        Tensor<1,dim>           interpolated_grad_p;
        Tensor<2,dim>           interpolated_grad_v;

        // jacobian is a constant in a triangle
        double jac = jacobian(0, 0,0, decomp_trg);

        for (unsigned int q=0; q<n_pt_quad; ++q)
        {
            interpolated_v =0;

            double JxW = weight[q]*jac ;

            // Get the values of the variables at the quadrature point //

            interpolate_velocity(quad_pt[q], veloc_trg, interpolated_v);

            interpolated_p = interpolate_pressure(quad_pt[q], press_trg);

            interpolate_grad_velocity( veloc_trg, interpolated_grad_v); // grad of the shape functions are constants in triangles
            interpolate_grad_pressure( press_trg, interpolated_grad_p);

            // Build the parameter of stabilisation
            const double u_mag= std::max(interpolated_v.norm(),1e-3);
            double tau = 1./ std::sqrt(std::pow(2.*u_mag/h,2)+9*std::pow(4*viscosity_/(h*h),2));

            // Get the values of the shape functions and their gradients at the quadrature points //

            // phi_u is such as [[phi_u_0,0], [0, phi_v_0] , [0,0], [phi_u_1,0], ...]
            // phi_p is such as [0, 0, phi_p_0, 0, ...]
            // div_phi_u is such as [[d(phi_u_0)/d(xi), d(phi_v_0)/d(eta)], [d(phi_u_0)/d(xi), d(phi_v_0)/d(eta)] , [0, 0], [d(phi_u_1)/d(xi), d(phi_v_1)/d(eta)], ...] (xi, eta) being the system of coordinates used in the ref element
            // grad_phi_u is such as [[[grad_phi_u_0],[0, 0]], [[0, 0], [grad_phi_v_0]], [[0, 0], [0, 0]], [[grad_phi_u_1],[0, 0]], ...]
            // grad_phi_p is such as [[0, 0], [0, 0], [grad_phi_p_0], [0, 0], ...]

            Tensor<2, dim> e1_x_e1;     Tensor<2, dim> e2_x_e2;     Tensor<2, dim> e1_x_e2;     Tensor<2, dim> e2_x_e1;
            e1_x_e1=0;                  e2_x_e2=0;                  e1_x_e2=0;                  e2_x_e1=0;
            e1_x_e1[0][0] =1;           e2_x_e2[1][1] =1;           e1_x_e2[0][1] =1;           e2_x_e1[1][0]=1;



            for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex


                div_phi_u_[dofs_per_node*i] = divergence(i, 0, pass_mat) // We apply the passage matrix
                                                                // in order to change of coordinates
                ;
                div_phi_u_[dofs_per_node*i +1] = divergence(i, 1, pass_mat);

                grad_phi_u[dofs_per_node*i] = div_phi_u_[dofs_per_node*i] * e1_x_e1 + div_phi_u_[dofs_per_node*i+1] * e1_x_e2;
                grad_phi_u[dofs_per_node*i+1] = div_phi_u_[dofs_per_node*i] * e2_x_e1 + div_phi_u_[dofs_per_node*i+1] * e2_x_e2;

                phi_u[dofs_per_node*i][0] = shape_function(i, quad_pt[q]);
                phi_u[dofs_per_node*i][(0)] = 0;
                phi_u[dofs_per_node*i+1][(1)] = 0;
                phi_u[dofs_per_node*i+1][(1)] = shape_function(i, quad_pt[q]);

                phi_p[dofs_per_node*(i+1)-1] = shape_function(i, quad_pt[q]);

                grad_shape_function(i, grad_phi_p[dofs_per_node*(i+1)-1], pass_mat);

                // we applied the change of coordinates to div_phi_u_, grad_phi_u, and to grad_phi_p
            }

            // Calculate and put in a local matrix and local rhs which will be returned
            for (unsigned int i=0; i<dofs_per_trg; ++i)
            {

                // matrix terms

                for (unsigned int j=0; j<dofs_per_trg; ++j)
                {
                    local_mat(i, j) += (     viscosity_ * scalar_product(grad_phi_u[j], grad_phi_u[i])          // ok + ok changement de coor

                                             + phi_u[j] * interpolated_grad_v * phi_u[i]                        // ok + ok changement de coor

                                             + (grad_phi_u[j] * interpolated_v) * phi_u[i]                      // ok + ok changement de coor

                                             - (div_phi_u_[i])*phi_p[j]                                         // ok + ok changement de coor

                                             + phi_p[i]*(div_phi_u_[j])                                         // ok + ok changement de coor

                                             ) * JxW ;

                    // PSPG GLS term //

                    local_mat(i, j) += tau* (  grad_phi_p[i] * interpolated_grad_v * phi_u[j]                   // ok + ok changement de coor
                                               + grad_phi_u[j] * interpolated_v * grad_phi_p[i]                 // ok + ok changement de coor
                                               + grad_phi_p[j]*grad_phi_p[i]                                    // ok + ok changement de coor
                                               )  * JxW;

                    // SUPG term //

                    local_mat(i, j) += tau* // convection and velocity terms
                                    (  (interpolated_grad_v * phi_u[j]) * (grad_phi_u[i] * interpolated_v)      // ok + ok changement de coor
                                    +  (grad_phi_u[i] * interpolated_v) * (grad_phi_u[j] * interpolated_v)      // ok + ok changement de coor
                                    +  phi_u[j] * ((interpolated_grad_v * interpolated_v) * grad_phi_u[i])      // ok + ok changement de coor
                                    )* JxW

                                    +  tau* // pressure terms
                                    (  grad_phi_p[j] * (grad_phi_u[i] * interpolated_v)                         // ok + ok changement de coor
                                    +  phi_u[j]*(interpolated_grad_p *grad_phi_u[i])                            // ok + ok changement de coor
//                                    -  force * (phi_u[j]*grad_phi_u[i])
                                    )* JxW;
                }



                // Evaluate the rhs, with corrective terms

                double present_velocity_divergence =  trace(interpolated_grad_v);

                local_rhs(i) += ( - viscosity_*scalar_product(interpolated_grad_v,grad_phi_u[i])
                                  - interpolated_grad_v * interpolated_v * phi_u[i]
                                  + interpolated_p * div_phi_u_[i]
                                  - present_velocity_divergence*phi_p[i]
//                                  + force * phi_u[i]
                                ) * JxW;


                // PSPG GLS term for the rhs //
                local_rhs(i) +=  tau*(  - interpolated_grad_v * interpolated_v* grad_phi_p[i]
                                        - interpolated_grad_p * grad_phi_p[i]
//                                        + force * grad_phi_p[i]
                                     )  * JxW;

                // SUPG term for the rhs //
                local_rhs(i) += tau*(   - interpolated_grad_v * interpolated_v * (grad_phi_u[i] * interpolated_v )
                                        - interpolated_grad_p * (grad_phi_u[i] * interpolated_v)
//                                        + force *(interpolated_v* grad_phi_u[i])
                                    )   * JxW;

            }
        }

    }
}

void condensate_NS_trg(FullMatrix<double> cell_mat, Vector<double> cell_rhs, FullMatrix<double> new_mat, Vector<double> new_rhs)
{
    // for a decomposition in triangles IN 2D with the function nouvtriangles, we create 2 more points, so 2*3 dofs
    // we then have a [(4 + 2)*3]x[(4 + 2)*3] cell matrix, but we only want a [4*3]x[4*3]
    // Thus we have to condensate this matrix

    // this algorithm is similar to the "condensate.h" one.

    int a;
    unsigned int nb_of_line, new_nb;

    nb_of_line = 18;
    new_nb = 12;

    for (unsigned int i = nb_of_line-2; i >new_nb-1; --i) { // We begin at the second to last line, i is the number associated to line we're modifying

        for (unsigned int k = 0; k < nb_of_line-1-i ; ++k) { // How many times we modify the line
            a = nb_of_line-1-k;

            for (unsigned int j = 0; j < i+1; ++j) { // number associated to the column
                cell_mat(i,j) -= cell_mat(i,a)*cell_mat(a,j)/cell_mat(a,a);
            }

            cell_rhs[i] -= cell_rhs[a]*cell_mat(i,a)/cell_mat(a,a);
        }
    }

    // We modified the bottom of the matrix, now we have to reinject the coefficients
    // into the part of the matrix we want to return

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int k = nb_of_line-1; k > new_nb-1 ; --k) {
            for (unsigned int j = 0; j < k; ++j) {
                cell_mat(i,j)-=cell_mat(i,k)*cell_mat(k,j)/cell_mat(k,k);
            }
            cell_rhs[i]-=cell_rhs[k]*cell_mat(i,k)/cell_mat(k,k);
        }
    }

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int j = 0; j < new_nb; ++j) {
            new_mat[i][j]=cell_mat[i][j];
        }
        new_rhs[i]=cell_rhs[i];
    }
}
