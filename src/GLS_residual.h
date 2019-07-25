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
void GLS_residual_trg(  Vector<Point<dim>>          decomp_trg,

                        Vector<Tensor<1,dim>>       veloc_trg,
                        Vector<double>              press_trg,
                        Vector<Tensor<2,dim>>       grad_veloc_trg,
                        Vector<Tensor<1,dim>>       grad_press_trg,

                        FullMatrix<double> &local_mat,
                        Vector<double> &local_rhs,

                        double viscosity_, double tau)

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
                pass_mat(i,var) = partial_coor_ref_2D(i,var, decomp_trg);
            }
        }

        // Vectors for the shapes functions //

        Vector<double>           div_phi_u_;
        Vector<Tensor<1,dim>>    phi_u;
        Vector<Tensor<2, dim> >  grad_phi_u;
        Vector<double>           phi_p;
        Vector<Tensor<1, dim> >  grad_phi_p;

        // we suppose the force vector equals zero ?

        // quadrature points + weight for a triangle : Hammer quadrature

        Vector<Point<dim>>             quad_pt;
        Vector<double>                 weight;

        unsigned int n_pt_quad = 4;

        quad_pt.reinit(n_pt_quad);
        weight.reinit(n_pt_quad);

        // Building the vector of quadrature points and vector associated weights //

        Point<dim> pt0(1./3., 1./3.);   Point<dim> pt1(1./5., 1./5.);   Point<dim> pt2(1./5., 3./5.);   Point<dim> pt3(3./5., 1./5.);
        quad_pt(0) = pt0;               quad_pt(1) = pt1;               quad_pt(2) = pt2;               quad_pt(3) = pt3;
        weight(0) = -0.28125;           weight(1) = 0.264167;           weight(2) = 0.264167;           weight(3) = 0.264167;

        // Values and gradients interpolated at the quadrature points shall be stored in the following vectors

        Tensor<1,dim+1>         interpolated_v;
        Tensor<1,dim>           interpolated_v_temp;
        double                  interpolated_p = 0;

        Vector<Tensor<1,dim>>   temp_interpolated_grad_v;
        Tensor<1, dim>          interpolated_grad_p; // ????????
        Tensor<2,dim>           interpolated_grad_v;

        // jacobian is a constant in a triangle
        double jac = jacobian(0, 0,0, decomp_trg);

        for (unsigned int q=0; q<n_pt_quad; ++q)
        {

            temp_interpolated_grad_v.reinit(dim);
            interpolated_v =0;

            double JxW = weight(q)*jac ;

            // Get the values of the variables at the quadrature point //

            interpolate_velocity(quad_pt(q), veloc_trg, interpolated_v_temp);
            interpolated_v(0) = interpolated_v_temp(0);
            interpolated_v(1) = interpolated_v_temp(1);

            interpolated_p = interpolate_pressure(quad_pt(q), press_trg);

            interpolate_grad_velocity(quad_pt(q), grad_veloc_trg, temp_interpolated_grad_v);
            interpolate_grad_pressure(quad_pt(q), grad_press_trg, interpolated_grad_p);

            interpolated_grad_v(0,0) = temp_interpolated_grad_v(0,0);
            interpolated_grad_v(1,1) = temp_interpolated_grad_v(1,1);
            interpolated_grad_v(0,1) = 0;
            interpolated_grad_v(1,0) = 0;

            // Get the values of the shape functions and their gradients at the quadrature points //

            div_phi_u_.reinit(dofs_per_trg);
            phi_p.reinit(dofs_per_trg);
            phi_u.reinit(dofs_per_trg);
            grad_phi_p.reinit(dofs_per_trg);
            grad_phi_u.reinit(dofs_per_trg);

            // phi_u is such as [[phi_u_0,0], [0, phi_v_0] , [0,0], [phi_u_1,0], ...]
            // phi_p is such as [0, 0, phi_p_0, 0, ...]
            // div_phi_u is such as [d(phi_u_0)/d(xi), d(phi_v_0)/d(eta) , 0, d(phi_u_1)/d(xi), ...] (xi, eta) being the system of coordinates used in the ref element
            // grad_phi_u is such as [[grad_phi_u_0], [grad_phi_v_0], [0, 0, ..], [grad_phi_u_1], ...]
            // grad_phi_p is such as [[0, 0, ..], [0, 0, ..], [grad_phi_p_0], [0, 0, ..], ...]

            Tensor<2, dim> e1_x_e1;
            e1_x_e1=0;
            e1_x_e1(0,0) =1;

            Tensor<2, dim> e2_x_e2;
            e2_x_e2=0;
            e2_x_e2(1,1) =1;

            for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex
                for (int j = 0; j < dim; ++j) { // j stands for u_j

                    div_phi_u_(3*i+j) = divergence(i, j, pass_mat) // We apply the passage matrix
                                                        // in order to change of coordinates
                    ;

                }
                grad_phi_u(3*i) = div_phi_u_(3*i) * e1_x_e1;
                grad_phi_u(3*i+1) = div_phi_u_(3*i+1) * e2_x_e2;

                phi_u(3*i)(0) = interp_pressure(quad_pt(q), i);
                phi_u(3*i)(0) = 0;
                phi_u(3*i+1)(1) = 0;
                phi_u(3*i+1)(1) = interp_pressure(quad_pt(q), i);

                phi_p(3*(i+1)-1) = interp_pressure(quad_pt(q), i);

                grad_interp_pressure(i, grad_phi_p(3*(i+1)-1), pass_mat);

                // we applied the change of coordinates to div_phi_u_, grad_phi_u, and to grad_phi_p
            }


            // Calculate and put in a local matrix and local rhs which will be returned
            for (unsigned int i=0; i<dofs_per_trg; ++i)
            {
                for (unsigned int j=0; j<dofs_per_trg; ++j)
                {
                    local_mat(i, j) += (     viscosity_ *divergence(i/3, i%3, pass_mat)
                                                        *divergence(j/3, j%3, pass_mat)                                 // ok + ok changement de coor

                                             + phi_u[j] * interpolated_grad_v * phi_u[i]                                // ok + ok changement de coor

                                             + div_phi_u_(j) * interpolated_v[i%3] * phi_u[i](j)                        // ok + ok changement de coor

                                             - (divergence(i/3, 0, pass_mat) + divergence(i/3, 1, pass_mat))
                                                                                                *phi_p[j]               // ok + ok changement de coor

                                             + phi_p[i]*
                                               (divergence(j/3, 0, pass_mat)+ divergence(j/3, 1, pass_mat))             // ok + ok changement de coor

                                             ) * JxW ;

                    // PSPG GLS term

                    local_mat(i, j) += tau* (  phi_u[j] * interpolated_grad_v * grad_phi_p[i]                           // ok + ok changement de coor
                                               + div_phi_u_(j) * interpolated_v * grad_phi_p[i]   //????????            // ok + ok changement de coor
                                               + grad_phi_p[j]*grad_phi_p[i]                                            // ok + ok changement de coor
                                               )  * JxW;

                    // SUPG term

                    local_mat(i, j) += tau*
                                    (  phi_u[j] * interpolated_grad_v * grad_phi_u[i] * interpolated_v                  // ok + ok changement de coor
                                    +  grad_phi_u[j]*present_velocity_values[q]*(present_velocity_values[q]* grad_phi_u[i])
                                    +  grad_phi_p[j]*(present_velocity_values[q]* grad_phi_u[i])
                                    )* JxW

                                    +  tau*
                                    (  present_velocity_gradients[q]*present_velocity_values[q]*(phi_u[j]*grad_phi_u[i])
                                    +  present_pressure_gradients[q]*(phi_u[j]*grad_phi_u[i])
                                 // -  force * (phi_u[j]*grad_phi_u[i])
                                    )* JxW;
                }

            }

        }

    }
}
