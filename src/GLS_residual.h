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

                        unsigned int dofs_per_trg,

                        FullMatrix<double> &local_mat,
                        Vector<double> &local_rhs,

                        double viscosity_)

// decomp_trg gives the coordinates of the vertices of the triangle considered
// the 4 following arguments are the values on these vertices of respectively the components of the velocity, the pressure, and the gradients of the velocity and the pressure
// dofs_per_trg is the number of dof per vertex multiplied by 3 (in 2D it should be 3, 3D 4)
// local_mat and local_rhs are the local matrix and right hand side we are going to fill
// the condensation is not made here
// viscosity is a parameter of the problem (the viscosity obviously)

{
    local_mat = 0;
    local_rhs = 0;

    double h; // size of the element

    if (dim==2)
    {

        h = size_el(decomp_trg) ;

        std::vector<double>           div_phi_u                 (dofs_per_trg);
        std::vector<Tensor<1, dim> >  phi_u                     (dofs_per_trg);
        std::vector<Tensor<2, dim> >  grad_phi_u                (dofs_per_trg);
        std::vector<double>           phi_p                     (dofs_per_trg);
        std::vector<Tensor<1, dim> >  grad_phi_p                (dofs_per_trg);

        // we suppose the force vector equals zero

        // quadrature points + weight for a triangle : Hammer quadrature

        Vector<Point<dim>>             quad_pt;
        Vector<double>                 weight;

        unsigned int n_pt_quad = 4;

        quad_pt.reinit(n_pt_quad);
        weight.reinit(n_pt_quad);

        Point<dim> pt0(1./3., 1./3.);
        quad_pt(0) = pt0;
        weight(0) = -0.28125;

        Point<dim> pt1(1./5., 1./5.);
        quad_pt(1) = pt1;
        weight(1) = 0.264167;

        Point<dim> pt2(1./5., 3./5.);
        quad_pt(2) = pt2;
        weight(2) = 0.264167;

        Point<dim> pt3(3./5., 1./5.);
        quad_pt(3) = pt3;
        weight(3) = 0.264167;

        // Values and gradients interpolated at the quadrature points shall be stored in the following vectors
        Vector<double>  interpolated_v;
        double          interpolated_p = 0;

        Vector<Tensor<1,dim>>   interpolated_grad_v;
        Vector<double>          interpolated_grad_p;

        interpolated_v.reinit(dim);
        interpolated_grad_v.reinit(dim);
        interpolated_grad_p.reinit(dim);

        // jacobian is a constant in a triangle
        double jac = jacobian(0, 0,0, decomp_trg);

        for (unsigned int q=0; q<n_pt_quad; ++q)
        {
            double JxW = weight(q)*jac ;

            for (unsigned int i=0; i<dofs_per_trg; ++i)
            {
                for (unsigned int j=0; j<dofs_per_trg; ++j)
                {
                    local_mat(i, j) += (  viscosity_*scalar_product(grad_interp_velocity(euhhhhhhh???), grad_phi_u[i])
                                             + present_velocity_gradients[q]*phi_u[j]*phi_u[i]
                                             + grad_phi_u[j]*present_velocity_values[q]*phi_u[i]
                                             - div_phi_u[i]*phi_p[j]
                                             + phi_p[i]*div_phi_u[j]
                                             ) * JxW ;

        }

    }
}
