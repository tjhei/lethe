#include <deal.II/lac/trilinos_vector.h>

//NUMERICS
#include <deal.II/numerics/data_out.h>

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

// Distributed
#include <deal.II/distributed/tria.h>


using namespace dealii;

void condensate(unsigned int nb_of_line, unsigned int new_nb, FullMatrix<double> &M, FullMatrix<double> &new_mat, Vector<double> &rhs, Vector<double> &new_rhs)
{
    int a;

    for (unsigned int i = nb_of_line-2; i >new_nb-1; --i) { // We begin at the second to last line, i is the number associated to line we're modifying

        for (unsigned int k = 0; k < nb_of_line-1-i ; ++k) { // How many times we modify the line
            a = nb_of_line-1-k;

            for (unsigned int j = 0; j < i+1; ++j) { // number associated to the column
                M(i,j) -= M(i,a)*M(a,j)/M(a,a);
            }

            rhs[i] -= rhs[a]*M(i,a)/M(a,a);
        }
    }

    // We modified the bottom of the matrix, now we have to reinject the coefficients
    // into the part of the matrix we want to return

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int k = nb_of_line-1; k > new_nb-1 ; --k) {
            for (unsigned int j = 0; j < k; ++j) {
                M(i,j)-=M(i,k)*M(k,j)/M(k,k);
            }
            rhs[i]-=rhs[k]*M(i,k)/M(k,k);
        }
    }

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int j = 0; j < new_nb; ++j) {
            new_mat[i][j]=M[i][j];
        }
        new_rhs[i]=rhs[i];
    }
}
