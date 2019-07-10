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

void condensate(unsigned int nb_of_line, unsigned int new_nb, FullMatrix<double> &M, FullMatrix<double> &new_mat, std::vector<double> &rhs, std::vector<double> &new_rhs)
{
//    unsigned int count =0;
//    for (int k = nb_of_line-2; k >0 ; --k) {
//        for (unsigned int i = 0; i < count+1; ++i) {
//            for (unsigned int j = 0; j < nb_of_line-1-i; ++j) {
//                M[k][j]=M[k][j]-M[nb_of_line-1-i][j] *
//                        M[k][nb_of_line-1-i] / M[nb_of_line-1-i][nb_of_line-1-i];
//                //std::cout << M_copy[nb_of_line-1-i][j] << M_copy[k][nb_of_line-1-i] << M_copy[nb_of_line-1-i][nb_of_line-1-i] << std::endl;
//                // for example, if we have 4 dofs, we will have a 6x6 matrix
//                //let's take the fifth line : M6_renum[4][0] += -M6_renum[5][0]*M6_renum[4][5]/M6_renum[5][5]
//            }

//            rhs[k]+=-rhs[nb_of_line-1-i] *
//                    M[k][nb_of_line-1-i]/M[nb_of_line-1-i][nb_of_line-1-i];
//        }
//        if (new_nb<nb_of_line-count)
//            count ++;
//    }

    int a;

    for (unsigned int i = nb_of_line-2; i >new_nb-1; --i) { // We begin at the second to last line, i is the number associated to line we're modifying

        for (unsigned int k = 0; k < nb_of_line-1-i ; ++k) { // How many times we modify the line
            a = nb_of_line-1-k;

            for (unsigned int j = 0; j < i+1; ++j) { // number associated to the column
                M(i,j) -= M(i,a)*M(a,j)/M(a,a);
//                std::cout << i << a << " : "<< M(i,a)<< a << i << " : "  << M(a,i) << a << a << " : "  << M(a,a) << " " << M(i,a)*M(a,i)/M(a,a) << std::endl;
//                std::cout << "M" << i << j << " : " <<M(i,j)<< std::endl;
            }

            rhs[i] -= rhs[a]*M(i,a)/M(a,a);
        }
        std::cout << "\n cond mat : " << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << M[i][0] << " " << M[i][1] << " " << M[i][2] << " " << M[i][3] << std::endl;
        }
        std::cout << "\n cond rhs : " << std::endl;
        std::cout << rhs[0] << " " << rhs[1] << " " << rhs[2] << " " << rhs[3]<< std::endl;
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

//    for (int k = nb_of_line-1; k > new_nb-1 ; --k) {
//        for (int j = 0; j < k; ++j) {
//            M(0,j)-=M(0,k)*M(k,j)/M(k,k);
//        }
//        rhs[0]-=rhs[k]*M(0,k)/M(k,k);
//    }

//    for (unsigned int j = nb_of_line-1; j > new_nb-1 ; --j){
//         for (unsigned int i = 0; i < new_nb; ++i) {
//            M(0,i) -= M(0,j)*M(j,i)/M(j,j);

//         }
//         rhs[0] -= M(0,j)/M(j,j)*rhs[j];
//    }

//
//    std::cout << "\n cond mat avant suppression : " << std::endl;
//    for (unsigned int i = 0; i < nb_of_line; ++i) {
//        std::cout << M[i][0] << " " << M[i][1] << " " << M[i][2] << " " << M[i][3] << std::endl;
//    }
//    std::cout << "\n cond rhs avant suppression : " << std::endl;
//    std::cout << rhs[0] << " " << rhs[1] << " " << rhs[2] << " " << rhs[3] << std::endl;
//

    for (unsigned int i = 0; i < new_nb; ++i) {
        for (unsigned int j = 0; j < new_nb; ++j) {
            new_mat[i][j]=M[i][j];
        }
        new_rhs[i]=rhs[i];
    }
}
