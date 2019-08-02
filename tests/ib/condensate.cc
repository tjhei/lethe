#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"

#include "condensate.h"

using namespace dealii;

void test_condensate()
{
    FullMatrix<double>      M(4,4);
    FullMatrix<double>      new_m(2,2);
    Vector<double>     rhs(4);
    Vector<double>     new_rhs(2);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            M(i,j)=std::pow(-1,i+j);
        }
        rhs[i]=0;
    }

    M(2,0)=-1;
    M(3,2)=1;
    rhs[3]=1;

    condensate(4,2,M,new_m,rhs,new_rhs);

    if (std::abs(new_rhs[0])>1e-10) throw std::runtime_error("Failed to build the condensated RHS 0");
    if (std::abs(new_rhs[1])>1e-10) throw std::runtime_error("Failed to build the condensated RHS 1");
    if (std::abs(new_m(0,0)-2)>1e-10) throw std::runtime_error("Failed to build the condensated matrix (0,0)");
    if (std::abs(new_m(0,1))>1e-10) throw std::runtime_error("Failed to build the condensated matrix (0,1)");
    if (std::abs(new_m(1,0)+2)>1e-10) throw std::runtime_error("Failed to build the condensated matrix (1,0)");
    if (std::abs(new_m(1,1))>1e-10) throw std::runtime_error("Failed to build the condensated matrix (1,1)");

}

int main()
{
    try
    {
      test_condensate();
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what()  << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
    catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

    return 0;
}
