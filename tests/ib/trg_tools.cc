#include "trg_tools_class.h"
#include <vector>
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"

using namespace dealii;

void test_interpolate()//initializing a TRG_tools object in order to test its interpolation functions
{
    //The point of this function is to check that the functions of interpolation of the TRG_tools objects work properly

    // creating a triangle
    std::vector<Point<2>>       coor_trg(3);
    Point<2>                    pt0(0,0);
    Point<2>                    pt1(1,0);
    Point<2>                    pt2(0,1);

    // creating a vector containing values of velocity on the vertices of the trg
    std::vector<Tensor<1, 2>>   veloce(3);

    veloce[0][0] = 0;
    veloce[0][1] = 0;

    veloce[1][0] = 1;
    veloce[1][1] = 0;

    veloce[2][0] = 0;
    veloce[2][1] = 1;

    // creating a vector containing the values of the pressure on the vertices of the trg
    std::vector<double>         pressure(3);
    pressure[0] = 1;
    pressure[1] = -1;
    pressure[2] = 0;

    // number of vertices of the triangle
    unsigned int nb_of_vertices = 3;

    //initializing a TRG_tools object in order to test its interpolation functions
    TRG_tools<2>    trg_(nb_of_vertices, pressure, veloce, coor_trg);

    // vectors for the interpolated variables
    Tensor<1,2>             interpolated_v;
    double                  interpolated_p = 0;
    Tensor<1,2>             interpolated_grad_p;
    Tensor<2,2>             interpolated_grad_v;

    Point<2>                evaluation_pt(0.5,0.5);

    interpolated_p = trg_.interpolate_pressure(evaluation_pt);
    trg_.interpolate_velocity(evaluation_pt, interpolated_v);
    trg_.interpolate_grad_pressure(interpolated_grad_p);
    trg_.interpolate_grad_velocity(interpolated_grad_v);

    // values we should obtain
    double p_th =-0.5;

    Tensor<1,2> veloce_th;
    veloce_th[0] =0.5;
    veloce_th[1] =0.5;

    Tensor<1,2> grad_p_th;
    grad_p_th[0] =-2;
    grad_p_th[1] = -1;

    Tensor<2,2> grad_v_th;
    grad_v_th=0;
    grad_v_th[0][0]=1;
    grad_v_th[1][1]=1;

    if (std::abs(p_th-interpolated_p)>1e-10) throw std::runtime_error("Failed to build interpolated p");

    for (int i = 0; i < 2; ++i) {
        if (std::abs(veloce_th[i]-interpolated_v[i])>1e-10) throw std::runtime_error("Failed to build interpolated v");
        if (std::abs(grad_p_th[i]-interpolated_grad_p[i])>1e-10) throw std::runtime_error("Failed to build interpolated grad p");
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (std::abs(grad_v_th[i][j]-interpolated_grad_v[i][j])>1e-10) throw std::runtime_error("Failed to build interpolated grad v");
        }
    }
    std::cout << " Tests of interpolation passed successfully" << std::endl;
}

int main()
{
  try
  {
    test_interpolate();
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
