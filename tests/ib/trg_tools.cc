#include "trg_tools_class.h"
#include <vector>
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"

using namespace dealii;

void test_interpolate()//initializing a TRG_tools object in order to test its interpolation functions
{
    //The point of this function is to check that the functions of interpolation of the TRG_tools objects work properly

    // first test //

    std::cout << " First test of interpolation " << std::endl;

    // creating a triangle
    std::vector<Point<2>>       coor_trg(3);
    Point<2>                    pt0(0,0);
    Point<2>                    pt1(1,0);
    Point<2>                    pt2(0,1);

    coor_trg[(0)] = pt0;
    coor_trg[(1)] = pt1;
    coor_trg[(2)] = pt2;

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
    unsigned int dofs_per_node = 3;

    //initializing a TRG_tools object in order to test its interpolation functions
    TRG_tools<2>    trg_(dofs_per_node, pressure, veloce, coor_trg);

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
    std::cout << "Error on interpolated pressure : " << std::abs(p_th-interpolated_p) << std::endl;
    for (int i = 0; i < 2; ++i) {
        if (std::abs(veloce_th[i]-interpolated_v[i])>1e-10) throw std::runtime_error("Failed to build interpolated v");
        std::cout << "Error on component " << i << " of interpolated velocity : " << std::abs(veloce_th[i]-interpolated_v[i]) << std::endl;
    }
    for (int i = 0; i < 2; ++i) {
        if (std::abs(grad_p_th[i]-interpolated_grad_p[i])>1e-10) throw std::runtime_error("Failed to build interpolated grad p");
        std::cout << "Error on component " << i << " of interpolated gradient of pressure : " << std::abs(grad_p_th[i]-interpolated_grad_p[i]) << std::endl;
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (std::abs(grad_v_th[i][j]-interpolated_grad_v[i][j])>1e-10) throw std::runtime_error("Failed to build interpolated grad v");
            std::cout << "Error on component " << i << "," << j << " of interpolated gradient of velocity : " << std::abs(grad_v_th[i][j]-interpolated_grad_v[i][j]) << std::endl;
        }
    }

    // Second test //

    std::cout << " Second test of interpolation " << std::endl;

    veloce[0][0] = 0.5;
    veloce[0][1] = -0.5;

    veloce[1][0] = 0.2;
    veloce[1][1] = -0.4;

    veloce[2][0] = -0.2;
    veloce[2][1] = 0.1;

    pressure[0] = 1;
    pressure[1] = 0.3;
    pressure[2] = 0.2;


    //initializing a TRG_tools object in order to test its interpolation functions
    TRG_tools<2>    trg_1(dofs_per_node, pressure, veloce, coor_trg);

    Point<2>                evaluation_pt1(1./5.,3./5.);

    interpolated_p = trg_1.interpolate_pressure(evaluation_pt1);
    trg_1.interpolate_velocity(evaluation_pt1, interpolated_v);
    trg_1.interpolate_grad_pressure(interpolated_grad_p);
    trg_1.interpolate_grad_velocity(interpolated_grad_v);

    // values we should obtain
    p_th =0.38;

    veloce_th[0] =0.02;
    veloce_th[1] =-0.12;

    grad_p_th[0] =-0.7;
    grad_p_th[1] = -0.8;

    grad_v_th=0;

    grad_v_th[0][0]=-0.3;
    grad_v_th[0][1]=-0.7;

    grad_v_th[1][0]=0.1;
    grad_v_th[1][1]=0.6;

    if (std::abs(p_th-interpolated_p)>1e-10) throw std::runtime_error("Failed to build interpolated p");
    std::cout << "Error on interpolated pressure : " << std::abs(p_th-interpolated_p) << std::endl;
    for (int i = 0; i < 2; ++i) {
        if (std::abs(veloce_th[i]-interpolated_v[i])>1e-10) throw std::runtime_error("Failed to build interpolated v");
        std::cout << "Error on component " << i << " of interpolated velocity : " << std::abs(veloce_th[i]-interpolated_v[i]) << std::endl;
    }
    for (int i = 0; i < 2; ++i) {
        if (std::abs(grad_p_th[i]-interpolated_grad_p[i])>1e-10) throw std::runtime_error("Failed to build interpolated grad p");
        std::cout << "Error on component " << i << " of interpolated gradient of pressure : " << std::abs(grad_p_th[i]-interpolated_grad_p[i]) << std::endl;
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (std::abs(grad_v_th[i][j]-interpolated_grad_v[i][j])>1e-10) throw std::runtime_error("Failed to build interpolated grad v");
            std::cout << "Error on component " << i << "," << j << " of interpolated gradient of velocity : " << std::abs(grad_v_th[i][j]-interpolated_grad_v[i][j]) << std::endl;
        }
    }

    std::cout << " Tests of interpolation passed successfully" << std::endl;
}

void test_shape_build()
{
    //The point of this function is to check that the functions of interpolation of the TRG_tools objects work properly

    // first test //

    std::cout << " Test on shape functions built with a TRG_tools object " << std::endl;

    // creating a triangle
    std::vector<Point<2>>       coor_trg(3);
    Point<2>                    pt0(0,0);
    Point<2>                    pt1(1,0);
    Point<2>                    pt2(0,1);

    coor_trg[(0)] = pt0;
    coor_trg[(1)] = pt1;
    coor_trg[(2)] = pt2;

    // number of vertices of the triangle
    unsigned int dofs_per_node = 3;

    //initializing a TRG_tools object in order to test its interpolation functions
    TRG_tools<2>    trg_;
    trg_.set_coor_trg(coor_trg);
    trg_.set_dofs_per_node(dofs_per_node);

    // Creating vectors to store the shape function vectors
    unsigned int dofs_per_trg = 3 * dofs_per_node;

    std::vector<double>         div_phi_u_(dofs_per_trg);
    std::vector<Tensor<1,2>>    phi_u(dofs_per_trg);
    std::vector<Tensor<2,2>>    grad_phi_u(dofs_per_trg);
    std::vector<double>         phi_p(dofs_per_trg);
    std::vector<Tensor<1,2>>    grad_phi_p(dofs_per_trg);

    // Point of evaluation
    Point<2>                    Pt_eval(0.2, 0.6);

    // Creating a passage matrix even if we are already in the ref triangle, it will be the Id matrix

    Tensor<2,2>                 Pass_mat;
    Pass_mat =0;
    trg_.matrix_pass_elem_to_ref(Pass_mat);

    if ((std::abs(Pass_mat[0][0]-1)>1e-10) || (std::abs(Pass_mat[1][1]-1)>1e-10)) throw std::runtime_error("Failed to build the passage matrix");

    double err = sqrt(std::pow(Pass_mat[0][0]-1>1e-10,2) + std::pow(Pass_mat[1][1]-1>1e-10,2));
    std::cout << "Error on the coefficients of the passage matrix : " << err << std::endl;

    // Building the vectors
    trg_.build_phi_p(Pt_eval, phi_p);
    trg_.build_phi_u(Pt_eval, phi_u);
    trg_.build_div_phi_u(Pass_mat, div_phi_u_);
    trg_.build_grad_phi_p(Pass_mat, grad_phi_p);
    trg_.build_grad_phi_u(Pass_mat, grad_phi_u);

    // Building the vectors we should obtain
    // We use trg_.shape_function() cause it is working for sure

    std::vector<Tensor<1,2>>    grad_shape_th(dofs_per_node);
    grad_shape_th[0][0]=-1;
    grad_shape_th[0][1]=-1;
    grad_shape_th[1][0]=1;
    grad_shape_th[1][1]=0;
    grad_shape_th[2][0]=0;
    grad_shape_th[2][1]=1;

    std::vector<double>         div_phi_u_th(dofs_per_trg);
    std::vector<Tensor<1,2>>    phi_u_th(dofs_per_trg);
    std::vector<Tensor<2,2>>    grad_phi_u_th(dofs_per_trg);
    std::vector<double>         phi_p_th(dofs_per_trg);
    std::vector<Tensor<1,2>>    grad_phi_p_th(dofs_per_trg);

    for (unsigned int i = 0; i < dofs_per_node; ++i) {
        phi_p_th[3*i] =0;
        phi_p_th[3*i+1] =0;
        phi_p_th[3*i+2] =trg_.shape_function(i, Pt_eval);

        phi_u_th[3*i][0] = trg_.shape_function(i, Pt_eval);
        phi_u_th[3*i][1] = 0;
        phi_u_th[3*i+1][0] = 0;
        phi_u_th[3*i+1][1] = trg_.shape_function(i, Pt_eval);
        phi_u_th[3*i+2][0] = 0;
        phi_u_th[3*i+2][1] = 0;

        grad_phi_p_th[3*i+2] = grad_shape_th[i];

        grad_phi_u_th[3*i][0][0] = grad_shape_th[i][0];
        grad_phi_u_th[3*i][0][1] = grad_shape_th[i][1];
        grad_phi_u_th[3*i+1][1][0] = grad_shape_th[i][0];
        grad_phi_u_th[3*i+1][1][1] = grad_shape_th[i][1];
        grad_phi_u_th[3*i+2] = 0;

        div_phi_u_th[3*i] = div_shape_velocity_ref(i, 0, 2); //trg_.divergence(i, 0, Pass_mat);
        div_phi_u_th[3*i+1] = trg_.divergence(i, 1, Pass_mat);
        div_phi_u_th[3*i+2] = 0 ;
    }

    double err_p, err_grad_p, err_v, err_grad_v, err_div;
    err_p =0;
    err_grad_p=0;
    err_v = 0;
    err_grad_v=0;
    err_div=0;

    for (unsigned int i = 0; i < dofs_per_trg; ++i) {
        if (std::abs(phi_p[i] - phi_p_th[i])>1e-10) throw std::runtime_error("Failed to build the shape function vector for p");
        err_p += std::pow(phi_p[i] - phi_p_th[i], 2);

        if (std::abs(div_phi_u_[i] - div_phi_u_th[i])>1e-10) throw std::runtime_error("Failed to build the shape function vector for div u");
        err_div += std::pow(div_phi_u_[i] - div_phi_u_th[i],2);

        for (int j = 0; j < 2; ++j) {
            if (std::abs(grad_phi_p[i][j] - grad_phi_p_th[i][j])>1e-10) throw std::runtime_error("Failed to build the gradient of the shape function vector for p");
            err_grad_p += std::pow(grad_phi_p[i][j] - grad_phi_p_th[i][j],2);

            if (std::abs(phi_u[i][j] - phi_u_th[i][j])>1e-10) throw std::runtime_error("Failed to build the shape function vector for v");
            err_v += std::pow(phi_u[i][j] - phi_u_th[i][j],2);

            for (int k = 0; k < 2; k ++) {
                if (std::abs(grad_phi_u[i][j][k] - grad_phi_u_th[i][j][k])>1e-10) throw std::runtime_error("Failed to build the gradient of the shape function vector for v");
                err_grad_v += std::pow(grad_phi_u[i][j][k] - grad_phi_u_th[i][j][k],2);
            }
        }
    }

    std::cout << "Error on the shape function vector for p : " << sqrt(err_p) << std::endl;
    std::cout << "Error on the shape function vector for v : " << sqrt(err_v) << std::endl;
    std::cout << "Error on the shape function vector for div(v) : " << sqrt(err_div) << std::endl;
    std::cout << "Error on the gradient of the shape function vector for p : " << sqrt(err_grad_p) << std::endl;
    std::cout << "Error on the gradient of the shape function vector for v : " << sqrt(err_grad_v) << std::endl;

    std::cout << "All tests on shape functions passed succesfully" << std::endl;
}

int main()
{
  try
  {
    test_interpolate();
    test_shape_build();
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
