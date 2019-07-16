#include "deal.II/base/point.h"
#include "deal.II/lac/vector.h"
#include "deal.II/lac/full_matrix.h"
#include "jacobian.h"

using namespace dealii;

//Tools for integration and finite elements in triangles, in order to solve the NS equation.



template <int dim>
double interptrg1D(int num_vertex, Point<dim> pt_eval)
{
    //evaluates the value of the 1D shape function linked to the vertex "num_vertex", at the point "pt_eval"

    double value;

    if (num_vertex==0) // scalar fct interp equals 1-x-y(-z if dim =3)
    {
        value=1;
        for (int j=0;j<dim;j++) {
            value-=pt_eval(j);
        }
    }

    else { // num_vertex > 0
        value = pt_eval(num_vertex-1);
    }
    return value;
}


void grad_interptrg1D(int num_vertex, Vector<double> &grad_return, int dim)
{
    // returns in grad_return the value of the gradient of the shape function associated to the vertex num_elem

    grad_return.reinit(dim);
    if (num_vertex==0) // scalar fct interp equals 1-x-y(-z if dim =3)
    {
        for (int i = 0; i < dim; ++i) {
            grad_return(i)=-1;
        }
    }

    else { // num_vertex > 0
       grad_return(num_vertex-1)=1;
    }
}



template<int dim>
void interp_trg_multidim(int num_vertex, Point<dim> pt_eval, Vector<double> &return_vec)
{
    // function for a triangle or a tetraedron
    // evaluates the value of the interpolation function associated to the vertex number "num_vertex"
    // at the point "pt_eval", which must be calculated to be in the elementary triangle/tetraedron
    // this means : let Tk be the transformation leading from the elementary triangle/tetraedron,
    // we have pt_eval = Tk^(-1) (actual_pt_in_real_element)


    //depending on the dimension of the problem

    // -> returns it in "return_vec"
    double value;

    if (num_vertex==0) // scalar fct interp equals 1-x-y(-z if dim =3)
    {
        value=1;
        for (int j=0;j<dim;j++) {
            value-=pt_eval(j);
        }
    }

    else { // num_vertex > 0
        value = pt_eval(num_vertex-1);
    }

    for (int j = 0; j < dim+1; ++j) {
        return_vec(j)=value;
    }
}



template<int dim>
void grad_interp_trg(int num_vertex, FullMatrix<double> &return_grad)
{
    // function for a triangle or a tetraedron
    // evaluates the gradient of the interpolation function associated to the vertex number "num_vertex"
    // it is a constant so it does not depend on the point of evaluation

    // depending on the dimension of the problem

    // -> returns it in "return_grad", which is a matrix of dimension (dim+1)x(dim+1)
    //dim+1 because to solve NS, we have dim+1 degrees of freedom (speed and pressure).

    //

    if (num_vertex==0)
    {
        for (int i = 0; i < dim+1; ++i) {
            for (int j = 0; j < dim+1; ++j) {
                return_grad(i,j)=-1;
            }
        }
    }

    else {
        for (int i = 0; i < dim+1; ++i) {
            return_grad(num_vertex-1,i)=1;
        }
    }
}



template<int dim, int dimension> // dimension here is the number of lines of the vector we want to interpolate in the triangle,
//for each line, you have to provide the valueon the vertices of the function to interpolate
//the values must be sorted as you sorted the vertices of the element

void interpolate_in_trg(Point<dim> pt_eval, Tensor<dimension, dim+1> values, Vector<double> &values_return)
{
    values_return.reinit(dimension);

    // we will suppose that the coordinates of pt_eval are given for the reference element

    double val;
    for (int i = 0; i < dimension; ++i) { // there are 3 vertices if we are in 2D and 4 if we are in 3D

        val = 0;

        for (int j = 0; j < dim+1; ++j) {
            val += interptrg1D(j,pt_eval)*values(i,j);
        }

        values_return(i) = val;
    }
}


template <int dim>
double size_el(Vector<Point<dim>> coor_elem)
{
    return std::sqrt(jacobian(0,0.,0.,coor_elem));
}



