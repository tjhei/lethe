#include "deal.II/base/point.h"
#include "deal.II/lac/vector.h"
#include "deal.II/lac/full_matrix.h"

using namespace dealii;

template<int dim>
void interp_trg(int num_vertex, Point<dim> pt_eval, Vector<double> &return_vec)
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

template<int dim> // dimension here is the dimension of the vector we want to interpolate in the triangle
void interpolate_in_trg(Vector<double> vertices_coor, Point<dim> pt_eval, Vector<double> values)
{
    int dimension = vertices_coor.size();

}


