#include "deal.II/base/point.h"
#include "deal.II/lac/vector.h"
#include "deal.II/lac/full_matrix.h"
#include "jacobian.h"

using namespace dealii;

//Tools for integration and finite elements in triangles, in order to solve the NS equation.


// Shape functions //

template <int dim>
double interp_pressure(int num_vertex, Point<dim> pt_eval)
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


void grad_interp_pressure(int num_vertex, Vector<double> &grad_return, int dim)
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
void interp_trg_velocity(int num_vertex, Point<dim> pt_eval, Vector<double> &return_vec)
{
    // function for a triangle or a tetraedron
    // evaluates the value of the interpolation function associated to the vertex number "num_vertex"
    // at the point "pt_eval", which must be calculated to be in the elementary triangle/tetraedron
    // this means : let Tk be the transformation leading from the elementary triangle/tetraedron to the considered element,
    // we have pt_eval = Tk^(-1) (actual_pt_in_real_element) (just apply the function change coor to get the corresponding coordinates in the reference element)


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
void grad_interp_velocity(int num_vertex, FullMatrix<double> &return_grad)
{
    // function for a triangle or a tetraedron
    // evaluates the gradient of the interpolation function associated to the vertex number "num_vertex"
    // it is a constant so it does not depend on the point of evaluation

    // depending on the dimension of the problem

    // -> returns it in "return_grad", which is a matrix of dimension (dim)x(dim)

    if (num_vertex==0)
    {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                return_grad(i,j)=-1;
            }
        }
    }

    else {
        for (int i = 0; i < dim; ++i) {
            return_grad(num_vertex-1,i)=1;
        }
    }
}

double div_phi_u(int num_vertex, int u_v_w)
{
    // returns the value d(phi_u_{num_vertex})/d(x_{u_v_w})

    if (num_vertex==0)
    {
        return -1.;
    }
    else if (num_vertex==1) {
        if (u_v_w==0)
            return 1.;
        else {
            return 0.;
        }
    }
    else if (num_vertex==2) {
        if (u_v_w==1)
            return 1.;
        else {
            return 0.;
        }
    }
    else {
        if (u_v_w==2)
            return 1.;
        else {
            return 0.;
        }
    }
}

// end of the shape functions //


// Functions to interpolate velocity, pressure and their gradients //

template<int dim>
double interpolate_pressure(Point<dim> pt_eval, Vector<double> pressure_node)
{
    // given the value of the pressure on each vertex of the element, and given a point "pt_eval" in the reference element
    // (if you don't have the corresponding coordinates in the ref element, just apply "change_coor" to the coordinates of the point in the element)
    // returns the interpolated value of the pressure at the point "pt_eval"

    double value = 0;
    for (int i = 0; i < dim+1; ++i) {
        value += pressure_node(i)*interp_pressure(i, pt_eval);
    }
    return value;

}


template<int dim> // dimension here is the number of lines of the vector we want to interpolate in the triangle,
//for each line, you have to provide the valueon the vertices of the function to interpolate
//the values must be sorted as you sorted the vertices of the element

void interpolate_velocity(Point<dim> pt_eval, Vector<Tensor<1,dim>> values, Vector<double> &values_return)
{
    // interpolates the vector (velocity_x, velocity_y (, velocity_z) )
    // depending on the values on each vertex of the triangle.
    // the tensor of index i in values is the velocity on the vertex of index i

    values_return.reinit(dim);

    // we will suppose that the coordinates of pt_eval are given for the reference element

    for (int i = 0; i < dim+1 ; ++i) { // there are 3 vertices if we are in 2D and 4 if we are in 3D
        for (int j = 0; j < dim; ++j) { // j stands for the component of the speed we interpolate
            values_return(j) += interp_pressure(i,pt_eval)*values(i,j);
        }
    }
}


template <int dim>
void interpolate_grad_pressure(Point<dim>  pt_eval, Vector<Tensor<1,dim>>  values_grad, Tensor<1,dim>  &grad_return)
{
    // each tensor in values_grad is the pressure gradient given for one of the vertices
    // this function returns in "grad_return" the value of the interpolated pressure gradient at "pt_eval"

    Vector<double>      vec_grad_p;


    for (int i = 0; i < dim; ++i) {

        // i here allows us to know which coordinate of the gradient we are interpolating

        vec_grad_p.reinit(dim+1);
        for (int j = 0; j < dim+1 ; ++j) {
            vec_grad_p(j) = values_grad(j,i);
        }

        grad_return(i) = interpolate_pressure(pt_eval, vec_grad_p);
    }
}


template <int dim>
void interpolate_grad_velocity(Point<dim>  pt_eval, Vector<Tensor<2,dim>>  values_grad, Tensor<2,dim>  &grad_return)
{
    // each tensor in values_grad is the velocity gradient given for one of the vertices
    // this function returns in "grad_return" the value of the interpolated velocity gradient at "pt_eval"

    Vector<Tensor<1,dim>>   grad_u_i; /* we will store in this vector the value of the gradients associated to one component of the
                                      velocity, on each vertex of the element */
    grad_u_i.reinit(dim+1);


    Tensor<1,dim> temp;

    for (int j = 0; j < dim; ++j) { // j stands for the component of the velocity we interpolate, grad(u_j)

        for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex considered
            grad_u_i(i) = values_grad(i,j);
        }

        temp =0;
        interpolate_grad_pressure(pt_eval, grad_u_i, temp);

        grad_return(j) = temp;
    }
}

// end of interpolating functions //


// Other tools //

template <int dim>
double size_el(Vector<Point<dim>> coor_elem)
{
    return std::sqrt(jacobian(0,0.,0.,coor_elem));
}


template <int dim>
void change_coor(Point<dim> pt_elem, Point<dim> &pt_ref, Vector<Point<dim>> coor_elem)
{
    // returns the corresponding coordinates in the reference element of the point "pt_elem"
    // works in 2D or in 3D

    double size = size_el(coor_elem);

    double x,y,x1,y1,x2,y2,x0,y0;

    x0=coor_elem(0,0);
    y0=coor_elem(0,1);

    x = pt_elem(0)-x0;
    y = pt_elem(1)-y0;

    x1=coor_elem(1,0)-x0;
    y1=coor_elem(1,1)-y0;

    x2=coor_elem(2,0)-x0;
    y2=coor_elem(2,1)-y0;

    if (dim==2)
    {
        if (std::abs(x1)>size*1e-4)
        {
            pt_ref(1) = (y1/x1*x-y)/(x2/x1*y1-y2);
            // x2/x1*y1-y2 != 0,
            // else we would have y1/x1 = y2/x2 which is not possible for a triangle
            //(a bit of geometry allows you to see it easily)

            pt_ref(0) = (x-x2*pt_ref(1))/x1;
        }

        else { // if x1 = 0 then x2 != 0 since we have a triangle
            pt_ref(1) = x/x2;
            pt_ref(0) = (y-pt_ref(1)*y2)/y1;
        }
    }

    else {
        double z0,z1,z2,x3,y3,z3,z;


        z0 = coor_elem(0,2);

        z = pt_elem(2)-z0;

        z1 = coor_elem(1,2)-z0;
        z2 = coor_elem(2,2)-z0;

        x3 = coor_elem(3,0)-x0;
        y3 = coor_elem(3,1)-y0;
        z3 = coor_elem(3,2)-z0;

        FullMatrix<double> A(3,3);
        A(0,0) = y2*z3 - z2*y3;
        A(1,0) = (y3)*(z1) - (z3)*(y1);
        A(2,0) = (y1)*z2 - (z1)*(y2);

        double det = (x1)*A(0,0) + (x2)*A(1,0) + (x3)*A(2,0);

        A(0,1) = (x3)*(z2) - (z3)*(x2);
        A(1,1) = (x1)*(z3) - (z1)*(x3);
        A(2,1) = (x2)*(z1) - (z2)*(x1);

        A(0,2) = (x2)*(y3) - (y2)*(x3);
        A(1,2) = (x3)*(y1) - (y3)*(z1);
        A(2,2) = (x1)*(y2) - (y1)*(x2);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A(i,j) = A(i,j)/det;
            }
        }

        pt_ref(0) = (x)*A(0,0) + (y)*A(0,1) + (z)*A(0,2);
        pt_ref(1) = (x)*A(1,0) + (y)*A(1,1) + (z)*A(1,2);
        pt_ref(2) = (x)*A(2,0) + (y)*A(2,1) + (z)*A(2,2);

    }
}
