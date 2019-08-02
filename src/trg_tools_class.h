/* -------------------------------------------------------------------------
 * trg_tools_class.h
 *
 * This class gives all the tools necessary to build the shape functions and
 * the vectors in order to integrate the Navier Stokes equations in a
 * triangle element (hence in 2D). Most functions will yet work in 3D, not
 * all though, that's why we talk about triangle elements and not simplexes
 *
 * -------------------------------------------------------------------------
 */


#include "deal.II/base/point.h"
#include <vector>
#include "deal.II/base/tensor.h"
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/full_matrix.h"
#include <deal.II/base/function.h>

#include "jacobian.h"

using namespace dealii;

template <int dim>
class TRG_tools: public Function<dim>
{
public:

    TRG_tools(unsigned int dofs_per_node_, std::vector<double> p_on_vertices, std::vector<Tensor<1,dim>>  v_on_vertices, std::vector<Point<2> > coor_vertices_trg) :
        Function<dim>(1),
        dofs_per_node(dofs_per_node_),
        P_node(p_on_vertices),
        V_node(v_on_vertices),
        coor_vertices_trg_(coor_vertices_trg)
      {}

    TRG_tools() :
        Function<dim>(1)
      {}

    virtual double shape_function(const int num_vertex, const Point<dim> p_eval) const;
    void grad_shape_function(const int num_vertex, Tensor<1,dim> &grad_return, const Tensor<2, dim> pass_matrix);

    void matrix_pass_elem_to_ref(const std::vector<Point<2>> coor_trg, Tensor<2, dim> &pass_matrix);
    double size_el(const std::vector<Point<dim>> coor_elem);
    double divergence(const int num_vertex, const int component, const Tensor<2, dim> pass_matrix);

    double interpolate_pressure(const Point<dim> pt_eval);
    void interpolate_velocity(const Point<dim> pt_eval, Tensor<1,dim> &values_return);
    void interpolate_grad_pressure(Tensor<1,dim>  &grad_return);
    void interpolate_grad_velocity(Tensor<2,dim>  &grad_return);

    void build_phi_u        (const Point<dim> pt_quad_eval, std::vector<Tensor<1, dim>>         &phi_u      );
    void build_phi_p        (const Point<dim> pt_quad_eval, std::vector<double>                 &phi_p      );
    void build_div_phi_u    (const Tensor<2,dim> passage_matrix, std::vector<double>            &div_phi_u_ );
    void build_grad_phi_u   (const Tensor<2,dim> passage_matrix, std::vector<Tensor<2, dim>>    &grad_phi_u );
    void build_grad_phi_p   (const Tensor<2,dim> passage_matrix, std::vector<Tensor<1, dim>>    &grad_phi_p );

private:
    double partial_coor_ref_2D(const int component, const int j_partial, const std::vector<Point<2>> coor_trg);
    std::vector<Tensor<1,dim>>  P_node;
    std::vector<double>         V_node;
    std::vector<Point<2> > coor_vertices_trg_;
    unsigned int dofs_per_node;
};

// functions to build the vectors of shape functions for u and p //
// at the moment it works for 2D only !!

template <int dim>
void TRG_tools<dim>::build_phi_u(const Point<dim> pt_quad_eval, std::vector<Tensor<1, dim>> &phi_u)
{

    for (int i = 0; i < dim+1; ++i) {
        phi_u[dofs_per_node*i][0] = shape_function(i, pt_quad_eval);
        phi_u[dofs_per_node*i][1] = 0;

        phi_u[dofs_per_node*i+1][0] = 0;
        phi_u[dofs_per_node*i+1][1] = shape_function(i, pt_quad_eval);

        phi_u[dofs_per_node*i+2][0] = 0;
        phi_u[dofs_per_node*i+2][1] = 0;
    }
}

template <int dim>
void TRG_tools<dim>::build_phi_p(const Point<dim> pt_quad_eval, std::vector<double> &phi_p)
{
    for (int i = 0; i < dim+1; ++i) {
        phi_p[dofs_per_node*i+dim] = shape_function(i, pt_quad_eval);
        phi_p[dofs_per_node*i] =0;
        phi_p[dofs_per_node*i+1]=0;
    }
}

template <int dim>
void TRG_tools<dim>::build_div_phi_u (const Tensor<2,dim> passage_matrix, std::vector<double> &div_phi_u_)
{
    for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex
        for (int j = 0; j < dim; ++j) {
            div_phi_u_[dofs_per_node*i+j] = divergence(i, j, passage_matrix); // We apply the passage matrix
        }
        div_phi_u_[dofs_per_node*i +dim] = 0;
    }
}

template <int dim>
void TRG_tools<dim>::build_grad_phi_u(const Tensor<2,dim> passage_matrix, std::vector<Tensor<2, dim>> &grad_phi_u)
{
    Tensor<2, dim> e1_x_e1;     Tensor<2, dim> e2_x_e2;     Tensor<2, dim> e1_x_e2;     Tensor<2, dim> e2_x_e1;
    e1_x_e1=0;                  e2_x_e2=0;                  e1_x_e2=0;                  e2_x_e1=0;
    e1_x_e1[0][0] =1;           e2_x_e2[1][1] =1;           e1_x_e2[0][1] =1;           e2_x_e1[1][0]=1;

    for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex
            grad_phi_u[dofs_per_node*i] = divergence(i, 0, passage_matrix) * e1_x_e1 + divergence(i, 1, passage_matrix) * e1_x_e2;
            grad_phi_u[dofs_per_node*i+1] = divergence(i, 0, passage_matrix) * e2_x_e1 + divergence(i, 1, passage_matrix) * e2_x_e2;
            grad_phi_u[dofs_per_node*i+2] = 0;
    }
}

template <int dim>
void TRG_tools<dim>::build_grad_phi_p(const Tensor<2,dim> passage_matrix, std::vector<Tensor<1, dim>> &grad_phi_p)
{
    for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex
        grad_phi_p[dofs_per_node*i] =0;
        grad_phi_p[dofs_per_node*i+1] =0;
        grad_shape_function(i, grad_phi_p[dofs_per_node*i+2], passage_matrix);
    }
}

//              //




// Define the shape and grad_shape functions //

template <int dim>
double TRG_tools<dim>::shape_function(const int num_vertex, const Point<dim> pt_eval) const
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


template <int dim>
void TRG_tools<dim>::grad_shape_function(const int num_vertex, Tensor<1, dim> &grad_return, const Tensor<2, dim> pass_matrix)
{
    // returns in grad_return the value of the gradient of the shape function associated to the vertex num_elem
    // pass_mat is the passage matrix, we apply it to change the coordinates

    grad_return =0;
    Tensor<1, dim> temp;
    temp=0;

    if (num_vertex==0) // scalar fct interp equals 1-x-y(-z if dim =3)
    {
        for (int i = 0; i < dim; ++i) {
            temp[(i)]=-1;
        }
    }

    else { // num_vertex > 0
       temp[(num_vertex-1)]=1;
    }
    grad_return = pass_matrix * temp;
}

//          //




// functions of interpolation for velocity and pressure //

template<int dim>
double TRG_tools<dim>::interpolate_pressure(Point<dim> pt_eval)
{
    // given the value of the pressure on each vertex of the element, and given a point "pt_eval" in the reference element
    // (if you don't have the corresponding coordinates in the ref element, just apply "change_coor" to the coordinates of the point in the element)
    // returns the interpolated value of the pressure at the point "pt_eval"

    double value = 0;
    for (int i = 0; i < dim+1; ++i) {
        value += P_node[i]*shape_function(i, pt_eval);
    }
    return value;

}


template<int dim>

void TRG_tools<dim>::interpolate_velocity(const Point<dim> pt_eval, Tensor<1,dim> &values_return)
{
    // interpolates the vector (velocity_x, velocity_y (, velocity_z) )
    // depending on the values on each vertex of the triangle.
    // the tensor of index i in "values" is the velocity on the vertex of index i

    // we will suppose that  pt_eval is given in the reference element

    for (int i = 0; i < dim+1 ; ++i) { // there are 3 vertices if we are in 2D and 4 if we are in 3D
        for (int j = 0; j < dim; ++j) { // j stands for the component of the speed we interpolate
            values_return[j] += shape_function(i,pt_eval)*V_node[i][j];
        }
    }
}


template <int dim>
void TRG_tools<dim>::interpolate_grad_pressure(Tensor<1,dim>  &grad_return)
{
    // we interpolate like this : p = p_i * Phi_i => grad(p) = p_i * grad(Phi_i) (a_i * b_i is a sum on i here)
    // this function returns in "grad_return" the value of the interpolated pressure gradient at "pt_eval"
    Tensor<1, dim>  grad_phi_i;
    Tensor<2, dim>  pass_mat;

    pass_mat=0;
    pass_mat[0][0]=1;
    pass_mat[1][1]=1;

    grad_return =0;

    for (int i = 0; i < dim; ++i) {
        grad_shape_function(i, grad_phi_i, pass_mat);
        // i is the index of the vertex
        grad_return[(0)] += grad_phi_i[0]*P_node[i];
        grad_return[(1)] += grad_phi_i[1]*P_node[i];
    }
}


template <int dim>
void TRG_tools<dim>::interpolate_grad_velocity(Tensor<2,dim>  &grad_return)
{
    // each tensor in values_grad is the velocity gradient given for one of the vertices
    // this function returns in "grad_return" the value of the interpolated velocity gradient at "pt_eval"

    std::vector<double>   v_i(dim+1); /* we will store in this vector the value of one component of the
                                      velocity, on each vertex of the element */

    Tensor<1, dim>        temp;

    for (int j = 0; j < dim; ++j) { // j stands for the component of the velocity we interpolate, grad(u_j)

        for (int i = 0; i < dim+1; ++i) { // i is the index of the vertex considered
            v_i[i] = V_node[i][(j)];
        }
        temp=0;
        interpolate_grad_pressure(v_i, temp);

        for (int i = 0; i < dim+1; ++i) {
        grad_return[i][j] = temp[(i)];
        }
    }

}

//          //



// Other functions that are useful //

template <int dim>
double TRG_tools<dim>::size_el(std::vector<Point<dim>> coor_elem)
{
    // calculates the "size" of a triangular element

    return std::sqrt(jacobian(0,0.,0.,coor_elem));
}

template <int dim>
double TRG_tools<dim>::partial_coor_ref_2D(const int component, const int j_partial, const std::vector<Point<2>> coor_trg)
{
    // calculates d(x_ref_component)/d(x_{j_partial}) in order to change of coordinates

    if (component == 0)
    {
        double jac = jacobian(0, 0. , 0., coor_trg);
        if (j_partial==component)
            return (coor_trg[2][1]-coor_trg[0][1])/jac;
        else {
            return (coor_trg[0][0]-coor_trg[2][0])/jac;
        }
    }

    else if (component == 1)
    {
        double jac = jacobian(0, 0. , 0., coor_trg);
        if (j_partial == component)
            return (coor_trg[(1)][(0)]-coor_trg[(0)][(0)])/jac;
        else {
            return (coor_trg[(0)][(1)]-coor_trg[(1)][(1)])/jac;
        }
    }
    else {
        return 0;
    }
}


template <int dim>
void TRG_tools<dim>::matrix_pass_elem_to_ref(const std::vector<Point<2>> coor_trg, Tensor<2, dim> &pass_matrix)
{
    // returns in "pass_matrix" the passage matrix from a triangle of coordinates "coor_trg" to the triangle of reference
    for (int i = 0; i < dim; ++i) {
        for (int var = 0; var < dim; ++var) {
            pass_matrix[i][var] = partial_coor_ref_2D(i,var, coor_trg);
        }
    }
}

double div_shape_velocity_ref(int num_vertex, int u_v_w, int dim)
{
    // returns the value d(phi_u_{num_vertex})/d(x_{u_v_w})

    if (u_v_w == dim)
        return 0; // we put 0 if u_v_w is associated to a pressure dof

    else if (num_vertex==0)
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


template <int dim>
double TRG_tools<dim>::divergence(int num_vertex, int component, Tensor<2, dim> pass_matrix)
{
    // returns the d(phi_{num_vertex})/dx_{component} in the ref element

    Tensor<1, dim>      div;
    for (int i = 0; i < dim; ++i) {
        div[(i)] = div_shape_velocity_ref(num_vertex, i, dim);
    }
    Tensor<1, dim>      temp;
    temp = pass_matrix*div; // changing the coordinates, from the actual element to the ref element
    return temp[(component)];
}




///
///
///
///
// Function to build a quadrature easily, precision : 3

template <int dim>
void get_Quadrature_trg(std::vector<Point<dim>> &quad_pt, std::vector<double> &weight)
{
    // creates the points and the weights of quadrature for a Hammer quad of precision 3
    // precision 3 means it perfectly integrates polynoms of degree 3 on the reference element

    Point<dim> pt0(1./3., 1./3.);   Point<dim> pt1(1./5., 1./5.);   Point<dim> pt2(1./5., 3./5.);   Point<dim> pt3(3./5., 1./5.);
    quad_pt[0] = pt0;               quad_pt[1] = pt1;               quad_pt[2] = pt2;               quad_pt[3] = pt3;
    weight[0] = -0.281250;          weight[1] = 0.264167;           weight[2] = 0.264167;           weight[3] = 0.264167;

}
