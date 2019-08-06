#include <vector>
#include "deal.II/lac/full_matrix.h"
#include "deal.II/base/tensor.h"
#include "deal.II/lac/vector.h"

#include "GLS_residual.h"
#include "ib_node_status.h"
#include "nouvtriangles.h"


using namespace dealii;

// We here check if we get the right evaluation for the elementary matrix and elementary rhs
// with GLS_residual, which will then be removed because no more used in the solver.
// Yet the code used in this file is pretty much the same as the function implemented in
// the class of the solver in ../../prototypes/IBSteadyNS/IBSteadyNS.cc

void test_residual()
{
    // building a "cell" and the distance to the boundary associated to each vertex
    std::vector<Point<2>>   coor(4);
    std::vector<double>     dist(4);

    // dist will be chosen so that the element is decomposed into one triangle (3 vertices in the solid, one in the fluid)

    dist[0] = 1.;
    dist[1] = -1.;
    dist[2] = -1.;
    dist[3] = -1.5;

    coor[0][0] = 0;
    coor[0][1] = 0;
    coor[1][0] = 1;
    coor[1][1] = 0;
    coor[2][0] = 0;
    coor[2][1] = 1;
    coor[3][0] = 1;
    coor[3][1] = 1;

    // Initializing tools for the conformal decomposition
    std::vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
    int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
    std::vector<Point<2> >               num_elem(6);
    std::vector<int>                     corresp(9);
    std::vector<node_status>             pts_statut(4);

    // initializing a vector of velocity and vector of pressure on the nodes of the triangle built with the decomposition function
    std::vector<Tensor<1,2>>             velocity_node(3);
    std::vector<double>                  pressure_node(3);

    // Matrix and RHS sides;
    unsigned int dofs_per_cell = 3*4;
    FullMatrix<double>   cell_mat (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    // We decompose the element
    decomposition(corresp, pts_statut, num_elem, decomp_elem, &nb_poly, coor, dist);

    // setting values to v and p on the vertices of the triangle
    velocity_node[0][0] = ;
    velocity_node[0][1] = ;
    velocity_node[1][0] = ;
    velocity_node[1][1] = ;
    velocity_node[2][0] = ;
    velocity_node[2][1] = ;

    pressure_node[0] = ;
    pressure_node[1] = ;
    pressure_node[2] = ;

    // Initializing a TRG_tools object
    TRG_tools<2>    trg;
    trg.set_coor_trg(decomp_elem);
    trg.set_dofs_per_node(3);
    trg.set_P_on_vertices(pressure_node);
    trg.set_V_on_vertices(velocity_node);

    // Creating the other arguments necesary for the function
    double viscosity =1;
    Tensor<1,2>   force;
    force=0; // force has no use in the GLS_residual at the moment

    GLS_residual_trg(trg, force, cell_mat, cell_rhs, viscosity);
}
