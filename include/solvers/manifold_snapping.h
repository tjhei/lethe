/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 -  by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Bruno Blais, Polytechnique Montreal, 2020 -
 */

#ifndef lethe_manifold_snapping_h
#define lethe_manifold_snapping_h


// Base
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

// Lac
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

// Grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// Dofs
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// Fe
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

// Numerics
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


// Lethe Includes
#include <core/parameters.h>


using namespace dealii;

template <int dim>
class DeformToClosestSphere : public Function<dim>
{
public:
  DeformToClosestSphere(std::vector<Point<dim>> spheres_loc,
                        std::vector<double>     spheres_radii,
                        unsigned                iteration_number = 1)
    : Function<dim>(dim)
    , spheres_location(spheres_loc)
    , spheres_radii(spheres_radii)
    , iteration_number(iteration_number)
  {}
  virtual void
  vector_value(const Point<dim> &point, Vector<double> &values) const override;

private:
  std::vector<Point<dim>> spheres_location;
  std::vector<double>     spheres_radii;
  unsigned                iteration_number;
};

template <int dim>
void
DeformToClosestSphere<dim>::vector_value(const Point<dim> &point,
                                         Vector<double> &  values) const
{
  const double step = this->get_time();
  double       relaxation_factor =
    step == iteration_number ? 1 : 1. / iteration_number;
  double smallest_displacement = DBL_MAX;
  for (unsigned int pt = 0; pt < spheres_location.size(); ++pt)
    {
      Point<dim>     center_point    = spheres_location[pt];
      double         radius          = spheres_radii[pt];
      Tensor<1, dim> radial_vector   = point - center_point;
      double         radial_distance = radial_vector.norm();
      double         displacement    = radius - radial_distance;
      if (std::abs(displacement) < smallest_displacement)
        {
          smallest_displacement              = std::abs(displacement);
          Tensor<1, dim> displacement_vector = displacement * (radial_vector) /
                                               radial_vector.norm() *
                                               relaxation_factor;

          for (unsigned int d = 0; d < dim; ++d)
            values[d] = displacement_vector[d];
        }
    }
}

/**
 * A mesh modification class that snaps boundary nodes to the nearest manifold
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 *
 * @ingroup solvers
 * @author Bruno Blais, 2020
 */

template <int dim>
class SphereSnapping
{
public:
  SphereSnapping(const Parameters::Mesh p_mesh_parameters);
  ~SphereSnapping();

  void
  solve_manual_snapping();

  void
  solve();

private:
  // Allocate the memory for the dofs
  void
  setup_dofs();

  // Allocate the memory for the dofs
  void
  setup_bcs();

  // Reads the mesh to be snapped
  void
  read();

  // Calculates the required displacement to match the mesh on the manifold
  void
  manual_displacement();

  // Read location and radii of spheres
  // Expect format is
  // x y z r
  // d d d d
  // d d d d
  // ...
  void
  read_spheres_information(std::string filename);

  // Displaces the mesh to match manifold
  void
  snap();

  // Assemble system of equation for linear elasticity
  void
  assemble_system();

  // Solve linear system for linear elasticity problem
  void
  solve_linear_system();

  // Writes the snapped mesh
  void
  write();

  // Output displacement field
  void
  output(unsigned int iter);

  const Parameters::Mesh mesh_parameters;

  Triangulation<dim> triangulation;
  DoFHandler<dim>    dof_handler;
  FESystem<dim>      fe;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>                              solution;
  Vector<double>                              system_rhs;
  Vector<int>                                 dof_snapped;
  std::shared_ptr<DeformToClosestSphere<dim>> deformation_function;

  std::vector<Point<dim>> spheres_location;
  std::vector<double>     spheres_radii;
  unsigned int            relaxation_iteration;



  Vector<double> nodal_displacement;
};


#endif
