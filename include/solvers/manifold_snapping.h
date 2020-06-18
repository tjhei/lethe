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
class ManifoldSnapping
{
public:
  ManifoldSnapping(const Parameters::Mesh p_mesh_parameters);
  ~ManifoldSnapping();

  void
  solve_manual_snapping();

  void
  solve();

private:
  // Allocate the memory for the dofs
  void
  setup_dofs();

  // Reads the mesh to be snapped
  void
  read();

  // Calculates the required displacement to match the mesh on the manifold
  void
  manual_displacement();

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

  Vector<double> solution;
  Vector<double> system_rhs;
  Vector<int>    dof_snapped;



  Vector<double> nodal_displacement;
};


#endif
