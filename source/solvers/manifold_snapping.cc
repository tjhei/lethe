/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 3.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Bruno Blais, Polytechnique Montreal, 2020-
 */

#include "solvers/manifold_snapping.h"

#include <string>

#include "core/parameters.h"



// Constructor for class ManifoldSnapping
template <int dim>
SphereSnapping<dim>::SphereSnapping(const Parameters::Mesh p_mesh_parameters)
  : mesh_parameters(p_mesh_parameters)
  , dof_handler(triangulation)
  , fe(FE_Q<dim>(1), 3)
{}

template <int dim>
SphereSnapping<dim>::~SphereSnapping()
{}

template <int dim>
void
SphereSnapping<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom : " << dof_handler.n_dofs()
            << std::endl;
  nodal_displacement.reinit(dof_handler.n_dofs());

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  dof_snapped.reinit(dof_handler.n_dofs());
}

template <int dim>
void
SphereSnapping<dim>::setup_bcs()
{
  constraints.clear();

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           *deformation_function,
                                           constraints);

  for (unsigned int i_bc = 1; i_bc < 6; ++i_bc)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             i_bc,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}


template <int dim>
void
SphereSnapping<dim>::read()
{
  if (mesh_parameters.type == Parameters::Mesh::Type::gmsh)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream input_file(mesh_parameters.file_name);
      grid_in.read_msh(input_file);
    }

  else
    throw std::runtime_error(
      "Unsupported mesh type - mesh will not be created");
}

template <int dim>
void
SphereSnapping<dim>::read_spheres_information(std::string filename)
{
  std::ifstream infile(filename.c_str());

  std::string line;

  // skip first line for header
  std::getline(infile, line);

  while (std::getline(infile, line))
    {
      std::istringstream iss(line);

      Point<dim> sphere_center;

      double x, y, z, r;

      if (dim == 2)
        {
          iss >> x >> y >> r;
          std::cout << "Snapping to circle " << x << " " << y << " " << r
                    << std::endl;
          spheres_location.push_back(Point<dim>(x, y));
        }

      else if (dim == 3)
        {
          iss >> x >> y >> z >> r;
          std::cout << "Snapping to sphere " << x << " " << y << " " << z << " "
                    << r << std::endl;
          spheres_location.push_back(Point<dim>(x, y, z));
        }
      spheres_radii.push_back(r);
    }
}

template <int dim>
void
SphereSnapping<dim>::write()
{
  GridOut output_grid;

  std::string   file_name("modified.msh");
  std::ofstream output_stream(file_name.c_str());

  GridOutFlags::Msh flags(true);

  output_grid.set_flags(flags);

  output_grid.write_msh(triangulation, output_stream);
}


template <int dim>
void
SphereSnapping<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);

  Functions::ConstantFunction<dim> lambda(1.), mu(1.);

  std::vector<Tensor<1, dim>> rhs_values(n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      double stiffness_scaling =
        size_stiffness ? std::sqrt(cell->measure()) : 1.;

      fe_values.reinit(cell);

      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);

      for (const unsigned int i : fe_values.dof_indices())
        {
          const unsigned int component_i =
            fe.system_to_component_index(i).first;

          for (const unsigned int j : fe_values.dof_indices())
            {
              const unsigned int component_j =
                fe.system_to_component_index(j).first;

              for (const unsigned int q_point :
                   fe_values.quadrature_point_indices())
                {
                  cell_matrix(i, j) +=
                    ((fe_values.shape_grad(i, q_point)[component_i] *
                      fe_values.shape_grad(j, q_point)[component_j] *
                      lambda_values[q_point]) +
                     (fe_values.shape_grad(i, q_point)[component_j] *
                      fe_values.shape_grad(j, q_point)[component_i] *
                      mu_values[q_point]) +
                     ((component_i == component_j) ?
                        (fe_values.shape_grad(i, q_point) *
                         fe_values.shape_grad(j, q_point) *
                         mu_values[q_point]) :
                        0)) *
                    fe_values.JxW(q_point) / stiffness_scaling;
                }
            }
        }

      // Assembling the right hand side is also just as discussed in the
      // introduction:
      for (const unsigned int i : fe_values.dof_indices())
        {
          const unsigned int component_i =
            fe.system_to_component_index(i).first;
        }

      // The transfer from local degrees of freedom into the global matrix
      // and right hand side vector does not depend on the equation under
      // consideration, and is thus the same as in all previous
      // examples.
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

template <int dim>
void
SphereSnapping<dim>::manual_displacement()
{
  Point<dim>              center_point({0.5, 0.5, 0.5});
  double                  radius        = 0.25;
  const unsigned int      dofs_per_cell = fe.dofs_per_cell;
  const unsigned int      dofs_per_face = fe.dofs_per_face;
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());

  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                       dof_handler,
                                       support_points);

  std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          if (cell->face(face)->at_boundary())
            {
              if (cell->face(face)->boundary_id() == 0)
                {
                  for (unsigned int i = 0;
                       i < GeometryInfo<2>::vertices_per_face;
                       ++i)
                    {
                      Point<dim> &   dof_position = cell->face(face)->vertex(i);
                      Tensor<1, dim> radial_vector =
                        dof_position - center_point;
                      double         radial_distance = radial_vector.norm();
                      double         displacement    = radius - radial_distance;
                      Tensor<1, dim> displacement_vector =
                        displacement * (radial_vector) / radial_vector.norm();
                      std::cout << "The displacement vector is : "
                                << displacement_vector << std::endl;
                      for (unsigned int d = 0; d < dim; ++d)
                        dof_position[d] += displacement_vector[d];
                    }


                  //                  Gather the dof indices of the face
                  cell->get_dof_indices(cell_dof_indices);
                  for (unsigned int i = 0; i < dofs_per_face; ++i)
                    {
                      const auto sys_to_comp =
                        fe.face_system_to_component_index(i);
                      unsigned int component = sys_to_comp.first;

                      unsigned int dof_index =
                        cell_dof_indices[fe.face_to_cell_index(i, face)];

                      Point<dim> dof_position = support_points[dof_index];

                      Tensor<1, dim> radial_vector =
                        dof_position - center_point;

                      double radial_distance = radial_vector.norm();

                      double         displacement = radius - radial_distance;
                      Tensor<1, dim> displacement_vector =
                        displacement * (radial_vector) / radial_vector.norm();
                      std::cout << "The displacement vector is : "
                                << displacement_vector << std::endl;

                      nodal_displacement[dof_index] =
                        displacement_vector[component];
                    }
                }
            }
        }
    }
}

template <int dim>
void
SphereSnapping<dim>::output(unsigned int iter)
{
  std::vector<std::string> solution_names(dim, "displacement");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim> data_out;

  // Attach the solution data to data_out object
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(nodal_displacement,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  MappingQ<dim> mapping(1);

  data_out.build_patches(mapping);

  std::string file_name("./nodal_displacement_" +
                        Utilities::int_to_string(iter) + ".vtu");

  std::ofstream output_stream(file_name.c_str());

  data_out.write_vtu(output_stream);
}

template <int dim>
void
SphereSnapping<dim>::snap()
{
  const unsigned int      dofs_per_cell = fe.dofs_per_cell;
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());

  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                       dof_handler,
                                       support_points);

  dof_snapped = 0;


  std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(cell_dof_indices);

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          Point<dim> &dof_position = cell->vertex(v);
          for (unsigned int d = 0; d < dim; ++d)
            {
              unsigned int dof_index = cell->vertex_dof_index(v, d);
              if (dof_snapped[dof_index] == 0)
                {
                  dof_position[d] +=
                    nodal_displacement[cell->vertex_dof_index(v, d)];
                  dof_snapped[dof_index] = 1;
                }
            }
        }
    }
}

template <int dim>
void
SphereSnapping<dim>::solve_manual_snapping()
{
  read();
  setup_dofs();
  output(0);

  manual_displacement();
  snap();
  output(1);
  write();
}

template <int dim>
void
SphereSnapping<dim>::solve_linear_system()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> cg(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  cg.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);

  nodal_displacement = solution;
}

template <int dim>
void
SphereSnapping<dim>::solve()
{
  relaxation_iteration = 3;
  size_stiffness       = true;

  read();
  read_spheres_information("particles.dat");

  deformation_function =
    std::make_shared<DeformToClosestSphere<dim>>(spheres_location,
                                                 spheres_radii,
                                                 relaxation_iteration);
  setup_dofs();
  setup_bcs();
  output(0);

  for (unsigned int it = 0; it < relaxation_iteration; ++it)
    {
      deformation_function->set_time(it + 1);
      setup_bcs();
      assemble_system();
      solve_linear_system();
      snap();
      output(it + 1);
    }
  write();
}



// Pre-compile the 2D and 3D version with the types that can occur
template class SphereSnapping<2>;
template class SphereSnapping<3>;
