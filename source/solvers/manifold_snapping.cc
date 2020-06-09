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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>

#include "core/parameters.h"



// Constructor for class ManifoldSnapping
template <int dim>
ManifoldSnapping<dim>::ManifoldSnapping(
  const Parameters::Mesh p_mesh_parameters)
  : mesh_parameters(p_mesh_parameters)
  , dof_handler(triangulation)
  , fe(FE_Q<dim>(1), 3)
{}

template <int dim>
ManifoldSnapping<dim>::~ManifoldSnapping()
{}

template <int dim>
void
ManifoldSnapping<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom : " << dof_handler.n_dofs()
            << std::endl;
  nodal_displacement.reinit(dof_handler.n_dofs());
}


template <int dim>
void
ManifoldSnapping<dim>::read()
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
ManifoldSnapping<dim>::write()
{
  GridOut output_grid;

  std::string   file_name("modified.msh");
  std::ofstream output_stream(file_name.c_str());

  output_grid.write_msh(triangulation, output_stream);
}

template <int dim>
void
ManifoldSnapping<dim>::calculate_displacement()
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
ManifoldSnapping<dim>::output(unsigned int iter)
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
ManifoldSnapping<dim>::snap()
{}

template <int dim>
void
ManifoldSnapping<dim>::solve()
{
  read();
  setup_dofs();
  output(0);

  calculate_displacement();
  snap();
  output(1);
  write();
}


// Pre-compile the 2D and 3D version with the types that can occur
template class ManifoldSnapping<2>;
template class ManifoldSnapping<3>;
