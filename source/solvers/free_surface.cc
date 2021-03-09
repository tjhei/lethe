#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/vector_tools.h>

#include <core/bdf.h>
#include <core/sdirk.h>
#include <core/time_integration_utilities.h>
#include <core/utilities.h>
#include <solvers/free_surface.h>


template <int dim>
void
FreeSurface<dim>::assemble_matrix_and_rhs(
  const Parameters::SimulationControl::TimeSteppingMethod time_stepping_method)
{
  std::cout << "entrée dans assemble_matrix_and_rhs..." << std::endl;
  assemble_system<true>(time_stepping_method);
  std::cout << "...sortie de assemble_matrix_and_rhs" << std::endl;
}


template <int dim>
void
FreeSurface<dim>::assemble_rhs(
  const Parameters::SimulationControl::TimeSteppingMethod time_stepping_method)
{
  std::cout << "entrée dans assemble_rhs..." << std::endl;
  assemble_system<false>(time_stepping_method);
  std::cout << "...sortie de assemble_rhs" << std::endl;
}


template <int dim>
template <bool assemble_matrix>
void
FreeSurface<dim>::assemble_system(
  const Parameters::SimulationControl::TimeSteppingMethod time_stepping_method)
{
  std::cout << "entrée dans assemble_system..." << std::endl;
  if (assemble_matrix)
    system_matrix = 0;
  system_rhs = 0;

  // Vector for the BDF coefficients
  // The coefficients are stored in the following fashion :
  // 0 - n+1
  // 1 - n
  // 2 - n-1
  // 3 - n-2
  std::vector<double> time_steps_vector =
    simulation_control->get_time_steps_vector();

  // Time steps and inverse time steps which is used for numerous calculations
  //  const double dt = time_steps_vector[0];
  //  const double sdt = 1. / dt;

  std::cout << "...calcul de bdf_coefs..." << std::endl;
  Vector<double> bdf_coefs;

  if (time_stepping_method ==
        Parameters::SimulationControl::TimeSteppingMethod::bdf1 ||
      time_stepping_method ==
        Parameters::SimulationControl::TimeSteppingMethod::steady_bdf)
    bdf_coefs = bdf_coefficients(1, time_steps_vector);

  if (time_stepping_method ==
      Parameters::SimulationControl::TimeSteppingMethod::bdf2)
    bdf_coefs = bdf_coefficients(2, time_steps_vector);

  if (time_stepping_method ==
      Parameters::SimulationControl::TimeSteppingMethod::bdf3)
    bdf_coefs = bdf_coefficients(3, time_steps_vector);

  if (time_stepping_method ==
        Parameters::SimulationControl::TimeSteppingMethod::sdirk22_1 ||
      time_stepping_method ==
        Parameters::SimulationControl::TimeSteppingMethod::sdirk33_1)
    {
      throw std::runtime_error(
        "SDIRK schemes are not supported by heat transfer physics");
    }


  //  auto &source_term =
  //  simulation_parameters.sourceTerm->heat_transfer_source;
  //  source_term.set_time(simulation_control->get_current_time());

  std::cout << "...fe_values_fs..." << std::endl;
  FEValues<dim> fe_values_fs(*fe,
                             *this->cell_quadrature,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values |
                               update_hessians);

  auto &evaluation_point = this->get_evaluation_point();

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  std::cout << "...autres définitions..." << std::endl;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const unsigned int     n_q_points = this->cell_quadrature->size();
  std::vector<double>    source_term_values(n_q_points);
  const DoFHandler<dim> *dof_handler_fluid =
    multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);
  FEValues<dim> fe_values_flow(dof_handler_fluid->get_fe(),
                               *this->cell_quadrature,
                               update_values | update_quadrature_points |
                                 update_gradients);

  // Shape functions and gradients
  std::cout << "...shape functions..." << std::endl;
  std::vector<double>         phi_alpha(dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_alpha(dofs_per_cell);
  std::vector<Tensor<2, dim>> hess_phi_alpha(dofs_per_cell);
  std::vector<double>         laplacian_phi_alpha(dofs_per_cell);


  // Velocity values
  std::cout << "...velocity values..." << std::endl;
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> velocity_gradient_values(n_q_points);

  std::vector<double>         present_alpha_values(n_q_points);
  std::vector<Tensor<1, dim>> alpha_gradients(n_q_points);
  std::vector<double>         present_alpha_laplacians(n_q_points);

  // Values for backward Euler scheme
  std::cout << "...backward Euler scheme..." << std::endl;
  std::vector<double> p1_alpha_values(n_q_points);
  std::vector<double> p2_alpha_values(n_q_points);
  std::vector<double> p3_alpha_values(n_q_points);

  std::cout << "...entrée dans active_cell_iterators()..." << std::endl;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;

          fe_values_fs.reinit(cell);

          fe_values_fs.get_function_gradients(evaluation_point,
                                              alpha_gradients);


          typename DoFHandler<dim>::active_cell_iterator velocity_cell(
            &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

          fe_values_flow.reinit(velocity_cell);

          if (multiphysics->fluid_dynamics_is_block())
            {
              fe_values_flow[velocities].get_function_values(
                *multiphysics->get_block_solution(PhysicsID::fluid_dynamics),
                velocity_values);
              fe_values_flow[velocities].get_function_gradients(
                *multiphysics->get_block_solution(PhysicsID::fluid_dynamics),
                velocity_gradient_values);
            }
          else
            {
              fe_values_flow[velocities].get_function_values(
                *multiphysics->get_solution(PhysicsID::fluid_dynamics),
                velocity_values);
              fe_values_flow[velocities].get_function_gradients(
                *multiphysics->get_solution(PhysicsID::fluid_dynamics),
                velocity_gradient_values);
            }

          // Gather present value
          fe_values_fs.get_function_values(evaluation_point,
                                           present_alpha_values);


          // Gather present laplacian
          fe_values_fs.get_function_laplacians(evaluation_point,
                                               present_alpha_laplacians);

          // Gather the previous time steps for heat transfer depending on
          // the number of stages of the time integration method
          if (time_stepping_method !=
              Parameters::SimulationControl::TimeSteppingMethod::steady)
            {
              fe_values_fs.get_function_values(this->solution_m1,
                                               p1_alpha_values);
              //              fe_values_fs.get_function_gradients(this->solution_m1,
              //                                                  p1_alpha_gradients);
              //                                                  //see if
              //                                                  needed for
              //                                                  normal vector
              //                                                  calculation
            }

          if (time_stepping_method_has_two_stages(time_stepping_method))
            {
              fe_values_fs.get_function_values(this->solution_m2,
                                               p2_alpha_values);

              //              fe_values_fs.get_function_gradients(this->solution_m2,
              //                                                  p2_alpha_gradients);
            }

          if (time_stepping_method_has_three_stages(time_stepping_method))
            {
              fe_values_fs.get_function_values(this->solution_m3,
                                               p3_alpha_values);

              //              fe_values_fs.get_function_gradients(this->solution_m3,
              //                                                  p3_alpha_gradients);
            }

          //          source_term.value_list(fe_values_fs.get_quadrature_points(),
          //                                 source_term_values);


          // assembling local matrix and right hand side
          for (const unsigned int q : fe_values_fs.quadrature_point_indices())
            {
              // Store JxW in local variable for faster access
              const double JxW = fe_values_fs.JxW(q);

              const auto velocity = velocity_values[q];


              // Calculation of the magnitude of the velocity for the
              // stabilization parameter
              //              const double u_mag = std::max(velocity.norm(),
              //              1e-12);

              // Calculation of the GLS stabilization parameter. The
              // stabilization parameter used is different if the simulation is
              // steady or unsteady. In the unsteady case it includes the value
              // of the time-step //TODO see if necessary here
              //              const double tau =
              //                is_steady(time_stepping_method) ?
              //                  1. / std::sqrt(std::pow(2. * 1 * u_mag /
              //                  h, 2) +
              //                                 9 * std::pow(4 * alpha / (h *
              //                                 h), 2)) :
              //                  1. / std::sqrt(std::pow(sdt, 2) +
              //                                 std::pow(2. * 1 * u_mag /
              //                                 h, 2) + 9 * std::pow(4 * alpha
              //                                 / (h * h), 2));
              //              const double tau_ggls = std::pow(h, fe->degree +
              //              1) / 6. / 1;

              // Gather the shape functions and their gradient
              for (unsigned int k : fe_values_fs.dof_indices())
                {
                  phi_alpha[k]      = fe_values_fs.shape_value(k, q);
                  grad_phi_alpha[k] = fe_values_fs.shape_grad(k, q);
                  hess_phi_alpha[k] = fe_values_fs.shape_hessian(k, q);

                  laplacian_phi_alpha[k] = trace(hess_phi_alpha[k]);
                }



              for (const unsigned int i : fe_values_fs.dof_indices())
                {
                  const auto phi_alpha_i      = phi_alpha[i];
                  const auto grad_phi_alpha_i = grad_phi_alpha[i];


                  if (assemble_matrix)
                    {
                      for (const unsigned int j : fe_values_fs.dof_indices())
                        {
                          const auto phi_alpha_j      = phi_alpha[j];
                          const auto grad_phi_alpha_j = grad_phi_alpha[j];
                          //                          const auto
                          //                          laplacian_phi_alpha_j =
                          //                            laplacian_phi_alpha[j];



                          // Weak form for : u * nabla(alpha) = 0
                          // TODO add compression term
                          cell_matrix(i, j) +=
                            (phi_alpha_i * velocity * grad_phi_alpha_j) * JxW;

                          //                          auto strong_jacobian =
                          //                            velocity *
                          //                            grad_phi_alpha_j -
                          //                            laplacian_phi_alpha_j;

                          // Mass matrix for transient simulation
                          if (is_bdf(time_stepping_method))
                            {
                              cell_matrix(i, j) +=
                                phi_alpha_j * phi_alpha_i * bdf_coefs[0] * JxW;

                              //                              strong_jacobian +=
                              //                              1 * phi_alpha_j *
                              //                              bdf_coefs[0];
                              // TODO see if stabilization necessary
                              //                              if (GGLS)
                              //                                {
                              //                                  cell_matrix(i,
                              //                                  j) +=
                              //                                    1 *
                              //                                    1 *
                              //                                    tau_ggls *
                              //                                    (grad_phi_alpha_i
                              //                                    *
                              //                                    grad_phi_alpha_j)
                              //                                    *
                              //                                    bdf_coefs[0]
                              //                                    * JxW;
                              //                                }
                            }

                          //                          cell_matrix(i, j) += //
                          //                          tau *
                          //                            strong_jacobian *
                          //                            (grad_phi_alpha_i *
                          //                            velocity_values[q]) *
                          //                            JxW;
                        }
                    }

                  // rhs for : u * nabla(alpha) = 0
                  // TODO add compression term
                  cell_rhs(i) -=
                    (phi_alpha_i * velocity_values[q] * alpha_gradients[q]) *
                    JxW;

                  // Calculate the strong residual for GLS stabilization
                  //                  auto strong_residual =
                  //                    1 * velocity_values[q] *
                  //                    alpha_gradients[q] - 1 *
                  //                    present_alpha_laplacians[q];
                  auto strong_residual = 0;

                  // Residual associated with BDF schemes
                  if (time_stepping_method == Parameters::SimulationControl::
                                                TimeSteppingMethod::bdf1 ||
                      time_stepping_method == Parameters::SimulationControl::
                                                TimeSteppingMethod::steady_bdf)
                    {
                      cell_rhs(i) -= 1 *
                                     (bdf_coefs[0] * present_alpha_values[q] +
                                      bdf_coefs[1] * p1_alpha_values[q]) *
                                     phi_alpha_i * JxW;

                      strong_residual +=
                        bdf_coefs[0] * present_alpha_values[q] +
                        bdf_coefs[1] * p1_alpha_values[q];

                      // TODO see if stabilization necessary
                      //                      if (GGLS)
                      //                        {
                      //                          cell_rhs(i) -=
                      //                            1 * 1 * tau_ggls *
                      //                            grad_phi_alpha_i *
                      //                            (bdf_coefs[0] *
                      //                            alpha_gradients[q] +
                      //                             bdf_coefs[1] *
                      //                             p1_alpha_gradients[q]) *
                      //                            JxW;
                      //                        }
                    }

                  if (time_stepping_method ==
                      Parameters::SimulationControl::TimeSteppingMethod::bdf2)
                    {
                      cell_rhs(i) -= 1 *
                                     (bdf_coefs[0] * present_alpha_values[q] +
                                      bdf_coefs[1] * p1_alpha_values[q] +
                                      bdf_coefs[2] * p2_alpha_values[q]) *
                                     phi_alpha_i * JxW;

                      strong_residual +=
                        bdf_coefs[0] * present_alpha_values[q] +
                        bdf_coefs[1] * p1_alpha_values[q] +
                        bdf_coefs[2] * p2_alpha_values[q];

                      // see if stabilization necessary
                      //                      if (GGLS)
                      //                        {
                      //                          cell_rhs(i) -=
                      //                            1 * 1 * tau_ggls *
                      //                            grad_phi_alpha_i *
                      //                            (bdf_coefs[0] *
                      //                            alpha_gradients[q] +
                      //                             bdf_coefs[1] *
                      //                             p1_alpha_gradients[q] +
                      //                             bdf_coefs[2] *
                      //                             p2_alpha_gradients[q]) *
                      //                            JxW;
                      //                        }
                    }

                  if (time_stepping_method ==
                      Parameters::SimulationControl::TimeSteppingMethod::bdf3)
                    {
                      cell_rhs(i) -= 1 *
                                     (bdf_coefs[0] * present_alpha_values[q] +
                                      bdf_coefs[1] * p1_alpha_values[q] +
                                      bdf_coefs[2] * p2_alpha_values[q] +
                                      bdf_coefs[3] * p3_alpha_values[q]) *
                                     phi_alpha_i * JxW;

                      strong_residual +=
                        1 * (bdf_coefs[0] * present_alpha_values[q] +
                             bdf_coefs[1] * p1_alpha_values[q] +
                             bdf_coefs[2] * p2_alpha_values[q] +
                             bdf_coefs[3] * p3_alpha_values[q]);

                      // TODO see if stabilization necessary
                      //                      if (GGLS)
                      //                        {
                      //                          cell_rhs(i) -=
                      //                            1 * 1 * tau_ggls *
                      //                            grad_phi_alpha_i *
                      //                            (bdf_coefs[0] *
                      //                            alpha_gradients[q] +
                      //                             bdf_coefs[1] *
                      //                             p1_alpha_gradients[q] +
                      //                             bdf_coefs[2] *
                      //                             p2_alpha_gradients[q] +
                      //                             bdf_coefs[3] *
                      //                             p3_alpha_gradients[q]) *
                      //                            JxW;
                      //                        }
                    }


                  cell_rhs(i) -= 1 *
                                 (strong_residual *
                                  (grad_phi_alpha_i * velocity_values[q])) *
                                 JxW;
                }

            } // end loop on quadrature points

          // transfer cell contribution into global objects
          cell->get_dof_indices(local_dof_indices);
          zero_constraints.distribute_local_to_global(cell_matrix,
                                                      cell_rhs,
                                                      local_dof_indices,
                                                      system_matrix,
                                                      system_rhs);
        } // end loop active cell
    }
  std::cout << "...end loop active cell..." << std::endl;
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  std::cout << "...sortie de assemble_system" << std::endl;
}


template <int dim>
void
FreeSurface<dim>::attach_solution_to_output(DataOut<dim> &data_out)
{
  std::cout << "entrée dans attach_solution_to_output..." << std::endl;
  data_out.add_data_vector(dof_handler, present_solution, "alpha");
  std::cout << "...sortie de attach_solution_to_output" << std::endl;
}


template <int dim>
void
FreeSurface<dim>::finish_simulation()
{
  std::cout << "entrée dans finish_simulation..." << std::endl;
  auto         mpi_communicator = triangulation->get_communicator();
  unsigned int this_mpi_process(
    Utilities::MPI::this_mpi_process(mpi_communicator));

  if (this_mpi_process == 0 &&
      simulation_parameters.analytical_solution->verbosity ==
        Parameters::Verbosity::verbose)
    {
      error_table.omit_column_from_convergence_rate_evaluation("cells");


      if (simulation_parameters.simulation_control.method ==
          Parameters::SimulationControl::TimeSteppingMethod::steady)
        {
          error_table.evaluate_all_convergence_rates(
            ConvergenceTable::reduction_rate_log2);
        }
      error_table.set_scientific("error_alpha", true);
      error_table.write_text(std::cout);
    }
  std::cout << "...sortie de finish_simulation" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::percolate_time_vectors()
{
  std::cout << "entrée dans percolate_time_vectors..." << std::endl;
  solution_m3 = solution_m2;
  solution_m2 = solution_m1;
  solution_m1 = present_solution;
  std::cout << "...sortie de percolate_time_vectors" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::finish_time_step()
{
  std::cout << "entrée dans finish_time_step..." << std::endl;
  percolate_time_vectors();
  std::cout << "sortie de finish_time_step" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::postprocess(bool first_iteration)
{
  std::cout << "entrée dans postprocess..." << std::endl;
  // TODO see if necessary
  if (simulation_parameters.analytical_solution->calculate_error() == true &&
      !first_iteration)
    {
      //      double alpha_error = calculate_L2_error();

      //      error_table.add_value("cells",
      //                            this->triangulation->n_global_active_cells());
      //      error_table.add_value("error_alpha", alpha_error);

      //      if (simulation_parameters.analytical_solution->verbosity ==
      //          Parameters::Verbosity::verbose)
      //        {
      //          this->pcout << "L2 error alpha : " << alpha_error <<
      //          std::endl;
      //        }
    }
  std::cout << "...sortie de postprocess" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::pre_mesh_adaptation()
{
  std::cout << "entrée dans pre_mesh_adaptation..." << std::endl;
  solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
  solution_transfer_m1.prepare_for_coarsening_and_refinement(solution_m1);
  solution_transfer_m2.prepare_for_coarsening_and_refinement(solution_m2);
  solution_transfer_m3.prepare_for_coarsening_and_refinement(solution_m3);
  std::cout << "...sortie de pre_mesh_adaptation" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::post_mesh_adaptation()
{
  std::cout << "entrée dans post_mesh_adaptation..." << std::endl;
  auto mpi_communicator = triangulation->get_communicator();


  // Set up the vectors for the transfer
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
  TrilinosWrappers::MPI::Vector tmp_m1(locally_owned_dofs, mpi_communicator);
  TrilinosWrappers::MPI::Vector tmp_m2(locally_owned_dofs, mpi_communicator);
  TrilinosWrappers::MPI::Vector tmp_m3(locally_owned_dofs, mpi_communicator);

  // Interpolate the solution at time and previous time
  solution_transfer.interpolate(tmp);
  solution_transfer_m1.interpolate(tmp_m1);
  solution_transfer_m2.interpolate(tmp_m2);
  solution_transfer_m3.interpolate(tmp_m3);

  // Distribute constraints
  nonzero_constraints.distribute(tmp);
  nonzero_constraints.distribute(tmp_m1);
  nonzero_constraints.distribute(tmp_m2);
  nonzero_constraints.distribute(tmp_m3);

  // Fix on the new mesh
  present_solution = tmp;
  solution_m1      = tmp_m1;
  solution_m2      = tmp_m2;
  solution_m3      = tmp_m3;
  std::cout << "...sortie de post_mesh_adaptation" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::write_checkpoint()
{
  std::cout << "entrée dans write_checkpoint..." << std::endl;
  std::vector<const TrilinosWrappers::MPI::Vector *> sol_set_transfer;

  sol_set_transfer.push_back(&present_solution);
  sol_set_transfer.push_back(&solution_m1);
  sol_set_transfer.push_back(&solution_m2);
  sol_set_transfer.push_back(&solution_m3);
  solution_transfer.prepare_for_serialization(sol_set_transfer);
  std::cout << "...sortie de write_checkpoint" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::read_checkpoint()
{
  std::cout << "entrée dans read_checkpoint..." << std::endl;
  auto mpi_communicator = triangulation->get_communicator();
  this->pcout << "Reading free surface checkpoint" << std::endl;

  std::vector<TrilinosWrappers::MPI::Vector *> input_vectors(4);
  TrilinosWrappers::MPI::Vector distributed_system(locally_owned_dofs,
                                                   mpi_communicator);
  TrilinosWrappers::MPI::Vector distributed_system_m1(locally_owned_dofs,
                                                      mpi_communicator);
  TrilinosWrappers::MPI::Vector distributed_system_m2(locally_owned_dofs,
                                                      mpi_communicator);
  TrilinosWrappers::MPI::Vector distributed_system_m3(locally_owned_dofs,
                                                      mpi_communicator);

  input_vectors[0] = &distributed_system;
  input_vectors[1] = &distributed_system_m1;
  input_vectors[2] = &distributed_system_m2;
  input_vectors[3] = &distributed_system_m3;

  solution_transfer.deserialize(input_vectors);

  present_solution = distributed_system;
  solution_m1      = distributed_system_m1;
  solution_m2      = distributed_system_m2;
  solution_m3      = distributed_system_m3;
  std::cout << "...sortie de read_checkpoint" << std::endl;
}


template <int dim>
void
FreeSurface<dim>::setup_dofs()
{
  std::cout << "FREE SURFACE - entrée dans setup_dofs..." << std::endl;
  std::cout << "...distribute_dofs..." << std::endl;
  dof_handler.distribute_dofs(*fe);
  std::cout << "...DoFRenumbering..." << std::endl;
  DoFRenumbering::Cuthill_McKee(this->dof_handler);

  std::cout << "...mpi_communicator..." << std::endl;
  auto mpi_communicator = triangulation->get_communicator();

  std::cout << "...locally_owned_dofs..." << std::endl;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  std::cout << "...present_solution..." << std::endl;
  present_solution.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);

  // Previous solutions for transient schemes
  std::cout << "...previous solutions for transient schemes..." << std::endl;
  solution_m1.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);
  solution_m2.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);
  solution_m3.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  newton_update.reinit(locally_owned_dofs, mpi_communicator);

  local_evaluation_point.reinit(this->locally_owned_dofs, mpi_communicator);

  std::cout << "...nonzero_constraints..." << std::endl;
  {
    nonzero_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            nonzero_constraints);

    //    for (unsigned int i_bc = 0;
    //         i_bc < this->simulation_parameters.boundary_conditions_fs.size;
    //         ++i_bc)
    //      {
    //        // Dirichlet condition : imposed alpha at i_bc
    //        if (this->simulation_parameters.boundary_conditions_fs.type[i_bc]
    //        ==
    //            BoundaryConditions::BoundaryType::alpha)
    //          {
    //            VectorTools::interpolate_boundary_values(
    //              this->dof_handler,
    //              this->simulation_parameters.boundary_conditions_fs.id[i_bc],
    //              dealii::Functions::ConstantFunction<dim>(
    //                this->simulation_parameters.boundary_conditions_fs.value[i_bc]),
    //              nonzero_constraints);
    //          }
    //      }
  }
  nonzero_constraints.close();

  // Boundary conditions for Newton correction
  std::cout << "...zero_constraints..." << std::endl;
  {
    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            zero_constraints);

    //    for (unsigned int i_bc = 0;
    //         i_bc < this->simulation_parameters.boundary_conditions_fs.size;
    //         ++i_bc)
    //      {
    //        if (this->simulation_parameters.boundary_conditions_fs.type[i_bc]
    //        ==
    //            BoundaryConditions::BoundaryType::alpha)
    //          {
    //            VectorTools::interpolate_boundary_values(
    //              this->dof_handler,
    //              this->simulation_parameters.boundary_conditions_fs.id[i_bc],
    //              Functions::ZeroFunction<dim>(),
    //              zero_constraints);
    //          }
    //      }
  }
  zero_constraints.close();

  // Sparse matrices initialization
  std::cout << "...sparse matrices..." << std::endl;
  DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  dsp,
                                  nonzero_constraints,
                                  /*keep_constrained_dofs = */ true);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);

  this->pcout << "   Number of free surface degrees of freedom: "
              << dof_handler.n_dofs() << std::endl;

  std::cout << "...sortie de setup_dofs" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::set_initial_conditions()
{
  std::cout << "entrée dans set_initial_conditions" << std::endl;
  VectorTools::interpolate(
    *this->mapping,
    dof_handler,
    simulation_parameters.initial_condition->free_surface,
    newton_update);
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
  finish_time_step();
  std::cout << "...sortie de set_initial_conditions" << std::endl;
}

template <int dim>
void
FreeSurface<dim>::solve_linear_system(const bool initial_step,
                                      const bool /*renewed_matrix*/)
{
  std::cout << "entrée dans solve_linear_system..." << std::endl;
  auto mpi_communicator = triangulation->get_communicator();

  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_constraints : this->zero_constraints;

  const double absolute_residual =
    simulation_parameters.linear_solver.minimum_residual;
  const double relative_residual =
    simulation_parameters.linear_solver.relative_residual;

  const double linear_solver_tolerance =
    std::max(relative_residual * system_rhs.l2_norm(), absolute_residual);

  if (this->simulation_parameters.linear_solver.verbosity !=
      Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << linear_solver_tolerance << std::endl;
    }

  const double ilu_fill = simulation_parameters.linear_solver.ilu_precond_fill;
  const double ilu_atol = simulation_parameters.linear_solver.ilu_precond_atol;
  const double ilu_rtol = simulation_parameters.linear_solver.ilu_precond_rtol;
  TrilinosWrappers::PreconditionILU::AdditionalData preconditionerOptions(
    ilu_fill, ilu_atol, ilu_rtol, 0);

  TrilinosWrappers::PreconditionILU ilu_preconditioner;

  ilu_preconditioner.initialize(system_matrix, preconditionerOptions);

  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(
    simulation_parameters.linear_solver.max_iterations,
    linear_solver_tolerance,
    true,
    true);

  TrilinosWrappers::SolverGMRES::AdditionalData solver_parameters(
    false, simulation_parameters.linear_solver.max_krylov_vectors);


  TrilinosWrappers::SolverGMRES solver(solver_control, solver_parameters);


  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs,
               ilu_preconditioner);

  if (simulation_parameters.linear_solver.verbosity !=
      Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Iterative solver took : " << solver_control.last_step()
                  << " steps " << std::endl;
    }

  constraints_used.distribute(completely_distributed_solution);
  newton_update = completely_distributed_solution;
  std::cout << "...sortie de solve_linear_system" << std::endl;
}



template class FreeSurface<2>;
template class FreeSurface<3>;
