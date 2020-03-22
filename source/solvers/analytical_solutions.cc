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
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */

#include "solvers/analytical_solutions.h"

#include "core/parameter_translator.h"

namespace
{
  const std::unordered_map<std::string, Parameters::Verbosity> verbosities = {
    {"verbose", Parameters::Verbosity::verbose},
    {"quiet", Parameters::Verbosity::quiet}};
} // namespace

namespace AnalyticalSolutions
{
  template <int dim>
  void
  AnalyticalSolution<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("analytical solution");
    prm.declare_entry(
      "enable",
      "false",
      Patterns::Bool(),
      "Enable the calculation of the analytical solution and L2 error");
    prm.declare_entry(
      "verbosity",
      "quiet",
      Patterns::Selection("quiet|verbose"),
      "State whether from the post-processing values should be printed "
      "Choices are <quiet|verbose>.");

    prm.declare_entry(
      "filename",
      "L2Error",
      Patterns::FileName(),
      "File name for the output for the L2Error table with respect to time or mesh ");
    prm.leave_subsection();
  }

  template <int dim>
  void
  AnalyticalSolution<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("analytical solution");
    enable               = prm.get_bool("enable");
    filename             = prm.get("filename");
    const std::string op = prm.get("verbosity");
    if (op == "verbose")
      verbosity = Parameters::Verbosity::verbose;
    if (op == "quiet")
      verbosity = Parameters::Verbosity::quiet;
    prm.leave_subsection();
  }

  template <int dim>
  void
  AnalyticalSolution<dim>::parse_parameters(boost::property_tree::ptree &root)
  {
    enable   = root.get("analytical solution.enable", false);
    filename = root.get("analytical solution.filename", "L2Error");
    verbosity =
      root.get("analytical solution.verbosity",
               Parameters::Verbosity::quiet,
               ParameterTranslator<Parameters::Verbosity>(verbosities));
  }

  template <int dim>
  void
  NSAnalyticalSolution<dim>::declare_parameters(ParameterHandler &prm)
  {
    this->AnalyticalSolution<dim>::declare_parameters(prm);
    prm.enter_subsection("analytical solution");
    prm.enter_subsection("uvw");

    auto velocity_function =
      std::make_shared<Functions::ParsedFunction<dim>>(dim + 1);
    velocity_function->declare_parameters(prm, dim);
    velocity = velocity_function;
    if (dim == 2)
      prm.set("Function expression", "0; 0; 0;");
    if (dim == 3)
      prm.set("Function expression", "0; 0; 0; 0;");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void
  NSAnalyticalSolution<dim>::parse_parameters(ParameterHandler &prm)
  {
    this->AnalyticalSolution<dim>::parse_parameters(prm);
    prm.enter_subsection("analytical solution");
    prm.enter_subsection("uvw");
    if (auto velocity_function =
          dynamic_cast<Functions::ParsedFunction<dim> *>(velocity.get()))
      {
        velocity_function->parse_parameters(prm);
      }
    else
      {
        throw std::runtime_error(
          "Could not convert velocity function in analytical solutions");
      }
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void
  NSAnalyticalSolution<dim>::parse_parameters(boost::property_tree::ptree &root)
  {
    this->AnalyticalSolution<dim>::parse_parameters(root);
    auto child = root.get_child_optional("analytical solution");
    if (!child)
      {
        // Throw an error?
        throw std::runtime_error("Analytical subsection does not exist");
      }

    // enable = root.get("analytical solution.enable", false);
    // filename = root.get("analytical solution.filename", "L2Error");
    // verbosity = root.get("analytical solution.verbosity",
    //                      Parameters::Verbosity::quiet,
    //                      ParameterTranslator<Parameters::Verbosity>(verbosities));
  }
} // namespace AnalyticalSolutions
// Pre-compile the 2D and 3D
template class AnalyticalSolutions::NSAnalyticalSolution<2>;
template class AnalyticalSolutions::NSAnalyticalSolution<3>;
template class AnalyticalSolutions::AnalyticalSolution<2>;
template class AnalyticalSolutions::AnalyticalSolution<3>;
