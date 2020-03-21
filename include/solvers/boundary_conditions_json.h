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
 * Authors: Bruno Blais & Simon Gauvin, Polytechnique Montreal, 2019 -
 */

// Possibly refactor this class with an std::variant depending on the function
// type and having one class per type instead of multiple vectors that have to
// be the same size

#ifndef LETHE_BOUNDARYCONDITIONSJSON_H
#define LETHE_BOUNDARYCONDITIONSJSON_H

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parsed_function.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "boundary_conditions.h"
#include "core/function_parser.h"
#include "core/parameter_translator.h"

using namespace dealii;

namespace
{
  template <int dim>
  std::string
  get_default_vnames()
  {
    return FunctionParser<dim>::default_variable_names() + ",t";
  }
} // namespace

namespace BoundaryConditions
{
  template <int dim>
  void
  NSBoundaryConditions<dim>::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, BoundaryConditions::BoundaryType>
      boundaryTypes = {{"noslip", BoundaryConditions::BoundaryType::noslip},
                       {"slip", BoundaryConditions::BoundaryType::slip},
                       {"function", BoundaryConditions::BoundaryType::function},
                       {"periodic",
                        BoundaryConditions::BoundaryType::periodic}};

    std::string        default_vnames = get_default_vnames<dim>();
    const unsigned int n_components   = 1;
    std::string        expr;
    for (unsigned int i = 0; i < n_components; i++)
      expr += "; 0";

    auto child = root.get_child_optional("boundary conditions");
    if (!child)
      {
        // boundary conditions subsection doesn't exist. Do we want to throw an
        // error?
        throw std::runtime_error(
          "boundary conditions subsection doesn't exist");
      }

    // Loop through all boundary conditions!
    for (auto it = child->begin(); it != child->end(); ++it)
      {
        auto child_function = it->second;
        type.push_back(child_function.get(
          "type",
          BoundaryConditions::BoundaryType::noslip,
          ParameterTranslator<BoundaryConditions::BoundaryType>(
            boundaryTypes)));
        bcFunctions.push_back({});
        periodic_id.push_back({});
        periodic_direction.push_back({});
        id.push_back({});

        switch (type.back())
          {
              case BoundaryConditions::BoundaryType::function: {
                bcFunctions.back().u = parse_function<dim>(child_function,
                                                           default_vnames,
                                                           expr,
                                                           "u");
                bcFunctions.back().v = parse_function<dim>(child_function,
                                                           default_vnames,
                                                           expr,
                                                           "v");
                bcFunctions.back().w = parse_function<dim>(child_function,
                                                           default_vnames,
                                                           expr,
                                                           "w");

                bcFunctions.back().cor[0] = child_function.get("x", 0);
                bcFunctions.back().cor[1] = child_function.get("y", 0);
                if (dim == 3)
                  {
                    bcFunctions.back().cor[2] = child_function.get("z", 0);
                  }
                break;
              }
              case BoundaryConditions::BoundaryType::periodic: {
                periodic_id.back() = child_function.get("periodic_id", 0);
                periodic_direction.back() =
                  child_function.get("periodic_direction", 0);
              }
          }
        id.back() = child_function.get("id", size);
        size++;
      }
  }
} // namespace BoundaryConditions


#endif
