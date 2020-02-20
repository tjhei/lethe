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

#ifndef LETHE_BOUNDARYCONDITIONSJSON_H
#define LETHE_BOUNDARYCONDITIONSJSON_H

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/function_parser.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "core/parameter_translator.h"

using namespace dealii;

namespace BoundaryConditionsJSON
{
  enum class BoundaryType
  {
    noslip,
    slip,
    function,
    periodic
  };

  template <int dim>
  class BoundaryFunction
  {
  public:
    // Velocity components
    Functions::ParsedFunction<dim> u;
    Functions::ParsedFunction<dim> v;
    Functions::ParsedFunction<dim> w;

    // Point for the center of rotation
    Point<dim> cor;
  };

  template <int dim>
  class NSBoundaryConditions
  {
  public:
    // ID of boundary condition
    std::vector<unsigned int> id;

    // List of boundary type for each number
    std::vector<BoundaryType> type;

    // Functions for (u,v,w) for all boundaries
    BoundaryFunction<dim> *bcFunctions;

    // Number of boundary conditions
    unsigned int size;
    unsigned int max_size;

    // Periodic boundary condition matching
    std::vector<unsigned int> periodic_id;
    std::vector<unsigned int> periodic_direction;

    void
    parse_parameters(boost::property_tree::ptree &root);
    void
    createDefaultNoSlip();

  private:
    std::string
    get_default_vnames();
  };

  template <int dim>
  void
  NSBoundaryConditions<dim>::createDefaultNoSlip()
  {
    id.resize(1);
    id[0] = 0;
    type.resize(1);
    type[0] = BoundaryType::noslip;
    size    = 1;
  }

  template <int dim>
  void
  NSBoundaryConditions<dim>::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, BoundaryType> boundaryTypes = {
      {"noslip", BoundaryType::noslip},
      {"slip", BoundaryType::slip},
      {"function", BoundaryType::function},
      {"periodic", BoundaryType::periodic}};

    type.push_back(root.get("type", BoundaryType::noslip, ParameterTranslator(boundaryTypes)));

    std::string default_vnames = get_default_vnames();
    const unsigned int n_components = 1;
    std::string expr;
    for (unsigned int i = 0; i < n_components; i++)
      expr += "; 0";

    switch(type.back())
    {
      case BoundaryType::function:
        // TODO:
        // get: Variable names, Function expression, Function constants
        // bcFunctions.back().u = Functions::ParsedFunction<dim>(var_names, func_expr, func_const)
        break;
    }
  }

  template <int dim>
  std::string
  NSBoundaryConditions<dim>::get_default_vnames()
  {
    std::string vnames;
    switch (dim)
      {
        case 1:
          vnames = "x,t";
          break;
        case 2:
          vnames = "x,y,t";
          break;
        case 3:
          vnames = "x,y,z,t";
          break;
        default:
          AssertThrow(false, ExcNotImplemented());
          break;
      }
    return vnames;
  }
} // namespace BoundaryConditionsJSON

// TODO: when replacing boundary_conditions with JSON, change the following:

#include "boundary_conditions.h"
// template <int dim>
// class FunctionDefined : public Function<dim>
// {
// private:
//   Functions::ParsedFunction<dim> *u;
//   Functions::ParsedFunction<dim> *v;
//   Functions::ParsedFunction<dim> *w;

// public:
//   FunctionDefined(Functions::ParsedFunction<dim> *p_u,
//                   Functions::ParsedFunction<dim> *p_v,
//                   Functions::ParsedFunction<dim> *p_w)
//     : Function<dim>(dim + 1)
//     , u(p_u)
//     , v(p_v)
//     , w(p_w)
//   {}

//   virtual double
//   value(const Point<dim> &p, const unsigned int component) const;
// };

// template <int dim>
// double
// FunctionDefined<dim>::value(const Point<dim> & p,
//                             const unsigned int component) const
// {
//   Assert(component < this->n_components,
//          ExcIndexRange(component, 0, this->n_components));
//   if (component == 0)
//     {
//       return u->value(p);
//     }
//   else if (component == 1)
//     {
//       return v->value(p);
//     }
//   else if (component == 2)
//     {
//       return w->value(p);
//     }
//   return 0.;
// }


#endif
