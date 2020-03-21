#ifndef LETHE_FUNCTIONPARSER_H
#define LETHE_FUNCTIONPARSER_H

#include <deal.II/base/function_parser.h>

#include <boost/property_tree/ptree.hpp>

#include <memory>

template <int dim>
std::shared_ptr<FunctionParser<dim>>
parse_function(boost::property_tree::ptree &root,
               const std::string &          default_vnames,
               const std::string &          expr,
               const std::string &          section)
{
  std::string vnames = root.get(section + ".Variable names", default_vnames);
  std::string expression = root.get(section + ".Function expression", expr);

  std::map<std::string, float>                 constants;
  boost::optional<boost::property_tree::ptree> child =
    root.get_child_optional(section + ".Function constants");
  if (child)
    {
      for (auto it = child->begin(); it != child->end(); ++it)
        {
          constants[it->first] = it->second.data();
        }
    }

  constants["pi"] = numbers::PI;
  constants["Pi"] = numbers::PI;

  auto function_object = std::make_shared<FunctionParser<dim>()>;

  const unsigned int nn = (Utilities::split_string_list(vnames)).size();
  switch (nn)
    {
      case dim:
        // Time independent function
        function_object->initialize(vnames, expression, constants);
        break;
      case dim + 1:
        // Time dependent function
        function_object->initialize(vnames, expression, constants, true);
        break;
      default:
        AssertThrow(false,
                    ExcMessage("The list of variables specified is <" + vnames +
                               "> which is a list of length " +
                               Utilities::int_to_string(nn) +
                               " but it has to be a list of length equal to" +
                               " either dim (for a time-independent function)" +
                               " or dim+1 (for a time-dependent function)."));
    }
  return function_object;
}

#endif