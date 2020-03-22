#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <core/parameters.h>

#include <iostream>
#include <tuple>

namespace
{
  void
  print_verbosity(Parameters::Verbosity verbosity)
  {
    std::cout << "verbosity: "
              << (verbosity == Parameters::Verbosity::quiet ? "quiet" :
                                                              "verbose")
              << '\n';
  }

  void
  print_mesh_type(Parameters::Mesh::Type mesh_type)
  {
    std::string mesh_type_str;
    switch (mesh_type)
      {
        case Parameters::Mesh::Type::dealii:
          mesh_type_str = "dealii";
          break;
        case Parameters::Mesh::Type::gmsh:
          mesh_type_str = "gmsh";
          break;
        case Parameters::Mesh::Type::primitive:
          mesh_type_str = "primitive";
          break;
        default:
          throw std::runtime_error("error printing mesh type");
      }
    std::cout << "type: " << mesh_type_str << '\n';
  }

  void
  print_mesh_primitive_type(Parameters::Mesh::PrimitiveType mesh_primitive_type)
  {
    std::string mesh_primitive_type_str;
    switch (mesh_primitive_type)
      {
        case Parameters::Mesh::PrimitiveType::cylinder:
          mesh_primitive_type_str = "cylinder";
          break;
        case Parameters::Mesh::PrimitiveType::hyper_cube:
          mesh_primitive_type_str = "hyper_cube";
          break;
        case Parameters::Mesh::PrimitiveType::hyper_shell:
          mesh_primitive_type_str = "hyper_shell";
          break;
        default:
          throw std::runtime_error("error printing mesh primitive type");
      }
    std::cout << mesh_primitive_type_str << '\n';
  }

  void
  print_mesh_adaptation_type(
    Parameters::MeshAdaptation::Type mesh_adaptation_type)
  {
    std::string type;
    switch (mesh_adaptation_type)
      {
        case Parameters::MeshAdaptation::Type::none:
          type = "none";
          break;
        case Parameters::MeshAdaptation::Type::kelly:
          type = "kelly";
          break;
        case Parameters::MeshAdaptation::Type::uniform:
          type = "uniform";
          break;
        default:
          throw std::runtime_error("error printing mesh adaptation type");
      }
    std::cout << "type: " << type << '\n';
  }

  void
  print_mesh_adaptation_variable(
    Parameters::MeshAdaptation::Variable mesh_adaptation_variable)
  {
    std::string variable;
    switch (mesh_adaptation_variable)
      {
        case Parameters::MeshAdaptation::Variable::pressure:
          variable = "pressure";
          break;
        case Parameters::MeshAdaptation::Variable::velocity:
          variable = "velocity";
          break;
        default:
          throw std::runtime_error("error printint mesh adapt variable");
      }
    std::cout << "variable: " << variable << '\n';
  }

  void
  print_mesh_adaptation_fraction_type(
    Parameters::MeshAdaptation::FractionType fraction_type)
  {
    std::string type;
    switch (fraction_type)
      {
        case Parameters::MeshAdaptation::FractionType::fraction:
          type = "fraction";
          break;
        case Parameters::MeshAdaptation::FractionType::number:
          type = "number";
          break;
        default:
          throw std::runtime_error("error printing fraction type");
      }
    std::cout << "fraction type: " << type << '\n';
  }
} // namespace

int
main()
{
  try
    {
      std::cout << std::boolalpha;

      boost::property_tree::ptree json_node;

      // Temp path
      boost::property_tree::read_json(
        "/home/simon/Desktop/taylorcouette_gd.json", json_node);

      // TODO: test simulation control

      // FEM
      std::cout << "FEM\n";

      Parameters::FEM femParameters;
      femParameters.parse_parameters(json_node);

      std::cout << "velocityOrder: " << femParameters.velocityOrder << '\n';
      std::cout << "pressureOrder: " << femParameters.pressureOrder << '\n';
      std::cout << "quadraturePoints: " << femParameters.quadraturePoints
                << '\n';
      std::cout << "qmapping_all: " << femParameters.qmapping_all << '\n';
      std::cout << '\n';

      // physical properties
      std::cout << "physicalProperties\n";

      Parameters::PhysicalProperties physicalProperties;
      physicalProperties.parse_parameters(json_node);

      std::cout << "viscosity: " << physicalProperties.viscosity << '\n';
      std::cout << '\n';

      // TODO: test analytical solution

      // forces
      std::cout << "forces\n";

      Parameters::Forces forces;
      forces.parse_parameters(json_node);

      print_verbosity(forces.verbosity);
      std::cout << "calculate_force: " << forces.calculate_force << '\n';
      std::cout << "calculate_torque: " << forces.calculate_torque << '\n';
      std::cout << "calculation_frequency: " << forces.calculation_frequency
                << '\n';
      std::cout << "output_frequency: " << forces.output_frequency << '\n';
      std::cout << "output_precision: " << forces.output_precision << '\n';
      std::cout << "display_precision: " << forces.display_precision << '\n';
      std::cout << "force_output_name: " << forces.force_output_name << '\n';
      std::cout << "torque_output_name: " << forces.torque_output_name << '\n';
      std::cout << '\n';

      // mesh
      std::cout << "mesh\n";

      Parameters::Mesh mesh;
      mesh.parse_parameters(json_node);

      print_mesh_type(mesh.type);
      // primitive type se fait pas lire de la config?
      // print_mesh_primitive_type(mesh.primitiveType);
      // Se passe quoi avec arg1 arg2 etc...
      std::cout << "file_name: " << mesh.file_name << '\n';
      std::cout << "initialRefinement: " << mesh.initialRefinement << '\n';
      std::cout << "grid_type: " << mesh.grid_type << '\n';
      std::cout << "grid_arguments: " << mesh.grid_arguments << '\n';
      std::cout << '\n';

      // TODO: boundary conditions

      // mesh adaptation
      std::cout << "mesh adaptation\n";

      Parameters::MeshAdaptation mesh_adaptation;
      mesh_adaptation.parse_parameters(json_node);
      print_mesh_adaptation_type(mesh_adaptation.type);
      print_mesh_adaptation_variable(mesh_adaptation.variable);
      print_mesh_adaptation_fraction_type(mesh_adaptation.fractionType);
      std::cout << "maxNbElements: " << mesh_adaptation.maxNbElements << '\n';
      std::cout << "maxRefLevel: " << mesh_adaptation.maxRefLevel << '\n';
      std::cout << "minRefLevel: " << mesh_adaptation.minRefLevel << '\n';
      std::cout << "frequency: " << mesh_adaptation.frequency << '\n';
      std::cout << "fractionRefinement: " << mesh_adaptation.fractionRefinement
                << '\n';
      std::cout << "fractionCoarsening: " << mesh_adaptation.fractionCoarsening
                << '\n';
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
}
