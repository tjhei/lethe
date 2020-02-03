#include "core/parameters_json.h"

#include "core/parameter_translator.h"

namespace ParametersJson
{
  void
  SimulationControl::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, TimeSteppingMethod> methods = {
      {"steady", TimeSteppingMethod::steady},
      {"bdf1", TimeSteppingMethod::steady},
      {"bdf2", TimeSteppingMethod::bdf1},
      {"bdf3", TimeSteppingMethod::bdf2},
      {"bdf3", TimeSteppingMethod::bdf3},
      {"sdirk2", TimeSteppingMethod::sdirk2},
      {"sdirk3", TimeSteppingMethod::sdirk3},
    };

    // auto node = root.get_child("simulation control");

    method  = root.get("simulation control.method",
                      TimeSteppingMethod::steady,
                      ParameterTranslator<TimeSteppingMethod>(methods));
    dt      = root.get("simulation control.time step", 1.0);
    timeEnd = root.get("simulation control.time end", 1.0);
    adapt   = root.get("simulation control.adapt", false);
    maxCFL  = root.get("simulation control.max cfl", 1.0);
    startup_timestep_scaling =
      root.get("simulation control.startup time scaling", 0.4);
    nbMeshAdapt     = root.get("simulation control.number mesh adapt", 0);
    output_folder   = root.get("simulation control.output path", "./");
    output_name     = root.get("simulation control.output name", "out");
    outputFrequency = root.get("simulation control.output frequency", 1);
    subdivision     = root.get("simulation control.subdivision", 1);
    group_files     = root.get("simulation control.group files", 1);
  }

  void
  PhysicalProperties::parse_parameters(boost::property_tree::ptree &root)
  {
    viscosity = root.get("physical properties.kinematic viscosity", 1.0);
  }

  void
  Timer::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, Type> types = {
      {"none", Type::none}, {"iteration", Type::iteration}, {"end", Type::end}};
    type = root.get("timer.type", Type::none, ParameterTranslator<Type>(types));
  }

  void
  Forces::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, Verbosity> verbosities = {
      {"verbose", Verbosity::verbose},
      {"quiet", Verbosity::quiet},
    };
    verbosity             = root.get("forces.verbosity",
                         Verbosity::quiet,
                         ParameterTranslator<Verbosity>(verbosities));
    calculate_force       = root.get("forces.calculate forces", false);
    calculate_torque      = root.get("forces.calculate torques", false);
    force_output_name     = root.get("forces.force name", "force");
    torque_output_name    = root.get("forces.torque name", "torque");
    output_precision      = root.get("forces.output precision", 10);
    display_precision     = root.get("forces.display precision", 6);
    calculation_frequency = root.get("forces.calculation frequency", 1);
    output_frequency      = root.get("forces.output frequency", 1);
  }

} // namespace ParametersJson
