#include "core/parameters_json.h"

#include "core/parameter_translator.h"

namespace
{
  const std::unordered_map<std::string, ParametersJson::Verbosity> verbosities =
    {{"verbose", ParametersJson::Verbosity::verbose},
     {"quiet", ParametersJson::Verbosity::quiet}};
} // namespace

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

    method  = root.get("simulation control.method",
                      TimeSteppingMethod::steady,
                      ParameterTranslator(methods));
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
    type = root.get("timer.type", Type::none, ParameterTranslator(types));
  }

  void
  Forces::parse_parameters(boost::property_tree::ptree &root)
  {
    verbosity = root.get("forces.verbosity",
                         Verbosity::quiet,
                         ParameterTranslator(verbosities));

    calculate_force       = root.get("forces.calculate forces", false);
    calculate_torque      = root.get("forces.calculate torques", false);
    force_output_name     = root.get("forces.force name", "force");
    torque_output_name    = root.get("forces.torque name", "torque");
    output_precision      = root.get("forces.output precision", 10);
    display_precision     = root.get("forces.display precision", 6);
    calculation_frequency = root.get("forces.calculation frequency", 1);
    output_frequency      = root.get("forces.output frequency", 1);
  }

  void
  PostProcessing::parse_parameters(boost::property_tree::ptree &root)
  {
    verbosity = root.get("post-processing.verbosity",
                         Verbosity::quiet,
                         ParameterTranslator(verbosities));

    calculate_kinetic_energy =
      root.get("post-pocessing.calculate kinetic energy", false);
    calculate_enstrophy = root.get("post-pocessing.calculate enstrophy", false);
    kinetic_energy_output_name =
      root.get("post-pocessing.kinetic energy name", "kinetic_energy");
    enstrophy_output_name =
      root.get("post-pocessing.enstrophy name", "enstrophy");
    calculation_frequency = root.get("post-pocessing.calculation frequency", 1);
    output_frequency      = root.get("post-pocessing.output frequency", 1);
  }

  void
  FEM::parse_parameters(boost::property_tree::ptree &root)
  {
    velocityOrder    = root.get("FEM.velocity order", 1);
    pressureOrder    = root.get("FEM.pressure order", 1);
    quadraturePoints = root.get("FEM.quadrature points", 0);
    qmapping_all     = root.get("FEM.qmapping all", false);
  }

  void
  NonLinearSolver::parse_parameters(boost::property_tree::ptree &root)
  {
    verbosity = root.get("non-linear solver.verbosity",
                         Verbosity::quiet,
                         ParameterTranslator(verbosities));

    std::unordered_map<std::string, SolverType> solverTypes = {
      {"newton", SolverType::newton}, {"skip_newton", SolverType::skip_newton}};

    solver = root.get("non-linear solver.solver",
                      SolverType::newton,
                      ParameterTranslator(solverTypes));

    tolerance         = root.get("non-linear solver.tolerance", 1e-6);
    max_iterations    = root.get("non-linear solver.max iterations", 10);
    skip_iterations   = root.get("non-linear solver.skip iterations", 1);
    display_precision = root.get("non-linear solver.residual precision", 4);
  }

  void
  LinearSolver::parse_parameters(boost::property_tree::ptree &root)
  {
    verbosity = root.get("linear solver.verbosity",
                         Verbosity::quiet,
                         ParameterTranslator(verbosities));

    std::unordered_map<std::string, SolverType> solverTypes = {
      {"amg", SolverType::amg},
      {"gmres", SolverType::gmres},
      {"bicgstab", SolverType::bicgstab}};

    solver = root.get("linear solver.method",
                      SolverType::gmres,
                      ParameterTranslator<SolverType>(solverTypes));

    residual_precision = root.get("linear solver.residual precision", 6);
    relative_residual  = root.get("linear solver.relative residual", 1e-3);
    minimum_residual   = root.get("linear solver.minimum residual", 1e-8);
    max_iterations     = root.get("linear solver.max iters", 1000);
    ilu_precond_fill   = root.get("linear solver.ilu preconditioner fill", 1.0);
    ilu_precond_atol =
      root.get("linear solver.ilu preconditioner absolute tolerance", 1e-6);
    ilu_precond_rtol =
      root.get("linear solver.ilu preconditioner relative tolerance", 1.0);
    amg_precond_ilu_fill =
      root.get("linear solver.amg preconditioner ilu fill", 1.0);
    amg_precond_ilu_atol =
      root.get("linear solver.amg preconditioner ilu absolute tolerance",
               1e-12);
    amg_precond_ilu_rtol =
      root.get("linear solver.amg preconditioner ilu relative tolerance", 1.0);
    amg_aggregation_threshold =
      root.get("linear solver.amg aggregation threshold", 1e-14);
    amg_n_cycles         = root.get("linear solver.amg n cycles", 1);
    amg_w_cycles         = root.get("linear solver.amg w cycles", false);
    amg_smoother_sweeps  = root.get("linear solver.amg smoother sweeps", 2);
    amg_smoother_overlap = root.get("linear solver.amg smoother overlap", 1);
  }

  void
  Mesh::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, Type> types = {{"gmsh", Type::gmsh},
                                                   {"dealii", Type::dealii}};
    type      = root.get("mesh.type", Type::dealii, ParameterTranslator(types));
    file_name = root.get("mesh.file name", "none");
    initialRefinement = root.get("mesh.initial refinement", 0);
    grid_type         = root.get("mesh.grid type", "hyper_cube");
    grid_arguments    = root.get("mesh.grid arguments", "-1 : 1 : false");
  }

  void
  MeshAdaptation::parse_parameters(boost::property_tree::ptree &root)
  {
    std::unordered_map<std::string, Type> types = {{"none", Type::none},
                                                   {"uniform", Type::uniform},
                                                   {"kelly", Type::kelly}};
    type = root.get("mesh adaptation.type",
                    Type::none,
                    ParameterTranslator<Type>(types));

    std::unordered_map<std::string, Variable> variables = {
      {"velocity", Variable::velocity}, {"pressure", Variable::pressure}};
    variable = root.get("mesh adaptation.variable",
                        Variable::velocity,
                        ParameterTranslator(variables));

    std::unordered_map<std::string, FractionType> fractionTypes = {
      {"number", FractionType::number}, {"fraction", FractionType::fraction}};
    fractionType = root.get("mesh adaptation.fraction type",
                            FractionType::number,
                            ParameterTranslator(fractionTypes));

    maxNbElements = root.get("mesh adaptation.max number elements", 100000000);
    maxRefLevel = root.get("mesh adaptation.max refinement level", 10);
    minRefLevel = root.get("mesh adaptation.min refinement level", 0);
    frequency = root.get("mesh adaptation.frequency", 1);
    fractionCoarsening = root.get("mesh adaptation.fraction coarsening", 0.05);
    fractionRefinement = root.get("mesh adaptation.fraction refinement", 0.1);
  }

  void
  Testing::parse_parameters(boost::property_tree::ptree &root)
  {
    enabled = root.get("test.enable", false);
  }

  void
  Restart::parse_parameters(boost::property_tree::ptree &root)
  {
    filename = root.get("restart.filename", "restart");
    checkpoint = root.get("restart.checkpoint", false);
    restart = root.get("restart.restart", false);
    frequency = root.get("restart.frequency", 1);
  }
} // namespace ParametersJson
