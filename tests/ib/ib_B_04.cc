
//BASE
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

//NUMERICS
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

//GRID
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>

//LACS
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

//DOFS
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


// Distributed
#include <deal.II/distributed/tria.h>

#include "ibcomposer.h"
#include "ibcombiner.h"
#include "iblevelsetfunctions.h"
#include "write_data.h"
#include "../tests.h"

// Triangles decomposition tools
#include "nouvtriangles.h"
#include "condensate.h"

using namespace dealii;


void integrate_sub_quad_element( Triangulation<2> &sub_triangulation,  DoFHandler<2> &dof_handler,FESystem<2> &fe,  FullMatrix<double> &system_matrix, Vector<double> &system_rhs)
{
  // Create a mapping for this new element
  const MappingQ<2>      mapping (1);
  QGauss<2>              quadrature_formula(4);
  std::map< types::global_dof_index, Point< 2 > > support_points;
  DoFTools::map_dofs_to_support_points ( mapping, dof_handler,support_points );

  // Integrate over this element, in this case we only integrate
  // over the quadrature to calculate the area
  FEValues<2> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<Point<2> >               dofs_points(dofs_per_cell);        // Array for the DOFs points
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell


  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  typename DoFHandler<2>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    cell_matrix = 0;
    cell_rhs = 0;

//    std::cout << "\n pts : " << support_points[0] << ", " << support_points[1] << ", " << support_points[2] << ", " << support_points[3] << ", " << support_points[4] << ", " << support_points[5] << ", " << support_points[6] << std::endl;

    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          //Stiffness Matrix
          cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                               fe_values.shape_grad (j, q_index) *
                               fe_values.JxW (q_index));

          // Right Hand Side
          cell_rhs(i) += 0.;
      }
    }
    // Assemble global matrix and RHS
    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
  }
}

void heat_integrator(int refinement_level,   std::vector<IBLevelSetFunctions<2> *> ib_functions, std::string output_name)
{
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);
  unsigned int n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator));
  unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

  // Create triangulation and square mesh
  parallel::distributed::Triangulation<2> triangulation (mpi_communicator, typename Triangulation<2>::MeshSmoothing
                                                         (Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening));
  GridGenerator::hyper_cube (triangulation,
                             -1,1,true);
  triangulation.refine_global(refinement_level);

  DoFHandler<2>                  dof_handler(triangulation);
  FESystem<2>                    fe(FE_Q<2>(1),1);
  dof_handler.distribute_dofs(fe);

  // Set-up global system of equation
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);

  // Initialize vector and sparsity patterns
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  // Quadrature formula for the element
  QGauss<2>            quadrature_formula(3);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points = quadrature_formula.size();

  // Matrix and RHS sides;
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  // Get the position of the support points
  const MappingQ<2>      mapping (1);

  FEValues<2> fe_values (mapping,
                         fe,
                         quadrature_formula,
                         update_values   | update_gradients |
                         update_quadrature_points | update_JxW_values);

  std::map< types::global_dof_index, Point< 2 > > support_points;
  DoFTools::map_dofs_to_support_points ( mapping, dof_handler,support_points );

  // Instantiations for the decomposition of the elements
  std::vector<int>                     corresp(9);
  std::vector<Point<2> >               decomp_elem(9);         // Array containing the points of the new elements created by decomposing the elements crossed by the boundary fluid/solid, there are up to 9 points that are stored in it
  std::vector<node_status>             No_pts_solid(4);
  int                                  nb_poly;                   // Number of sub-elements created in the fluid part for each element ( 0 if the element is entirely in the solid or the fluid)
  std::vector<Point<2> >               num_elem(6);

  // Set the values of the dof points position
  std::vector<double>                  distance(dofs_per_cell); // Array for the distances associated with the DOFS
  std::vector<Point<2> >               dofs_points(dofs_per_cell);// Array for the DOFs points
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); // Global DOFs indices corresponding to cell

  IBCombiner<2>  ib_combiner(ib_functions);

  typename DoFHandler<2>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    cell->get_dof_indices (local_dof_indices);

    for (unsigned int dof_index=0 ; dof_index < local_dof_indices.size() ; ++dof_index)
    {
      dofs_points[dof_index] = support_points[local_dof_indices[dof_index]];
      distance[dof_index]    = ib_combiner.value(dofs_points[dof_index]);
    }

    // Decompose the geometry
    decomposition(corresp, No_pts_solid, num_elem, decomp_elem, &nb_poly, dofs_points, distance);

    if (nb_poly==0 && (distance[0]>0))
    {
      fe_values.reinit(cell);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              //Stiffness Matrix
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));

              // Right Hand Side
              cell_rhs(i) += 0.;
          }
        }

    }
    if(nb_poly==0 && (distance[0]<0))
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        cell_matrix(i,i)=1.;
        cell_rhs(i)=1;
      }
    }
    if (nb_poly==-1)
    {
      // Create triangulation points
      std::vector<Point<2> > triangulation_points(GeometryInfo<2>::vertices_per_cell);
      // Create 4 points for triangulation:
      for (unsigned int i_pt =0 ; i_pt < 4 ; ++i_pt)
        triangulation_points[i_pt]=decomp_elem[i_pt];

      // Prepare cell data
      std::vector<CellData<2> > cells (1);
      for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i)
          cells[0].vertices[i] = i;
      cells[0].material_id = 0;

      Triangulation<2> sub_triangulation;
      sub_triangulation.create_triangulation (triangulation_points, cells, SubCellData());

      // Create a FE system for this element
      DoFHandler<2>                  sub_dof_handler(sub_triangulation);
      FESystem<2>                    sub_fe(FE_Q<2>(1),1);
      sub_dof_handler.distribute_dofs(sub_fe);

      FullMatrix<double> sub_system_matrix;
      Vector<double>     sub_system_rhs;
      // Initialize vector and sparsity patterns
      sub_system_matrix.reinit (sub_dof_handler.n_dofs(),sub_dof_handler.n_dofs());
      sub_system_rhs.reinit (sub_dof_handler.n_dofs());
      integrate_sub_quad_element(sub_triangulation, sub_dof_handler, sub_fe, sub_system_matrix, sub_system_rhs);

      // Condensate
      for (unsigned int i=0; i<4; ++i)
      {
        if (corresp[i]>3)
        {
          for (unsigned int j=0; j<4; ++j)
            if (i!=j) sub_system_matrix(i,j)=0.;
            else sub_system_matrix(i,i)=1.;
          sub_system_rhs[i]=1;
        }
        else
        {
          for (unsigned int j=0; j<4; ++j)
          {
            if (corresp[j]>3)
            {
              sub_system_rhs(i) -= sub_system_matrix(i,j) * 1.;
              sub_system_matrix(i,j)=0.;
            }
          }
        }
      }

      // Copy in larger matrix
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<4; ++i)
      {
        if(corresp[i]<3)
        {
          for (unsigned int j=0; j<4; ++j)
          {
            if(corresp[j]<3)
            {
              system_matrix.add (local_dof_indices[corresp[i]],
                                 local_dof_indices[corresp[j]],
                                 sub_system_matrix(i,j));
            }
          }
          system_rhs(local_dof_indices[corresp[i]]) += sub_system_rhs(i);
        }

      }
    }
    if (nb_poly==1)
    {

      // Create triangulation points
      std::vector<Point<2> > triangulation_points(3);
      // Create 4 random points:
      for (unsigned int i_pt =0 ; i_pt < 3 ; ++i_pt)
      {
        triangulation_points[i_pt]=decomp_elem[i_pt];
      }
      Triangulation<2> sub_triangulation;
      GridGenerator::simplex(sub_triangulation,triangulation_points);
      // Create a FE system for this element
      DoFHandler<2>                  sub_dof_handler(sub_triangulation);
      FESystem<2>                    sub_fe(FE_Q<2>(1),1);
      sub_dof_handler.distribute_dofs(sub_fe);


      FullMatrix<double> sub_system_matrix;
      Vector<double>     sub_system_rhs;

      // Initialize vector and sparsity patterns
      sub_system_matrix.reinit (sub_dof_handler.n_dofs(),sub_dof_handler.n_dofs());
      sub_system_rhs.reinit    (sub_dof_handler.n_dofs());

      integrate_sub_quad_element(sub_triangulation, sub_dof_handler, sub_fe, sub_system_matrix, sub_system_rhs);

      for (int i = 0; i < 7; ++i) {
          sub_system_matrix(2,i)=0.;
          sub_system_matrix(1,i)=0.;
          sub_system_matrix(6,i)=0.;
          sub_system_matrix(5,i)=0.;
          sub_system_matrix(4,i)=0.;
      }

      sub_system_matrix(5,5)=1.;
      sub_system_matrix(4,4)=1.;
      sub_system_matrix(6,6)=1.;

      sub_system_matrix(1,1)=1.;
      sub_system_matrix(2,2)=1.;

      sub_system_matrix(2,0)=-0.5;
      sub_system_matrix(2,6)=-0.5;
      sub_system_matrix(1,0)=-0.5;
      sub_system_matrix(1,4)=-0.5;

      sub_system_rhs[1]=0.;
      sub_system_rhs[2]=0.;

      sub_system_rhs[6]=1.;
      sub_system_rhs[5]=1.;
      sub_system_rhs[4]=1.;

      FullMatrix<double>                Mat_sorted(sub_dof_handler.n_dofs(),sub_dof_handler.n_dofs());
      Vector<double>               rhs_sorted(sub_dof_handler.n_dofs());
      std::vector<int>                  change_coor = {0, 4, 6, 1, 2, 3, 5};
      for (unsigned int i = 0; i < sub_dof_handler.n_dofs(); ++i) {
          for (unsigned int j = 0; j < sub_dof_handler.n_dofs(); ++j) {
              Mat_sorted(i,j) = sub_system_matrix(change_coor[i], change_coor[j]);
          }
          rhs_sorted[i] = sub_system_rhs[change_coor[i]];
      }

      FullMatrix<double>                mat(1,1);
      Vector<double>               rhs(1);

      condensate(sub_dof_handler.n_dofs(), 1, Mat_sorted, mat, rhs_sorted, rhs);

      cell_matrix(corresp[0],corresp[0]) = mat(0,0);

      cell_rhs[corresp[0]]=rhs[0];

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          if (No_pts_solid[i]){
              cell_matrix(i,i)=1;
              cell_rhs(i)=1;
          }
      }

    }

    if (nb_poly==3)
    {
      //Create new triangulation and integrate

      FullMatrix<double>        mat6(6,6);
      Vector<double>            rhs6(6);
      cell_matrix=0;
      cell_rhs=0;
      mat6=0;
      rhs6=0.;

      for (int sub_element = 0 ; sub_element<nb_poly ; ++sub_element)
      {

        // Create triangulation points
        std::vector<Point<2> > triangulation_points(3);
        // Create 4 random points:
        for (unsigned int i_pt =0 ; i_pt < 3 ; ++i_pt)
        {
          triangulation_points[i_pt]=decomp_elem[3*sub_element+i_pt];
        }
        Triangulation<2> sub_triangulation;
        GridGenerator::simplex(sub_triangulation,triangulation_points);
        // Create a FE system for this element
        DoFHandler<2>                  sub_dof_handler(sub_triangulation);
        FESystem<2>                    sub_fe(FE_Q<2>(1),1);
        sub_dof_handler.distribute_dofs(sub_fe);

        FullMatrix<double> sub_system_matrix;
        Vector<double>     sub_system_rhs;

        // Initialize vector and sparsity patterns
        sub_system_matrix.reinit (sub_dof_handler.n_dofs(),sub_dof_handler.n_dofs());
        sub_system_rhs.reinit (sub_dof_handler.n_dofs());

        sub_system_matrix=0;
        sub_system_rhs=0;

        integrate_sub_quad_element(sub_triangulation, sub_dof_handler, sub_fe, sub_system_matrix, sub_system_rhs);

            for (int i = 0; i < 7; ++i) {
                sub_system_matrix(2,i)=0.;
                sub_system_matrix(1,i)=0.;
                sub_system_matrix(5,i)=0.;
            }

            sub_system_matrix(1,1)=1.;
            sub_system_matrix(2,2)=1.;
            sub_system_matrix(5,5)=1.;

            sub_system_matrix(2,0)=-0.5;
            sub_system_matrix(2,6)=-0.5;
            sub_system_matrix(1,0)=-0.5;
            sub_system_matrix(1,4)=-0.5;
            sub_system_matrix(5,6)=-0.5;
            sub_system_matrix(5,4)=-0.5;

            sub_system_rhs[1]=0.;
            sub_system_rhs[2]=0.;
            sub_system_rhs[5]=0.;

        FullMatrix<double>                Mat_sorted(sub_dof_handler.n_dofs(),sub_dof_handler.n_dofs());
        Vector<double>               rhs_sorted(sub_dof_handler.n_dofs());
        std::vector<int>                  change_coor = {0, 4, 6, 1, 2, 3, 5};

        for (unsigned int i = 0; i < sub_dof_handler.n_dofs(); ++i) {
            for (unsigned int j = 0; j < sub_dof_handler.n_dofs(); ++j) {
                Mat_sorted(i,j) = sub_system_matrix(change_coor[i], change_coor[j]);
            }
            rhs_sorted[i] = sub_system_rhs[change_coor[i]];
        }

        FullMatrix<double>                mat3(3,3);
        Vector<double>               rhs3(3);
        mat3=0;
        rhs3=0.;
        condensate(sub_dof_handler.n_dofs(), 3, Mat_sorted, mat3, rhs_sorted, rhs3);

        for (unsigned int i = 0; i < 3; ++i) {
            for (unsigned int j = 0; j < 3; ++j) {
                mat6(corresp[3*sub_element + i],corresp[3*sub_element + j]) += mat3(i,j);
            }
            rhs6[corresp[3*sub_element + i]]+=rhs3[i];
        }
      }

      for (int i = 0; i < 6; ++i) {
          mat6(4,i)=0;
          mat6(5,i)=0;
      }
      mat6(4,4)=1;
      mat6(5,5)=1;
      rhs6[4]=1;
      rhs6[5]=1;

      Vector<double>       copy_cell_rhs(4);

      condensate(6,4,mat6,cell_matrix,rhs6,copy_cell_rhs);

      for (unsigned int i = 0; i<dofs_per_cell ; ++i) {
          cell_rhs[i] = copy_cell_rhs[i];
      }
    }

    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        if (No_pts_solid[i])
        {
            cell_matrix(i,i)=1;
            cell_rhs(i)=1;
        }
    }

    // Assemble global matrix and RHS
    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
  }

  std::map<types::global_dof_index,double> boundary_values;

//  boundary_values[11]=0.;

  VectorTools::interpolate_boundary_values (dof_handler,
                                            1,
                                            Functions::ZeroFunction<2>(),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

  // Solve System
  SolverControl           solver_control (100000, 1e-10);
  SolverGMRES<>           solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());


  // Data output
  DataOut<2> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output ("solution.gpl");
  data_out.write_gnuplot (output);

//  std::string fname=output_name+"-Scalar"+".vtk";
//  std::ofstream output (fname.c_str());
//  data_out.write_vtk (output);

}

void square()
{
  // Generate the IB composer
  Point<2> center1(0.51,0);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=1; normal[1]=0;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  heat_integrator(3, ib_functions,"IB_B_03_square");
}

void cut_square()
{
  // Generate the IB composer
  Point<2> center1(0.04,0);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,2> normal;
  normal[0]=1; normal[1]=-1;
  double T_scal=1;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetPlane<2> plane(center1, normal,velocity, T_scal);
  ib_functions.push_back(&plane);
  heat_integrator(2, ib_functions,"IB_B_03_cut_square");
}

// Square with a hole inside
void square_hole()
{
  // Generate the IB composer
  Point<2> center1(0,0);
  Tensor<1,2> velocity;
  velocity[0]=1.; velocity[1]=0.;
  Tensor<1,3> angular;
  angular[0]=0; angular[1]=0; angular[2]=0;
  double radius =0.47; double T_scal=1; bool inside=0;

  // IB composer
  std::vector<IBLevelSetFunctions<2> *> ib_functions;
  // Add a shape to it
  IBLevelSetCircle<2> circle1(center1,velocity,angular, T_scal, inside, radius);
  ib_functions.push_back(&circle1);
  for (int i=3 ; i<3 ; ++i)
  {
    heat_integrator(i, ib_functions,"IB_B_03_circle");
  }
}

int
main(int argc, char* argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    initlog();

    //square();
    //cut_square();
    //square_hole();


  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what()  << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
