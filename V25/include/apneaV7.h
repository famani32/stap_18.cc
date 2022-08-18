/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

 #ifndef Apnea_V8_include_file
 #define Apnea_V8_include_file
 #include <deal.II/base/conditional_ostream.h>
 #include <deal.II/base/function.h>
 #include <deal.II/base/index_set.h>
 #include <deal.II/base/quadrature_lib.h>
 #include <deal.II/base/timer.h>

 #include <deal.II/distributed/grid_refinement.h>
 #include <deal.II/distributed/tria.h>

 #include <deal.II/dofs/dof_handler.h>
 #include <deal.II/dofs/dof_tools.h>

 #include <deal.II/fe/fe_q.h>
 #include <deal.II/fe/fe_values.h>

 #include <deal.II/grid/grid_generator.h>
 #include <deal.II/grid/grid_out.h>
 #include <deal.II/grid/tria.h>

 #include <deal.II/base/parameter_acceptor.h>
 #include <deal.II/lac/affine_constraints.h>
 #include <deal.II/lac/dynamic_sparsity_pattern.h>
 #include <deal.II/lac/full_matrix.h>
 #include <deal.II/lac/generic_linear_algebra.h>
 #include <deal.II/lac/precondition.h>
 #include <deal.II/lac/solver_cg.h>
 #include <deal.II/lac/sparse_matrix.h>
 #include <deal.II/lac/sparsity_tools.h>
 #include <deal.II/lac/vector.h>

 #include <deal.II/grid/grid_in.h>
 #include <deal.II/grid/grid_out.h>
 #include <deal.II/lac/linear_operator.h>
 #include <deal.II/lac/linear_operator_tools.h>
 #include <deal.II/numerics/data_out.h>
 #include <deal.II/numerics/matrix_tools.h>
 #include <deal.II/numerics/vector_tools.h>
 #include <fstream>
 #include <iostream>
//#include "../../utils/progress-cpp/include/progresscpp/ProgressBar.hpp"

#define FORCE_USE_OF_TRILINOS
namespace LA {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) &&     \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

  using namespace dealii;

  // Initialisation of the first model for apnea
  template <int dim>
  class Apnea_V8
  {
  public:
    Apnea_V8();
    void run(std::string force);
  private:
    void setup_system();
    void assemble_system(double force);
    void solve();
    void move_mesh();
    void move_mesh_backward();
    void output_results(double force) const;
    void output_results_with_move(double force) const;
    void get_grid();
    void write_info_solution(const unsigned int    iteration,
			     const double          , //check_value
			     const Vector<double> &current_iterate) const;


    MPI_Comm mpi_communicator;

   IndexSet locally_relevant_dofs;
   IndexSet locally_owned_dofs;
  //  Triangulation<dim> triangulation;
  parallel::distributed::Triangulation<dim> triangulation;

    DoFHandler<dim>    dof_handler;
    //FESystem<dim> fe;
    FE_Q<dim> fe;

    AffineConstraints<double> constraints;

   SparsityPattern      sparsity_pattern;

  /*  SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;*/

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector locally_relevant_solution;
    LA::MPI::Vector system_rhs;
    LA::MPI::Vector solution;



    double       present_time;
    double       present_timestep;
    double       end_time;
    unsigned int timestep_no;


    const QGauss<dim> quadrature_formula;
  };

  #endif
