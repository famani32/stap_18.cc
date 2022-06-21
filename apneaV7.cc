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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <fstream>
#include <iostream>
#include <numbers>
#include <cmath>
#include "progresscpp/ProgressBar.hpp"

namespace Apnea
{
  using namespace dealii;

  // Initialisation of the first model for apnea
  template <int dim>
  class Apnea_V7
  {
  public:
    Apnea_V7();
    //void run(std::string force);
    //void run();
    void run(std:: string force);
  private:
    void setup_system();
    void assemble_system(double force);
    void solve();
    //void do_timestep(std:: string force);
    //void run(std:: string force);
    //void output_results(double force) const;
    void output_results();
    void output_results_with_move(double force) const;
    void get_grid();
    void write_info_solution(const unsigned int    iteration,
			     const double          , //check_value
			     const Vector<double> &current_iterate) const;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    FESystem<dim> fe;
    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    double       present_time;
    double       present_timestep;
    double       end_time;
    unsigned int timestep_no;


    const QGauss<dim> quadrature_formula;
  };

    // Initiate the right_hand_side vector (represente the force)
  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &  values)
  {

    // Check on the size of the vector values and the vector of points
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    // Check on the dimension of the computation
    Assert(dim >= 2, ExcNotImplemented());

    // For each point of the vector, the force is set to (0, 0) or (0, 0, 0)
    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
        for (unsigned int d = 0; d < dim; ++d)
        {
          values[point_n][d] = 0.0;
        }
      }
  }

  // Initiate the water pressure (Neumann condition)
  template <int dim>
  void diaphragm_pressure(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &  values,
                       double force, double present_time)
  {

    double pi=M_PI;
    // Check on the size of the vector values and the vector of points
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    // Check on the dimension of the computation
    Assert(dim >= 2, ExcNotImplemented());

    // For each point of the vector, the pressure is set to of each coordinate is -2 bar
    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
        for (unsigned int d = 0; d < dim; ++d)
        {
          values[point_n][d] = force * -100000.0*(std::sin(2*pi*0.25*present_time));
        }
      }
  }

  // Define the DoF, FE and Quadrature formula for the model
  template <int dim>
  Apnea_V7<dim>::Apnea_V7()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , quadrature_formula(fe.degree + 1)
    , present_time(0.0)
    , present_timestep(0.25)
    , end_time(4.25)
    , timestep_no(0)
  {}

  // Get the mesh for computation
  template <int dim>
  void Apnea_V7<dim>:: get_grid()
  {
    std::string filename = "mesh/lung_saved.msh";
    std::cout << filename << std::endl;
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(filename);
    gridin.read_msh(f);
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell->set_material_id(0);
      for (const auto &face : cell->face_iterators())
      {
        if ((face->at_boundary()) && (face->boundary_id() == 1))
        {
          cell->set_material_id(1);
        }
      }
    }
  }

  // Function to setup the vector and matrix of the system
  template <int dim>
  void Apnea_V7<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    constraints.condense(dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  // Function to assemble the vector and the matrix of the system
  template <int dim>
  void Apnea_V7<dim>::assemble_system(double force)
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim-1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                 face_quadrature_formula,
                                 update_values | update_quadrature_points |
                                   update_normal_vectors |
                                   update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> lambda_lung_values(n_q_points);
    std::vector<double> mu_lung_values(n_q_points);
    Functions::ConstantFunction<dim> lambda_lung(1794.285714), mu_lung(448.571428);
    std::vector<double> lambda_rib_values(n_q_points);
    std::vector<double> mu_rib_values(n_q_points);
    Functions::ConstantFunction<dim> lambda_rib(6923076923.0769), mu_rib(4615384615.3846);
    std::vector<Tensor<1, dim>> rhs_values(n_q_points);
    std::vector<Tensor<1, dim>> pressure_values(n_face_q_points);

    ProgressBar progressBar(triangulation.n_active_cells(), 70);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
       ++progressBar;
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        lambda_lung.value_list(fe_values.get_quadrature_points(), lambda_lung_values);
        mu_lung.value_list(fe_values.get_quadrature_points(), mu_lung_values);
        lambda_rib.value_list(fe_values.get_quadrature_points(), lambda_rib_values);
        mu_rib.value_list(fe_values.get_quadrature_points(), mu_rib_values);
        right_hand_side(fe_values.get_quadrature_points(), rhs_values);

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
            for (const unsigned int j : fe_values.dof_indices())
              {
                const unsigned int component_j = fe.system_to_component_index(j).first;
                for (const unsigned int q_point : fe_values.quadrature_point_indices())
                  {
                  //  if (cell->material_id() == 1)
                    //{
                      cell_matrix(i, j) +=
                      (
                        (fe_values.shape_grad(i, q_point)[component_i] *
                         fe_values.shape_grad(j, q_point)[component_j] *
                         lambda_lung_values[q_point])
                        +
                        (fe_values.shape_grad(i, q_point)[component_j] *
                         fe_values.shape_grad(j, q_point)[component_i] *
                         mu_lung_values[q_point])
                        +
                        ((component_i == component_j) ?
                           (fe_values.shape_grad(i, q_point) *
                            fe_values.shape_grad(j, q_point) *
                            mu_lung_values[q_point]) :
                           0)
                        ) *
                      fe_values.JxW(q_point);
                  }
              }
          }

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
              cell_rhs(i) += fe_values.shape_value(i, q_point) *
                             rhs_values[q_point][component_i] *
                             fe_values.JxW(q_point);
          }

        for (const auto &face : cell->face_iterators())
          {
            if (face->at_boundary())
              {
                fe_face_values.reinit(cell, face);
                diaphragm_pressure(fe_face_values.get_quadrature_points(), pressure_values, force, present_time);
                for (const unsigned int q_point : fe_face_values.quadrature_point_indices())
                  {
                    const double neumann_value =
                      (pressure_values[q_point] *
                      fe_face_values.normal_vector(q_point));

                    for (const unsigned int i : fe_face_values.dof_indices())
                      {
                        cell_rhs(i) +=
                          (fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                          neumann_value *                          // g(x_q)
                          fe_face_values.JxW(q_point));            // dx

                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        progressBar.display();
      };
      progressBar.done();
  }

  template <int dim>
  void Apnea_V7<dim>::write_info_solution (const unsigned int    iteration,
							   const double          , //check_value
							   const Vector<double> &current_iterate) const
  {
    std::cout << iteration << std::endl;

  }


  template <int dim>
  void Apnea_V7<dim>::solve()
  {
    std::cout << "      Norm of rhs: " << system_rhs.l2_norm() << std::endl;
    std::cout << "      Nb step max = " << solution.size() << std::endl;

    SolverControl            solver_control(solution.size(), 1e-8 * system_rhs.l2_norm());

    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;

    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
    // std::cout << solution[0] << std::endl;
    std::cout << "      Solver converged: " << solver_control.last_step() << " iterations." << std::endl;
  }

  template <int dim>
  void Apnea_V7<dim>::output_results()
  //void Apnea_V7<dim>::output_results()
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 1:
          solution_names.emplace_back("displacement");
          break;
        case 2:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          break;
        case 3:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          solution_names.emplace_back("z_displacement");
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();

    std::vector<std::string> filenames;
    const std::string pvtu_master_filename =
    ("solution-" + Utilities::int_to_string(timestep_no, 4) + ".pvtu");
    std::ofstream pvtu_master(pvtu_master_filename);
    data_out.write_pvtu_record(pvtu_master, filenames);
      std::cout << "hmar"<< std::endl;
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back(
    std::pair<double, std::string>(present_time, pvtu_master_filename));
    std::ofstream pvd_output("solution.pvd");

    DataOutBase::write_pvd_record(pvd_output, times_and_names);


    /*if (dim == 2)
      {
        std::ofstream output("solution_2D.vtk");
        data_out.write_vtk(output);
      }
    else if (dim == 3)
      {
        std::ofstream output(filename+".vtk");
        data_out.write_vtk(output);
      }*/
  }


  template <int dim>
  //void Apnea_V7<dim>::do_timestep(std::string force)
  void Apnea_V7<dim>::run(std::string force)
  {
    std::cout << "   Start Get Grid for ";
    get_grid();
    std::cout << "      Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    // Compute number of boundary face
    unsigned int boundary_face_nb = 0;
    unsigned int boundary_face_nb_0 = 0;
    unsigned int boundary_face_nb_1 = 0;
    unsigned int cell_nb_0 = 0;
    unsigned int cell_nb_1 = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->material_id() == 1)
        cell_nb_1 += 1;
      else
        cell_nb_0 += 1;
    }
    for (const auto &face : triangulation.active_face_iterators())
    {
      if (face->at_boundary())
      {
        boundary_face_nb += 1;

        if (face->boundary_id()==0)
          boundary_face_nb_0 += 1;
        else
          boundary_face_nb_1 += 1;
      }
    }
    std::cout << "      Number of active cells:       " << triangulation.n_active_cells() << " with " << cell_nb_0 << " for 0 and " << cell_nb_1 << " for 1" << std::endl;
    std::cout << "      Number of boundary faces:     " << boundary_face_nb << " with " << boundary_face_nb_0 << " for 0 and " << boundary_face_nb_1 << " for 1" << std::endl;
    setup_system();
    std::cout << "      Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    std::cout << "   End Get Grid with :" << std::endl;
    double f = std::stod(force);
    while (present_time < end_time)
    {
        present_time += present_timestep;
        ++timestep_no;
        if (present_time > end_time)
          {
            present_timestep -= (present_time - end_time);
            present_time = end_time;
            std::cout << present_time << std::endl;
          }
          std::cout << "   Start Assemble System with a pressure = " << std::to_string(f) << " bar" << std::endl;
          assemble_system(f);
          std::cout << "   End Assemble System" << std::endl;

          std::cout << "   Start Solve" << std::endl;
          solve();
          output_results();
          std::cout << "   End Solve" << std::endl;
    }
  }




  /*template <int dim>
  void Apnea_V7<dim>::run(std::string force)
  {
    std::cout << "   Start Get Grid for ";
    get_grid();
    std::cout << "      Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    // Compute number of boundary face
    unsigned int boundary_face_nb = 0;
    unsigned int boundary_face_nb_0 = 0;
    unsigned int boundary_face_nb_1 = 0;
    unsigned int cell_nb_0 = 0;
    unsigned int cell_nb_1 = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->material_id() == 1)
        cell_nb_1 += 1;
      else
        cell_nb_0 += 1;
    }
    for (const auto &face : triangulation.active_face_iterators())
    {
      if (face->at_boundary())
      {
        boundary_face_nb += 1;

        if (face->boundary_id()==0)
          boundary_face_nb_0 += 1;
        else
          boundary_face_nb_1 += 1;
      }
    }
    std::cout << "      Number of active cells:       " << triangulation.n_active_cells() << " with " << cell_nb_0 << " for 0 and " << cell_nb_1 << " for 1" << std::endl;
    std::cout << "      Number of boundary faces:     " << boundary_face_nb << " with " << boundary_face_nb_0 << " for 0 and " << boundary_face_nb_1 << " for 1" << std::endl;
    setup_system();
    std::cout << "      Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    std::cout << "   End Get Grid with :" << std::endl;
    double f = std::stod(force);
    /*std::cout << "   Start Assemble System with a pressure = " << std::to_string(f) << " bar" << std::endl;
    assemble_system(f);
    std::cout << "   End Assemble System" << std::endl;
    std::cout << "   Start Solve" << std::endl;*/
    //solve();
    //while (present_time < end_time)
    //do_timestep(f);
    //std::cout << "   End Solve" << std::endl;
    //std::cout << "   Start Output" << std::endl;
    //output_results(f);
    //std::cout << "   End Output" << std::endl;
//  }

}

// namespace Apnea

int main(int argc, char **argv)
{
  try
    {
      Apnea::Apnea_V7<3> elastic_problem_3d;
      elastic_problem_3d.run(argv[1]);
      //  elastic_problem_3d.run();
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
  return 0;
}
