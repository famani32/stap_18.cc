#include "apneaV7.h"
using namespace dealii;

  template <int dim>
  Apnea_V8<dim>::Apnea_V8()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    ,// fe(FE_Q<dim>(1), dim)
      fe(1)
    , quadrature_formula(fe.degree + 1)
    , present_time(0.0)
    , present_timestep(0.25)
    , end_time(4.5)
    , timestep_no(0)

  {}


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


  // Get the mesh for computation
  template <int dim>
  void Apnea_V8<dim>:: get_grid()
  {
    // Initiate the grid and attach it to the triangulation then load the mesh in the grid
    // std::string filename = "mesh/Right_Lung.msh";
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
  void Apnea_V8<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);


    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
          dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
          locally_relevant_dofs);

      system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                           mpi_communicator);
      solution.reinit(locally_owned_dofs, mpi_communicator);
      locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                       mpi_communicator);
      // Reinit the system rhs with mpi_communicator
      system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    /*constraints.condense(dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());*/
  }


  // Function to assemble the vector and the matrix of the system
  template <int dim>
  void Apnea_V8<dim>::assemble_system(double force)
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

  //  progresscpp::ProgressBar progressBar(triangulation.n_active_cells(), 70);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
      //  ++progressBar;
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
                    if (cell->material_id() == 1)
                    {
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
                    else
                    {
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
            if ((face->at_boundary()) && (face->boundary_id() == 0))
            // if (face->at_boundary())
              {
                fe_face_values.reinit(cell, face);
                diaphragm_pressure(fe_face_values.get_quadrature_points(), pressure_values, force,present_time);
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
        //progressBar.display();
      };
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler,
                                              1,
                                              Functions::ZeroFunction<dim>(dim),
                                              boundary_values);
      MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);
      //progressBar.done();
  }

  template <int dim>
  void Apnea_V8<dim>::write_info_solution (const unsigned int    iteration,
							   const double          , //check_value
							   const Vector<double> &current_iterate) const
  {
    std::cout << iteration << std::endl;
    //return SolverControl::success;
  }


  template <int dim>
  void Apnea_V8<dim>::solve()
  {
    std::cout << "      Norm of rhs: " << system_rhs.l2_norm() << std::endl;
    std::cout << "      Nb step max = " << solution.size() << std::endl;

    SolverControl            solver_control(solution.size(), 1e-8 * system_rhs.l2_norm());

  //  SolverCG<Vector<double>> cg(solver_control);
    SolverCG<LA::MPI::Vector> cg(solver_control);
    LA::MPI::PreconditionSSOR preconditioner;
    preconditioner.initialize(system_matrix);

  //  PreconditionSSOR<SparseMatrix<double>> preconditioner;

  //  preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
    // std::cout << solution[0] << std::endl;
    std::cout << "      Solver converged: " << solver_control.last_step() << " iterations." << std::endl;
  }

  template <int dim>
  void Apnea_V8<dim>::output_results(double force) const
  {
    std::string filename;
    if (force < 0)
    {
      filename = "solution/solution_right_lung_bar_" + std::to_string(abs(force));
    }
    else
    {
      filename = "solution/solution_right_lung_bar_" + std::to_string(force);
    }

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

      const std::string fname =
        "solution-" + Utilities::int_to_string(timestep_no, 3) + ".vtk";
        std::ofstream output(fname);
        data_out.write_vtk(output);

      //Enregistrer dans le fichier pvd
        std::vector<std::string> filenames;
        filenames.push_back("solution-" +
          Utilities::int_to_string(timestep_no, 4) + "." +Utilities::int_to_string(0, 3) + ".vtu");
        const std::string visit_master_filename =("solution-" + Utilities::int_to_string(timestep_no, 4) + ".visit");
        std::ofstream visit_master(visit_master_filename);
        DataOutBase::write_visit_record(visit_master, filenames);
        const std::string pvtu_master_filename = ("solution-" + Utilities::int_to_string(timestep_no, 4) + ".pvtu");
        std::ofstream pvtu_master(pvtu_master_filename);
        data_out.write_pvtu_record(pvtu_master, filenames);
        static std::vector<std::pair<double, std::string>> times_and_names;
        times_and_names.push_back(std::pair<double, std::string>(present_time, pvtu_master_filename));


  }

  template <int dim>
  void Apnea_V8<dim>::run(std::string force)
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
        std::cout << "   End Solve" << std::endl;

          std::cout << "   Start Output" << std::endl;
        output_results(f);
        std::cout << "   End Output" << std::endl;

  }
}

template class Apnea_V8<3>;
