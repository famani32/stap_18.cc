#include "apnea-v9.h"

int main(int argc, char *argv[]) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    dealii::deallog.depth_console(2);
    apneaV9<3> elastic_problem_3d;
    elastic_problem_3d.run(argv[1]);
  } catch (std::exception &exc) {
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
  } catch (...) {
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
