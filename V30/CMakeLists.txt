CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

FIND_PACKAGE(deal.II 9.3 REQUIRED
HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Executable
FILE(GLOB executable_files source/*cc)
ADD_EXECUTABLE(apneaV7 ${executable_files})
DEAL_II_SETUP_TARGET(apneaV7)

# Library of the executable
ADD_LIBRARY(apneaV7-lib SHARED $<TARGET_OBJECTS:apneaV7>)
DEAL_II_SETUP_TARGET(apneaV7-lib)

INCLUDE_DIRECTORIES(./include/)
