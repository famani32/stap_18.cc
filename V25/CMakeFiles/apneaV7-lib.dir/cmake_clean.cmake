file(REMOVE_RECURSE
  "libapneaV7-lib.pdb"
  "libapneaV7-lib.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/apneaV7-lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
