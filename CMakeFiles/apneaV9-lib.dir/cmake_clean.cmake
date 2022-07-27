file(REMOVE_RECURSE
  "libapneaV9-lib.pdb"
  "libapneaV9-lib.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/apneaV9-lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
