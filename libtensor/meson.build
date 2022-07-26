install_subdir('include/libtensor', install_dir : 'include')

libtensor_sources = [
  'src/Initializer.cpp',
  'src/SL.cpp',
  'src/MatrixProduct.cpp',
  'src/Image.cpp',
  'src/Broadcast.cpp',
  'src/Reduce.cpp',
  'src/Norm.cpp',
  'src/MaxCoeff.cpp',
  'src/memops/Stack.cpp',
  'src/memops/Split.cpp',
  'src/memops/Index.cpp',
]

libtensor_lib = library('tensor', libtensor_sources,
  include_directories : ['include'],
  cpp_pch             : ['src/pch/pch.hpp'],
  dependencies        : [fmt_dep, range_v3_dep, eigen_dep],
  gnu_symbol_visibility : 'hidden'
)

libtensor_dep = declare_dependency(
  include_directories : ['include'],
  dependencies        : [fmt_dep, range_v3_dep],
  link_with : libtensor_lib,
)

pkg = import('pkgconfig')
pkg.generate(libtensor_lib,
  name : 'libtensor'
)

