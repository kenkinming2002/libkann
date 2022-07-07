libtensor_sources = [
  'Backend.cpp',
  'Math.cpp',
  'MatrixProduct.cpp',
  'Image.cpp',
  'Broadcast.cpp',
  'Reduce.cpp',
  'Norm.cpp',
  'MaxCoeff.cpp',
  'Utils.cpp',
]

libtensor_lib = library('tensor', libtensor_sources,
  include_directories : ['../include'],
  cpp_pch             : ['pch/pch.hpp'],
  dependencies        : [fmt_dep, spdlog_dep, range_v3_dep, eigen_dep],
  gnu_symbol_visibility : 'hidden'
)

libtensor_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies        : [fmt_dep, spdlog_dep, range_v3_dep, eigen_dep],
  link_with : libtensor_lib,
)

