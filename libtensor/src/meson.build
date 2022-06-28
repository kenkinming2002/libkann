libtensor_sources = [
  'Backend.cpp',
  'Math.cpp',
  'Utils.cpp',
]

libtensor_lib = library('tensor', libtensor_sources,
  include_directories : ['../include'],
  cpp_pch             : ['pch/pch.hpp'],
  dependencies        : [fmt_dep, spdlog_dep, range_v3_dep],
  gnu_symbol_visibility : 'hidden'
)

libtensor_dep = declare_dependency(
  include_directories : ['../include'],
  link_with : libtensor_lib,
)

