libtensor_sources = [
  'Math.cpp',
  'Utils.cpp',
]

libtensor_lib = library('tensor', libtensor_sources,
  include_directories : ['../include'],
  dependencies        : [fmt_dep, range_v3_dep, eigen_dep],
  gnu_symbol_visibility : 'hidden'
)

libtensor_dep = declare_dependency(
  include_directories : ['../include'],
  link_with : libtensor_lib,
)

