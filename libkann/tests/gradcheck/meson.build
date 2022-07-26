gradcheck_sources = [ 'check.cpp', ]
gradcheck = executable('gradcheck', gradcheck_sources,
  dependencies : [libkann_dep, catch2_dep]
)
test('gradcheck', gradcheck, workdir : meson.current_source_dir())
