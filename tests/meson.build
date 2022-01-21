test_all_sources = [
  'main.cpp',
  'Executor.cpp'
]

catch2_dep = dependency('catch2')

test_all = executable('test_all', test_all_sources,
  dependencies : [libkann_dep, catch2_dep]
)

test('test_all', test_all)
