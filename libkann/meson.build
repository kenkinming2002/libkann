install_subdir('include/libkann', install_dir : 'include')

libkann_sources = [
  'src/Initialize.cpp',
  'src/SL.cpp',

  # Layers
  'src/Layer.cpp',
  'src/LayerDef.cpp',
  'src/layers/Activation.cpp',
  'src/layers/Convolutional.cpp',
  'src/layers/Dense.cpp',
  'src/layers/Reshape.cpp',
  'src/layers/Sequential.cpp',
  'src/layers/SoftMax.cpp',

  # Datasets
  'src/datasets/MNIST.cpp',

  # Optimizers
  'src/optimizers/SimpleOptimizer.cpp',
  'src/optimizers/AdamOptimizer.cpp',

  # Loss functions
  'src/loss_functions/Lp.cpp',
  'src/loss_functions/CrossEntropy.cpp',

  'src/ProgressBar.cpp',
]

libkann_lib = library('kann', libkann_sources,
  include_directories : ['include'],
  cpp_pch             : ['src/pch/pch.hpp'],
  dependencies        : [libtensor_dep, fmt_dep, range_v3_dep, yaml_cpp_dep],
  gnu_symbol_visibility : 'hidden'
)

libkann_dep = declare_dependency(
  include_directories : ['include'],
  dependencies : [libtensor_dep, fmt_dep, range_v3_dep, yaml_cpp_dep],
  link_with : libkann_lib,
)

subdir('tests')
subdir('examples')

