
libkann_sources = [
  'layers/Layer.cpp',
  'layers/IdentityLayer.cpp',
  'layers/WeightLayer.cpp',
  'layers/ActivationLayer.cpp',
  'layers/ConvolutionalLayer.cpp',
  'layers/DeconvolutionalLayer.cpp',
  'datasets/IDXFile.cpp',
  'datasets/MNISTDataSet.cpp',
  'datasets/RandomDataSet.cpp',
  'datasets/write.cpp',
  'Model.cpp',
  'Build.cpp',
  'FunctionalModel.cpp',
  'FunctionalVariable.cpp',
  'Algorithm.cpp',
  'ActivationFunction.cpp',
  'DefaultExecutor.cpp',
  'Predictor.cpp',
  'Optimizer.cpp',
  'Differentiate.cpp',
  'operations/IdentityOperation.cpp',
  'operations/ReduceOperation.cpp',
  'operations/MatrixMultiplyOperation.cpp',
  'operations/CWiseProductOperation.cpp',
  'operations/MultiplyOperation.cpp',
  'operations/SubtractOperation.cpp',
]

eigen_dep = dependency('eigen3')
boost_dep = dependency('boost', modules : ['graph'])

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

libkann_lib = static_library('libkann', libkann_sources,
  include_directories : ['../include'],
  dependencies        : [cereal_dep, eigen_dep, boost_dep, sfml_dep],
  cpp_args            : ['-fvisibility=hidden']
)

libkann_dep = declare_dependency(
  include_directories : ['../include'],
  dependencies : [cereal_dep, eigen_dep, sfml_dep],
  link_with : libkann_lib,
)
