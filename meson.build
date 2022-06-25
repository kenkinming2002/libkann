project('libkann', 'cpp', version : '0.1.0', default_options : ['cpp_std=c++2a'])

fmt_dep = dependency('fmt')
range_v3_dep = dependency('range-v3')
eigen_dep = dependency('eigen3')
yaml_cpp_dep = dependency('yaml-cpp')

# TODO: Make this optional
sfml_dep = dependency('sfml-graphics')

catch2_dep = dependency('catch2')

subdir('src')
subdir('include')
subdir('tests')
subdir('examples')
