project('libkann', 'cpp', version : '0.1.0', default_options : ['cpp_std=c++2a'])

add_global_arguments('-fconcepts', language : 'cpp')
add_global_arguments('-Wno-deprecated-enum-enum-conversion', language : 'cpp')

nlohmann_json_proj = subproject('nlohmann-json')
nlohmann_json_dep  = nlohmann_json_proj.get_variable('nlohmann_json_dep')

cereal_dep = declare_dependency(
  include_directories : ['third_party/cereal/include']
)

subdir('src')
subdir('include')
subdir('examples')
subdir('tests')
