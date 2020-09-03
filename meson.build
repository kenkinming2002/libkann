project('libkann', 'cpp', version : '0.1.0', default_options : ['cpp_std=c++2a'])

nlohmann_json_proj = subproject('nlohmann-json')
nlohmann_json_dep  = nlohmann_json_proj.get_variable('nlohmann_json_dep')

subdir('src')
subdir('include')
subdir('examples')

