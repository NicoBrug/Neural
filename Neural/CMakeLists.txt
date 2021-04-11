cmake_minimum_required(VERSION 3.0.0)

project(P_MultiLayer
    VERSION 1.0
    DESCRIPTION "Multilayer Perceptron & Backpropagation"
    LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})


add_executable(P_MultiLayer
    src/network.cpp
    src/fc_layer.cpp
    src/layer.cpp
    src/activation_layer.cpp
    main.cpp
    
    header/activation_layer.h
    header/fc_layer.h
    header/layer.h
    header/network.h
    )

target_link_libraries(P_MultiLayer ${JSONCPP_LIBRARIES})



target_compile_features(P_MultiLayer PRIVATE cxx_std_20)
target_compile_options(P_MultiLayer PRIVATE -Wall -Wextra -Wpedantic)



