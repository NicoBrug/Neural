cmake_minimum_required(VERSION 3.9.0)

project(Neural
    VERSION 1.0
    DESCRIPTION "Multilayer Perceptron & Backpropagation"
    LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 

find_package(PkgConfig REQUIRED)
find_package(sciplot)

pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

include_directories(lib/EigenRand-0.3.3)
link_directories(lib/EigenRand-0.3.3)



add_executable(Neural
    src/network.cpp
    src/fc_layer.cpp
    src/layer.cpp
    src/activation_layer.cpp

    src/main.cpp
    
    header/activation.h
    header/activation_layer.h
    header/fc_layer.h
    header/layer.h
    header/network.h
    )


target_link_libraries(Neural PUBLIC ${JSONCPP_LIBRARIES})
target_link_libraries(Neural PUBLIC sciplot::sciplot)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Neural PUBLIC OpenMP::OpenMP_CXX)
endif()



target_compile_features(Neural PUBLIC cxx_std_20)
target_compile_options(Neural PUBLIC -Wall -Wextra -Wpedantic)



