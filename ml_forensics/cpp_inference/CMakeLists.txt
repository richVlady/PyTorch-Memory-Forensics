cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cifar_inference CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure full DWARF debug symbols and disable compiler optimizations
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")

# Locate LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define the executable target
add_executable(cifar_inference main.cpp)

# Link against LibTorch runtime shared libraries
target_link_libraries(cifar_inference "${TORCH_LIBRARIES}")

# Ensure the executable resolves LibTorch shared objects at runtime
set_property(TARGET cifar_inference PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)