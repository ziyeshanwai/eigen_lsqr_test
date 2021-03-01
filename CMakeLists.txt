cmake_minimum_required(VERSION 3.17)
project(eigen_lsqr_test)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories("E:\\mylib\\eigen-3.3.9")  # 注意使用\\
add_executable(eigen_lsqr_test
        LSQR.cpp example.cpp)
