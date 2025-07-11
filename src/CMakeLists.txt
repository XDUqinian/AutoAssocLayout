# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(SyclTest)

set(CMAKE_CXX_STANDARD 17)

# 设置使用的编译器为 icpx
set(CMAKE_C_COMPILER "/data/qinian/share/intel/oneapi2025/compiler/2025.0/bin/icpx")
set(CMAKE_CXX_COMPILER "/data/qinian/share/intel/oneapi2025/compiler/2025.0/bin/icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")

# 添加 SYCL 头文件路径
set(SYCL_PATH "/data/qinian/share/intel/oneapi2025/compiler/latest/include")
set(OpenCL_LIBRARY "/data/qinian/share/intel/oneapi2025/compiler/latest/lib/libOpenCL.so")

include_directories(${SYCL_PATH})

# 添加源文件
add_executable(DataRecon DataReconNewTest.sycl.cpp)
add_executable(waveEq waveEquationStandardSycl.cpp)

set_target_properties(DataRecon PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
)
set_target_properties(waveEq PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
)

# 链接库到可执行文件
target_link_libraries(DataRecon PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)
target_link_libraries(waveEq PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)