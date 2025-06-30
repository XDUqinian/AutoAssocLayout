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
# add_executable(wave.dac waveEquation.dac_sycl.cpp)
# add_executable(wave.sycl waveEquationStandardSycl.cpp)
add_executable(devicePackingTest devicePackingTest.sycl.cpp)
# add_executable(DataReconstructor DataReconstructor.cpp)
# add_executable(test test.cpp)
add_executable(packingTest DataReconstructorTest.cpp)

# set_target_properties(wave.dac PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
# )
# set_target_properties(wave.sycl PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
# )
set_target_properties(devicePackingTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
)
# set_target_properties(DataReconstructor PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
# )
# set_target_properties(test PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
# )
set_target_properties(packingTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src
)

# 链接库到可执行文件
# target_link_libraries(wave.dac PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)
# target_link_libraries(wave.sycl PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)
target_link_libraries(devicePackingTest PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)
# target_link_libraries(DataReconstructor PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)
# target_link_libraries(test PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)
target_link_libraries(packingTest PRIVATE ${OpenCL_LIBRARY} sycl RewriterLib)