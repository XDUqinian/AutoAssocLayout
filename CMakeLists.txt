# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(AutoAssocLayout)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dacppLib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rewriter/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpcppLib/include)
include_directories(/usr/include)

set(REWRITER_SOURCES
  rewriter/lib/dacInfo.cpp
)

add_library(RewriterLib ${REWRITER_SOURCES})

add_subdirectory(src)