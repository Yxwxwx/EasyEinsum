cmake_minimum_required(VERSION 3.20)

# 设置项目名称和语言
project(Einsum LANGUAGES CXX)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加额外的编译选项（只在 Release 模式下）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(UNIX)
        set(CMAKE_CXX_FLAGS_RELEASE "$${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")
        #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest/googletest/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest/googlemock/include)

# 添加可执行文件
add_executable(Einsum main.cpp)

find_library(gtest libgtest.a ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest/lib)
find_library(gtest_main libgtest_main.a ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest/lib)
find_library(gmock libgmock.a ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest/lib)
find_library(gmock_main libgmock_main.a ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest/lib)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

#GTEST_BOTH_LIBRARIES表示链接google test的两个库
target_link_libraries( Einsum
        PRIVATE 
        ${gtest}
        ${gtest_main}  # 使用gtest带的main函数,如果检测到外部有main函数，则使用外部main函数,外部main函数要配置gtest初始化。
        ${gmock}
        ${gmock_main} # 使用gmock带的main函数,如果检测到外部有main函数，则使用外部main函数,与gtest_main同时存在则自动配置。
        Threads::Threads
        OpenMP::OpenMP_CXX
        )