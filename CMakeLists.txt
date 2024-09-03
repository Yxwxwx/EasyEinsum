cmake_minimum_required(VERSION 3.28)

# 设置项目名称和语言
project(Einsum LANGUAGES CXX)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2 # 你可以选择 fmt 的具体版本
)

FetchContent_MakeAvailable(fmt)

# 添加额外的编译选项（只在 Release 模式下）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(UNIX)
    set(CMAKE_CXX_FLAGS_RELEASE "$${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)

enable_testing()
# 添加可执行文件
add_executable(Einsum main.cpp)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

# GTEST_BOTH_LIBRARIES表示链接google test的两个库
target_link_libraries(Einsum GTest::gtest_main Threads::Threads
                      OpenMP::OpenMP_CXX fmt::fmt)

include(GoogleTest)
gtest_discover_tests(Einsum)
