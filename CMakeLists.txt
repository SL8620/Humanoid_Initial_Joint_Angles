cmake_minimum_required(VERSION 3.10)
project(mj_egl_sdl CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

set(MUJOCO_DIR "" CACHE PATH "MuJoCo dir with include/ and lib/")
include_directories(mujoco-3.4.0/include)
link_directories(mujoco-3.4.0/lib)

add_executable(mj_egl_sdl src/main.cpp)
target_include_directories(mj_egl_sdl PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(mj_egl_sdl
  mujoco
  yaml-cpp
  ${SDL2_LIBRARIES}
  EGL GL
)