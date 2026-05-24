cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)
find_package(TBB REQUIRED)


include_directories(${TBB_INCLUDE_DIRS})
file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s SHARED ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} ${TBB_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_s ${TBB_LIBRARIES})