set(CMAKE_CXX_STANDARD 20)

# add_executable(main main.cpp)
# target_link_libraries(main ${PROJECT_NAME})
add_executable(update update.cpp)
target_link_libraries(update ${PROJECT_NAME} -ltbb)

add_executable(build_index build_index.cpp)
target_link_libraries(build_index ${PROJECT_NAME} -ltbb)

add_executable(insert insert.cpp)
target_link_libraries(insert ${PROJECT_NAME} -ltbb)
add_executable(search search.cpp)
target_link_libraries(search ${PROJECT_NAME} -ltbb)
# add_executable(bsl blcok_skiplist.cpp)
# target_link_libraries(bsl ${PROJECT_NAME})
