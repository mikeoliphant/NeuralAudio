function(create_util util_name)
    message(STATUS "Configuring util: ${util_name}")
    add_executable(${util_name} ${util_name}.cpp)
    target_include_directories(${util_name} PUBLIC ${CMAKE_SOURCE_DIR})
    )
    target_link_libraries(${util_name} PUBLIC NeuralAudio)
    target_compile_features(${util_name} PRIVATE cxx_std_17)
endfunction()

add_subdirectory(ModelTest)

file(COPY Models DESTINATION ./)
