cmake_minimum_required(VERSION 3.10 FATAL_ERROR)



message(STATUS "OpenACC C version " ${OpenACC_C_VERSION})
message(STATUS "OpenACC Cxx version " ${OpenACC_CXX_VERSION})
message(STATUS "OpenACC C flags " ${OpenACC_CXX_FLAGS})
message(STATUS "OpenACC Cxx flags " ${OpenACC_CXX_FLAGS})

#add_executable(ex7 example7.c)
add_executable(ex5 example5.c)
add_definitions(-DROWS=3000 -DCOLS=3000)
find_package(OpenACC)
#set(OpenACC_ACCEL_TARGET ex7)
#target_compile_options(ex7 PRIVATE ${OpenACC_C_FLAGS})
target_compile_options(ex5 PRIVATE "${OpenACC_C_FLAGS}")
target_link_libraries(ex5 PRIVATE ${OpenACC_CXX_FLAGS})

# run as ACC_DEVICE_TYPE=host ./ex5
# https://gcc.gnu.org/wiki/OpenACC#ACC_DEVICE_TYPE means it will run single threaded
