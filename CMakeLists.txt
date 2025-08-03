cmake_minimum_required(VERSION 3.12)
project(SLICESTokenizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Print build information
message(STATUS "Building ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# Set up external dependencies directory
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
file(MAKE_DIRECTORY ${EXTERNAL_DIR})

# Download and setup nlohmann/json using FetchContent
include(FetchContent)

message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
	nlohmann_json
	GIT_REPOSITORY https://github.com/nlohmann/json.git
	GIT_TAG v3.12.0
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)

message(STATUS "Fetching fmtlib/fmt...")
FetchContent_Declare(
	fmt
	GIT_REPOSITORY https://github.com/fmtlib/fmt.git
	GIT_TAG 11.2.0
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG v3.0.0
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
	FetchContent_Populate(pybind11)
endif()

add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})

FetchContent_MakeAvailable(pybind11 fmt nlohmann_json)

# Configure nlohmann/json options (don't build tests/examples)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_MultipleHeaders OFF CACHE INTERNAL "")

add_executable(slices_tokenizer_demo main.cpp)

pybind11_add_module(
	slices_tokenizer
	pybind11_bindings.cpp
)

if(MSVC)
	target_compile_options(slices_tokenizer_demo PRIVATE
		/W4 # Enable level 4 warnings
		$<$<CONFIG:Release>:/O2>	 # Optimize for speed in Release
		$<$<CONFIG:Debug>:/Od /Zi> # No optimization + debug info in Debug
	)
else()
	target_compile_options(slices_tokenizer_demo PRIVATE
		-Wall -Wextra -Wpedantic# Enable comprehensive warnings
		$<$<CONFIG:Release>:-Ofast -march=native -DNDEBUG>  # Aggressive optimization in Release
		$<$<CONFIG:Debug>:-g -O0>	# Debug info + no optimization in Debug
	)
	
	# Enable additional optimizations for Release builds
	if(CMAKE_BUILD_TYPE STREQUAL "Release")
		target_compile_options(slices_tokenizer_demo PRIVATE
			-flto # Link-time optimization
			-ffast-math	# Fast math operations
		)
		target_link_options(slices_tokenizer_demo PRIVATE -flto)
	endif()
endif()

add_library(slices_tokenizer_static INTERFACE)
target_include_directories(slices_tokenizer_static INTERFACE 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include>
)
target_link_libraries(slices_tokenizer_static INTERFACE 
	nlohmann_json::nlohmann_json
	fmt::fmt
	pybind11::headers
)

add_library(SLICESTokenizer::static ALIAS slices_tokenizer_static)

target_link_libraries(slices_tokenizer_demo PRIVATE SLICESTokenizer::static pybind11::embed)
target_link_libraries(slices_tokenizer PRIVATE SLICESTokenizer::static)
target_compile_definitions(slices_tokenizer PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Create a custom target for running the demo
add_custom_target(run_demo
	COMMAND slices_tokenizer_demo
	DEPENDS slices_tokenizer_demo
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Running SLICES tokenizer demo"
)

enable_testing()
add_test(
	NAME basic_tokenizer_test 
	COMMAND slices_tokenizer_demo
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(
	TARGETS slices_tokenizer_demo
	RUNTIME DESTINATION bin
)

install(
	FILES slices_tokenizer.hpp
	DESTINATION include
)

message(STATUS "Configuration complete!")
message(STATUS "  Executable: slices_tokenizer_demo")
message(STATUS "  Static library: slices_tokenizer_static") 
if(pybind11_FOUND)
	message(STATUS "  Python module: slices_tokenizer")
endif()
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "  ./slices_tokenizer_demo")
