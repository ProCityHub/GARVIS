cmake_minimum_required(VERSION 3.16)
project(garvis_opencl C)

set(CMAKE_C_STANDARD 11)

# ── Find OpenCL ────────────────────────────────────────────────────────────
find_package(OpenCL REQUIRED)

# OpenCL 1.2+ API — suppress deprecation warnings on macOS (CL_TARGET_OPENCL_VERSION)
add_compile_definitions(CL_TARGET_OPENCL_VERSION=120)

# ── Executable ────────────────────────────────────────────────────────────
add_executable(garvis_opencl
    agent_loop.c
)

target_include_directories(garvis_opencl PRIVATE
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(garvis_opencl PRIVATE
    ${OpenCL_LIBRARIES}
)

# ── Copy kernel files next to the binary at build time ────────────────────
add_custom_command(TARGET garvis_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/kernels
        $<TARGET_FILE_DIR:garvis_opencl>/kernels
    COMMENT "Copying OpenCL kernels to build directory"
)

# ── Optional: warn if OpenCL version is below 1.2 ─────────────────────────
if(OpenCL_VERSION_STRING AND OpenCL_VERSION_STRING VERSION_LESS "1.2")
    message(WARNING "OpenCL < 1.2 detected — some features may not work.")
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS garvis_opencl DESTINATION bin)
install(DIRECTORY kernels/ DESTINATION share/garvis_opencl/kernels)
