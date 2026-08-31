################################################################################
# Build libsimulator
################################################################################


add_library(simulator STATIC)

target_sources(simulator PRIVATE
    src/AABB.cpp
    src/AABB.hpp
    src/AgentRemovalSystem.hpp
    src/AgentView.hpp
    src/CfgCgal.hpp
    src/CollisionGeometry.cpp
    src/CollisionGeometry.hpp
    src/Ellipse.cpp
    src/Ellipse.hpp
    src/Floorfield.cpp
    src/Floorfield.hpp
    src/GenericAgent.hpp
    src/GeometricFunctions.hpp
    src/GeometryBuilder.cpp
    src/GeometryBuilder.hpp
    src/Graph.hpp
    src/Grid2D.hpp
    src/HashCombine.hpp
    src/IteratorPair.hpp
    src/Journey.cpp
    src/Journey.hpp
    src/LineSegment.cpp
    src/LineSegment.hpp
    src/Logger.cpp
    src/Logger.hpp
    src/Macros.hpp
    src/Mathematics.cpp
    src/Mathematics.hpp
    src/Mesh.cpp
    src/Mesh.hpp
    src/NeighborhoodSearch.hpp
    src/OperationalDecisionSystem.hpp
    src/Point.cpp
    src/Point.hpp
    src/Polygon.cpp
    src/Polygon.hpp
    src/Routing.cpp
    src/Routing.hpp
    src/RoutingEngine.cpp
    src/RoutingEngine.hpp
    src/Simulation.cpp
    src/Simulation.hpp
    src/SimulationClock.cpp
    src/SimulationClock.hpp
    src/SimulationError.hpp
    src/Stage.cpp
    src/Stage.hpp
    src/StageDescription.hpp
    src/StageManager.cpp
    src/StageManager.hpp
    src/StageSystem.cpp
    src/StageSystem.hpp
    src/StrategicalDesicionSystem.hpp
    src/TacticalDecisionSystem.hpp
    src/TemplateHelper.hpp
    src/Timing.cpp
    src/Timing.hpp
    src/Tracing.cpp
    src/Tracing.hpp
    src/UniqueID.hpp
    src/Util.hpp
)

add_subdirectory(src/OperationalModels)
target_link_libraries(simulator PRIVATE project_options)
target_compile_definitions(simulator PUBLIC
    JPSCORE_VERSION="${PROJECT_VERSION}"
)

################################################################################
# Optional parallel STL (C++17 par_unseq):
#   Apple Clang  — -fexperimental-library enables the GCD backend (no TBB)
#   MSVC         — built-in, no extra flags or libraries
#   GCC / Clang  — requires TBB for the parallel backend
################################################################################
option(JUPEDSIM_ENABLE_PARALLEL "Enable C++17 par_unseq for the simulation hot loop" ON)
if(JUPEDSIM_ENABLE_PARALLEL)
    if(APPLE)
        target_compile_options(simulator PRIVATE -fexperimental-library)
        # Propagate to consumers so their link step resolves the libdispatch symbols
        # embedded in libsimulator.a's object files.
        target_link_options(simulator PUBLIC -fexperimental-library)
        target_compile_definitions(simulator PRIVATE JUPEDSIM_PARALLEL_STL)
        message(STATUS "JupedSim: parallel STL enabled (Apple Clang / GCD via -fexperimental-library)")
    elseif(MSVC)
        target_compile_definitions(simulator PRIVATE JUPEDSIM_PARALLEL_STL)
        message(STATUS "JupedSim: parallel STL enabled (MSVC built-in)")
    else()
        find_package(TBB QUIET)
        if(TBB_FOUND)
            target_link_libraries(simulator PRIVATE TBB::tbb)
            target_compile_definitions(simulator PRIVATE JUPEDSIM_PARALLEL_STL)
            message(STATUS "JupedSim: parallel STL enabled (TBB ${TBB_VERSION})")
        else()
            message(STATUS "JupedSim: TBB not found — parallel STL disabled (install libtbb-dev or pass -DJUPEDSIM_ENABLE_PARALLEL=OFF to silence)")
        endif()
    endif()
endif()
target_link_libraries(simulator PUBLIC
    common
    fmt::fmt
    CGAL::CGAL
    build_info
    glm::glm
    perfetto
    floorfield
)
target_link_options(simulator PUBLIC
    $<$<AND:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>,$<BOOL:${BUILD_WITH_SANITIZERS}>>:-fsanitize=address,undefined>
    $<$<AND:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,$<BOOL:${BUILD_WITH_SANITIZERS}>>:-shared-libasan>
    $<$<CXX_COMPILER_ID:GNU>:-Werror=odr>  # error out if LTO finds multiple definitions
)
target_include_directories(simulator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# Compute the Corrosion cargo target directory for the floorfield crate.
# Corrosion uses ${CMAKE_BINARY_DIR}/cargo/${crate_dir_name}_${sha1_first5(manifest_path)} as
# target dir, and cxx places the bridge headers at <target_dir>/<rust_triple>/cxxbridge/.
string(SHA1 _ff_hash "${CMAKE_SOURCE_DIR}/third-party/floorfield/Cargo.toml")
string(SUBSTRING "${_ff_hash}" 0 5 _ff_hash)
set(_ff_cxxbridge "${CMAKE_BINARY_DIR}/cargo/floorfield_${_ff_hash}/${_CORROSION_RUST_CARGO_TARGET}/cxxbridge")

target_include_directories(simulator SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/third-party/floorfield/include
    ${_ff_cxxbridge}
)

################################################################################
# libsimulator unit tests
################################################################################
if (BUILD_TESTS)
    add_executable(libsimulator-tests
        test/TestAABB.cpp
        test/TestAgentView.cpp
        test/TestBasicPrimitiveTests.cpp
        test/TestCollisionGeometry.cpp
        test/TestCustomModel.cpp
        test/TestGenericAgentFormatter.cpp
        test/TestGraph.cpp
        test/TestJourney.cpp
        test/TestLineSegment.cpp
        test/TestMesh.cpp
        test/TestNeighborhoodSearch.cpp
        test/TestPoint.cpp
        test/TestSimulationClock.cpp
        test/TestStage.cpp
        test/TestUniqueID.cpp
    )

    target_link_libraries(libsimulator-tests PRIVATE
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        simulator
    )

    target_link_libraries(libsimulator-tests PRIVATE project_options)

endif()

################################################################################
# libjupedsim micro benchmarks
################################################################################
if (BUILD_BENCHMARKS)
    add_executable(libsimulator-benchmarks
        benchmark/BenchmarkMain.cpp
        benchmark/benchmarkLineSegment.hpp
        benchmark/benchmarkCollisionGeometry.hpp
        benchmark/buildGeometries.hpp
    )

    target_link_libraries(libsimulator-benchmarks PRIVATE
        benchmark::benchmark
        simulator
    )

    target_link_libraries(libsimulator-benchmarks PRIVATE project_options)

endif ()
