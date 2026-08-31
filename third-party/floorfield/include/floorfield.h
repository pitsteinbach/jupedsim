#pragma once
// Stable include for the floorfield Rust library.
//
// The actual opaque Floorfield type, all constructor free functions, and router
// methods are declared in the cxx-generated bridge header.  Add
//   ${CMAKE_SOURCE_DIR}/third-party/floorfield/target/cxxbridge
// to your target's include paths and include this header.
//
// CMake: link against the `floorfield` Corrosion target from third-party/.
#include "rust/cxx.h"
#include "floorfield/src/lib.rs.h"
