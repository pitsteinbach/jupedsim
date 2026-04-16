## Generalized Centrifugal Force Model sources
list(APPEND SIMULATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneralizedCentrifugalForceModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneralizedCentrifugalForceModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneralizedCentrifugalForceModelBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneralizedCentrifugalForceModelBuilder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneralizedCentrifugalForceModelData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneralizedCentrifugalForceModelUpdate.hpp
)

set(SIMULATOR_SOURCES ${SIMULATOR_SOURCES} PARENT_SCOPE)