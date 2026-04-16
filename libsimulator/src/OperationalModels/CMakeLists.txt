# include per-model lists
add_subdirectory(AnticipationVelocityModel)
add_subdirectory(CollisionFreeSpeedModel)
add_subdirectory(CollisionFreeSpeedModelV2)
add_subdirectory(GeneralizedCentrifugalForceModel)
add_subdirectory(SocialForceModel)

## OperationalModels central list
list(APPEND SIMULATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/OperationalDecisionSystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OperationalModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OperationalModelUpdate.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OperationalModelType.hpp
)

set(SIMULATOR_SOURCES ${SIMULATOR_SOURCES} PARENT_SCOPE)