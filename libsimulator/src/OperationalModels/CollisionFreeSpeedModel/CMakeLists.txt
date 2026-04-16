## CollisionFreeSpeedModel sources
list(APPEND SIMULATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelBuilder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CollisionFreeSpeedModelUpdate.hpp
)

set(SIMULATOR_SOURCES ${SIMULATOR_SOURCES} PARENT_SCOPE)