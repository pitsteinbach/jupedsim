## AnticipationVelocityModel sources
list(APPEND SIMULATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/AnticipationVelocityModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnticipationVelocityModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnticipationVelocityModelBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnticipationVelocityModelBuilder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnticipationVelocityModelData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnticipationVelocityModelUpdate.hpp
)

set(SIMULATOR_SOURCES ${SIMULATOR_SOURCES} PARENT_SCOPE)