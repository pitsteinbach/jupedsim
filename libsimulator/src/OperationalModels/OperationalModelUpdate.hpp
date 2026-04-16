// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AnticipationVelocityModel/AnticipationVelocityModelUpdate.hpp"
#include "CollisionFreeSpeedModel/CollisionFreeSpeedModelUpdate.hpp"
#include "CollisionFreeSpeedModelV2/CollisionFreeSpeedModelV2Update.hpp"
#include "GeneralizedCentrifugalForceModel/GeneralizedCentrifugalForceModelUpdate.hpp"
#include "SocialForceModel/SocialForceModelUpdate.hpp"

#include <variant>

using OperationalModelUpdate = std::variant<
    GeneralizedCentrifugalForceModelUpdate,
    CollisionFreeSpeedModelUpdate,
    CollisionFreeSpeedModelV2Update,
    AnticipationVelocityModelUpdate,
    SocialForceModelUpdate>;
