// SPDX-License-Identifier: LGPL-3.0-or-later
#include "GenericAgent.hpp"
#include "OperationalModels/AnticipationVelocityModel/AnticipationVelocityModel.hpp"
#include "OperationalModels/CollisionFreeSpeedModel/CollisionFreeSpeedModel.hpp"
#include "OperationalModels/CollisionFreeSpeedModelV2/CollisionFreeSpeedModelV2.hpp"
#include "OperationalModels/GeneralizedCentrifugalForceModel/GeneralizedCentrifugalForceModel.hpp"
#include "OperationalModels/SocialForceModel/SocialForceModel.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

static GenericAgent make_agent(GenericAgent::Model model)
{
    return GenericAgent(
        GenericAgent::ID{},
        jps::UniqueID<Journey>::Invalid,
        jps::UniqueID<BaseStage>::Invalid,
        Point{1.0, 2.0},
        std::move(model));
}

TEST(GenericAgentFormatter, FormatsGeneralizedCentrifugalForceModelAgent)
{
    auto agent = make_agent(GeneralizedCentrifugalForceModel::MakeState({}));
    ASSERT_NO_THROW((void) fmt::format("{}", agent));
}

TEST(GenericAgentFormatter, FormatsCollisionFreeSpeedModelAgent)
{
    auto agent = make_agent(CollisionFreeSpeedModel::MakeState({}));
    ASSERT_NO_THROW((void) fmt::format("{}", agent));
}

TEST(GenericAgentFormatter, FormatsCollisionFreeSpeedModelV2Agent)
{
    auto agent = make_agent(CollisionFreeSpeedModelV2::MakeState({}));
    ASSERT_NO_THROW((void) fmt::format("{}", agent));
}

TEST(GenericAgentFormatter, FormatsAnticipationVelocityModelAgent)
{
    auto agent = make_agent(AnticipationVelocityModel::MakeState({}));
    ASSERT_NO_THROW((void) fmt::format("{}", agent));
}

TEST(GenericAgentFormatter, FormatsSocialForceModelAgent)
{
    auto agent = make_agent(SocialForceModel::MakeState({}));
    ASSERT_NO_THROW((void) fmt::format("{}", agent));
}
