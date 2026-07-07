from jupedsim.models.custom_model import (
    CustomModelAgentUpdate,
    CustomOperationalModel,
)
from jupedsim_examples.models.pysocial_force import PythonSocialForceModel
from jupedsim_examples.models.pycollision_free_speed import (
    PythonColisionFreeSpeedModel,
)

from jupedsim.models.collision_free_speed import CollisionFreeSpeedModel


class MultiModel(CustomOperationalModel):
    def __init__(self, cfsm: CollisionFreeSpeedModel):
        CustomOperationalModel.__init__(self)
        self._model1 = PythonColisionFreeSpeedModel(cfsm)
        self._model2 = PythonSocialForceModel()

    def compute_new_position(
        self, dt: float, ped, geometry, neighborhood_search
    ):
        if ped.model.model_id == 1:
            return self._model1.compute_new_position(
                dt, ped, geometry, neighborhood_search
            )
        elif ped.model.model_id == 2:
            return self._model2.compute_new_position(
                dt, ped, geometry, neighborhood_search
            )

    def check_model_constraint(self, ped, neighborhoodsearch, geometry):
        model_id = ped.model.model_id
        if model_id == 1:
            return self._model1.check_model_constraint(
                ped, neighborhoodsearch, geometry
            )
        elif model_id == 2:
            return self._model2.check_model_constraint(
                ped, neighborhoodsearch, geometry
            )
