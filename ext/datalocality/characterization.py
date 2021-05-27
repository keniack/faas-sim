from typing import Dict

from ext.datalocality import images
from sim.faas import FunctionCharacterization
from sim.oracle.oracle import ResourceOracle, FetOracle


def get_function_characterizations(resource_oracle: ResourceOracle,
                                   fet_oracle: FetOracle) -> Dict[str, FunctionCharacterization]:
    return {
        images.f1_ml_pre_manifest: FunctionCharacterization(
            images.f1_ml_pre_manifest, fet_oracle,
            resource_oracle),
        images.f2_ml_train_manifest: FunctionCharacterization(
            images.f2_ml_train_manifest, fet_oracle,
            resource_oracle),
        images.f3_ml_eval_manifest: FunctionCharacterization(
            images.f3_ml_eval_manifest, fet_oracle,
            resource_oracle),

    }
