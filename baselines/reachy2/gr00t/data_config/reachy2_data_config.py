# Copyright (c) 2025. SPDX-License-Identifier: Apache-2.0
"""
GR00T data config for the Reachy2 bimanual pick-and-place task.

SCAFFOLD: the key names and index ranges below are a starting point derived
from Reachy2's kinematics (7 DOF per arm + 1 gripper per side) and from the
structure of GR00T N1's `BimanualPandaGripperDataConfig`, which is the closest
stock analogue. They MUST be reconciled against the actual column layout of
whatever demonstration dataset you train on -- if `modality.json` and this file
disagree, training fails on a shape mismatch.

Registration (Isaac-GR00T N1 has no plugin mechanism, so this is a runtime
patch rather than a config entry):

    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from reachy2_data_config import Reachy2DataConfig
    DATA_CONFIG_MAP["reachy2_bimanual"] = Reachy2DataConfig()

then run gr00t_finetune.py with `--data-config reachy2_bimanual
--embodiment-tag new_embodiment`. `new_embodiment` is GR00T's generic
novel-robot tag; no new EmbodimentTag enum member is needed. The tag used at
eval must match the one used at training.
"""

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform

# GR00TTransform pads to max_state_dim=64 / max_action_dim=32, so Reachy2's
# 16-DOF bimanual layout fits with room to spare.
_VIDEO_KEYS = ["video.ego_view"]
_STATE_KEYS = ["state.left_arm", "state.right_arm", "state.left_gripper", "state.right_gripper"]
_ACTION_KEYS = ["action.left_arm", "action.right_arm", "action.left_gripper", "action.right_gripper"]
_LANGUAGE_KEYS = ["annotation.human.action.task_description"]

_OBSERVATION_INDICES = [0]
_ACTION_INDICES = list(range(16))


class Reachy2DataConfig(BaseDataConfig):
    """Joint-space bimanual config: 7 arm joints + 1 gripper per side."""

    video_keys = _VIDEO_KEYS
    state_keys = _STATE_KEYS
    action_keys = _ACTION_KEYS
    language_keys = _LANGUAGE_KEYS
    observation_indices = _OBSERVATION_INDICES
    action_indices = _ACTION_INDICES

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                # video
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                # state: min-max normalized. (The gr1_* configs use a sin/cos
                # encoding instead, but that is specific to GR1's joint
                # conventions -- min_max is the right default for a new robot.)
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                # action
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
                # concat + model transform
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    max_state_dim=64,
                    max_action_dim=32,
                ),
            ]
        )
