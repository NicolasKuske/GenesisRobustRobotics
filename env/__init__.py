#env/_init_.py


#xzy position only
from .reach_cube_position_IK import ReachCubePositionCurrEnv
from .reach_cube_position_IKsimple import ReachCubePositionEnv

#third person vision only
from .reach_cube_vision import ReachCubeVisionEnv
from .reach_cube_vision_stacked_IKsimple import ReachCubeVisionStackedEnv

#end effector ego perspective vision only
from .reach_cube_ego_vision import ReachCubeEgoVisionEnv
from .reach_cube_ego_vision_stacked import ReachCubeEgoVisionStackedEnv

#end effector microphone only
from .reach_cube_ego_audio import ReachCubeEgoAudioEnv
from .reach_cube_ego_audio_stacked import ReachCubeEgoAudioStackedEnv

#both end effector vision and microphone
from .reach_cube_ego_multimodal_stacked import ReachCubeEgoMultimodalStackedEnv


#directJointcontrol
from .reach_cube_torque_control import ReachCubeTorqueEnv