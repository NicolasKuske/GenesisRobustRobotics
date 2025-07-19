#env/_init_.py


#xzy position only
from .reach_cube_position import ReachCubePositionEnv
from .reach_cube_position_stacked import ReachCubePositionStackedEnv

#third person vision only
from .reach_cube_vision import ReachCubeVisionEnv
from .reach_cube_vision_stacked import ReachCubeVisionStackedEnv

#end effector ego perspective vision only
from .reach_cube_ego_vision import ReachCubeEgoVisionEnv
from .reach_cube_ego_vision_stacked import ReachCubeEgoVisionStackedEnv

#end effector microphone only
from .reach_cube_ego_audio import ReachCubeEgoAudioEnv
from .reach_cube_ego_audio_stacked import ReachCubeEgoAudioStackedEnv

#both end effector vision and microphone
from .reach_cube_ego_multimodal_stacked import ReachCubeEgoMultimodalStackedEnv


#directJointcontrol
from .reach_fixed_cube_directJointcontrol import ReachFixedCubeDirectJointControlEnv
from .reach_cube_torque_control import ReachCubeTorqueEnv