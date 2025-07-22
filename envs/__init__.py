#envs/_init_.py


#xzy position only
from envs.ik.reach_cube_position_IK import ReachCubePositionCurrEnv
from envs.ik.reach_cube_position_IKsimple import ReachCubePositionEnv

#third person vision only
from envs.torque.reach_cube_vision_torque import ReachCubeVisionTorqueEnv
from envs.ik.reach_cube_vision_IK import ReachCubeVisionEnv
from envs.ik.reach_cube_vision_stacked_IKsimple import ReachCubeVisionStackedEnv

#end effector ego perspective vision only
from envs.ik.reach_cube_ego_vision_IKsimple import ReachCubeEgoVisionEnv
from envs.ik.reach_cube_ego_vision_stacked_IKsimple import ReachCubeEgoVisionStackedEnv

#end effector microphone only
from envs.ik.reach_cube_ego_audio_IKsimple import ReachCubeEgoAudioEnv
from envs.ik.reach_cube_ego_audio_stacked_IKsimple import ReachCubeEgoAudioStackedEnv

#both end effector vision and microphone
from envs.ik.reach_cube_ego_multimodal_stacked_IKsimple import ReachCubeEgoMultimodalStackedEnv


#directJointcontrol
from envs.torque.reach_cube_position_torque import ReachCubeTorqueEnv