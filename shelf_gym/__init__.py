from gymnasium.envs.registration import register

register(
    id='ShelfEnv-v0',
    entry_point='shelf_gym.environments.shelf_environment:ShelfEnv',
    kwargs={'render': True}
)

register(
    id='MapCollection-v0',
    entry_point='shelf_gym.scripts.data_generation.map_collection:MapCollection',
    kwargs={'render': True}
)

register(
    id='GraspCollection-v0',
    entry_point='shelf_gym.scripts.data_generation.grasping_collection:GraspingCollection',
    kwargs={'render': True}
)
