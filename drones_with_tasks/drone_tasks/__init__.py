from gymnasium.envs.registration import register

register(
    id = 'Env_task1',
    entry_point = 'drone_tasks.envs:Env_Task1'
)