from gym.envs.registration import register

register(
    id='boxworld-v0',
    entry_point='gym_boxworld.atari:BoxWorldEnv',
)

register(
    id='BoxWorldNoFrameskip-v3',
    entry_point='gym_boxworld.atari:BoxWorldEnv',
    kwargs={'tcp_tagging': True}, # A frameskip of 1 means we get every frame
    max_episode_steps=10000,
    nondeterministic=False,
)
