MUJOCO = ['InvertedPendulum-v5', 'Hopper-v5']
DOUBLE = 'InvertedDoublePendulum-v5'
CONTROL = ['cartpole/swingup', 'walker/walk', 'cheetah/run']
METHODS = ['ppo', 'trpo', 'raw6', 'sn6', 'sac_keep', 'sac_reset']
PANELS = {
    2: ('initial', MUJOCO, ['ppo', 'trpo', 'raw6', 'sn1']),
    3: ('initial', MUJOCO, ['sn1', 'sn3', 'sn6']),
    4: ('initial', MUJOCO + [DOUBLE], ['raw6', 'sn6']),
    5: ('dense', MUJOCO, METHODS[:4]),
    6: ('stress', MUJOCO, METHODS),
    7: ('control', CONTROL, METHODS),
}
SEEDS = [0, 1, 2, 3, 4]
ASSUMPTIONS = dict(gamma=.99, lr=.0003, value_coef=.5, kl_coef=.01,
                   entropy_coef=.01, grad_cap=.5, ppo_epochs=10,
                   ppo_clip=.2, ppo_minibatch=64, trpo_target_kl=.01,
                   sac_lr=.0003, sac_warmup=100, sac_buffer=100000,
                   sac_batch_size=256, sac_gradient_steps=1, sac_train_freq=8)


def schedule(protocol, task):
    if protocol == 'control':
        updates, switch = (30, 10) if task == CONTROL[0] else (25, 8)
        return updates, switch, 512, 3
    updates, switch = (60, 18) if task == DOUBLE else (40, 12)
    return updates, switch, 1024, 3 if protocol == 'stress' else 5


def jobs():
    return list(dict.fromkeys((protocol, task, method, seed)
                for protocol, tasks, methods in PANELS.values()
                for task in tasks for method in methods
                for seed in SEEDS[:5 if protocol == 'stress' else 3]))
