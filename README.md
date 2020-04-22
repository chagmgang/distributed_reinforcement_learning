# Implementation of Distributed Reinforcement Learning with Tensorflow

## Information

* 20 actors with 1 learner.
* Tensorflow implementation with `distributed tensorflow` of server-client architecture.
* `Recurrent Experience Replay in Distributed Reinforcement Learning` is implemented in CartPole-v0 environment with POMDP(only position state)

## Dependency
```
opencv-python
gym[atari]
tensorboardX
tensorflow==1.14.0
```


## Implementation

- [x] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- [x] [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/abs/1803.00933)
- [x] [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)

## How to Run

* Ape-x: DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
```
python train_apex.py --job_name learner --task 0

CUDA_VISIBLE_DEVICES=-1 python train_apex.py --job_name actor --task 0
CUDA_VISIBLE_DEVICES=-1 python train_apex.py --job_name actor --task 1
CUDA_VISIBLE_DEVICES=-1 python train_apex.py --job_name actor --task 2
...
CUDA_VISIBLE_DEVICES=-1 python train_apex.py --job_name actor --task 19
```

* IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
```
python train_impala.py --job_name learner --task 0

CUDA_VISIBLE_DEVICES=-1 python train_impala.py --job_name actor --task 0
CUDA_VISIBLE_DEVICES=-1 python train_impala.py --job_name actor --task 1
CUDA_VISIBLE_DEVICES=-1 python train_impala.py --job_name actor --task 2
...
CUDA_VISIBLE_DEVICES=-1 python train_impala.py --job_name actor --task 19
```

* R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning
```
python train_r2d2.py --job_name learner --task 0

CUDA_VISIBLE_DEVICES=-1 python train_r2d2.py --job_name actor --task 0
CUDA_VISIBLE_DEVICES=-1 python train_r2d2.py --job_name actor --task 1
CUDA_VISIBLE_DEVICES=-1 python train_r2d2.py --job_name actor --task 2
```

# Reference

1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
2. [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/abs/1803.00933)
3. [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)
4. [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
5. [google-research/seed-rl](https://github.com/google-research/seed_rl)
6. [Asynchronous_Advatnage_Actor_Critic](https://github.com/alphastarkor/distributed_tensorflow_a3c)
7. [Relational_Deep_Reinforcement_Learning](https://github.com/RLOpensource/Relational_Deep_Reinforcement_Learning)
8. [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)
