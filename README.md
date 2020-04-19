# Implementation of Distributed Reinforcement Learning with Tensorflow

## Information

* 20 actors with 1 learner.
* Tensorflow implementation with `distributed tensorflow` of server-client architecture.

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
- [ ] [R2D2: Repeatable and Reliable Detector and Descriptor](https://arxiv.org/abs/1906.06195)

## How to Run

* DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
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

# Reference

1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
2. [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/abs/1803.00933)
3. [R2D2: Repeatable and Reliable Detector and Descriptor](https://arxiv.org/abs/1906.06195)
4. [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
5. [Asynchronous_Advatnage_Actor_Critic](https://github.com/alphastarkor/distributed_tensorflow_a3c)
6. [Relational_Deep_Reinforcement_Learning](https://github.com/RLOpensource/Relational_Deep_Reinforcement_Learning)
