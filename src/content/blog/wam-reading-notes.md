---
title: 'WAM 读书笔记'
description: '整理我对 WAM 相关论文的理解，包括 DreamZero、Survey、FastWAM 和后续工作。'
pubDate: 'Jul 20 2026'
heroImage: '../../assets/blog/wam/dreamzero/equation.png'
category: 'note'
mood: 'reading'
tags: ['WAM', 'DreamZero', 'FastWAM', 'paper-reading']
featured: false
---

## World Action Models are Zero-shot Policies

问题在于

- 怎么对齐输出的视频和动作呢？
  - 去噪的时候同时得到 video和action，
  - $[x_{action},x_{video}]$ concat 在一起denoise出来
- 关于模型架构设计，是 bidirectional 好还是autoregressive好？
  - 使用了自回归的架构，然后一个action chunk执行完毕之后，kv cache中，使用真实的观察替换predict的kv
- diffusion有去噪的过程，如何做到实时性

实际上等价于一个视频生成模型 + 一个逆向动力学模型

![image-20260720003445793](../../assets/blog/wam/dreamzero/equation.png)

img同时作为了条件和监督的目标

视频和动作共享一个 t

就模型的架构来说感觉不复杂。实际上是denoise的时候把视频和动作concat在一起了，两者使用相同的timestep；

机器人的状态 + 文本条件作为条件；其中state是拼接到 `x` 上的，但是不 attend 任何key和value，也不加噪，不解码，只是拼接在token序列中作为一个 **只读的条件**

论文后面相当大的一部分在介绍怎么提升速度，如何异步地同时进行 推理 + 动作执行

一个有趣的加速方法是，观察到时间步一样的话，模型在video模糊的时候生成的动作也不可靠

但是少步生成的时候，我们需要模型对于不清晰的视频，也能预测清晰的动作

训练模型对video和action加上不同程度的噪声，构造<视频模糊，动作清晰>的训练对；让模型学会——视频模糊的时候也能生成正确的动作

代码：https://github.com/dreamzero0/dreamzero

## World Action Models: A Survey

## FastWAM

现有的范式：

- 联合生成 未来预测+动作序列，例如上面的 DreamZero
  - 这么做的优点是视频和动作对齐的好
  - 缺点在于比较慢，视频的维度比动作要高，其噪声可能会污染动作
- 先生成对未来的观测，然后使用IDM，基于未来观测解码出未来的动作
  - 串行执行，延迟高
  - 未来预测 visible

在 train 的时候，把 world modeling 作为一种 协同训练的信号；在inference的时候，只生成动作
$$
p_{\theta}(a_{1:H}|o,l)
$$
使用 `Wan 2.2 5B`  action expert 参数量为 `1B` 

## 代码中的实现

`video` 和 `action` 作为两个分支，分别先过 `pre_dit` 然后过一个 `mot` ; 在 `mot` 中 `video` 和 `action`  会做联合的 `attention` 

具体的mask 是这样的

![image-20260720211506284](../../assets/blog/wam/fastwam/mask.png)

左边是 `query` 上面的是 `key` 

生成的 `video` 只能对自己做 `bidirectional` 的 attention；

`action` 可以 `attend` 到 首帧 $f_0$ 和 对 action 序列做 `bidirectional` 的attention

过完 `mot` 之后 `action` 和 `video` 分开， 再来一个 `post-dit` 

**inference**

Video 那一支 `dit` 就只走一个pass，然后存kv cache；action dit只依赖首帧；因为在 `mot` 中，action的query只会attend**首帧**和**action序列** 所以对action的生成没有影响

## GigaWorld-Policy
