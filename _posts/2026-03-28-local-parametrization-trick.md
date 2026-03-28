---
title: "[DeepLearning] Local Reparametrization Trick"
date: 2026-03-28T00:00:00+0900
author: 이정목
layout: post
permalink: /local-reparam-trick/
categories: AI
tags: [CS, Deep Learning, reparametrization trick, Bayesian]
---

대학교 때 들었던 reparametrization trick을 local reparametrization trick을 보면서 다시 보게 되었다.
expectation을 계산하기 위해서 항이 같아야 하는데 그렇지 않아서 같게 만들어 주는 것이 reparametrization trick이다. (아래에 자세히 설명하겠다.)
우리는 항상 사실 expectation을 계산하기 위해서 실제 적분 대신 Monte Carlo Estimation을 쓰고 있었던 것이다! 그런데 이 MC estimation을 가능하게 하려면 같은 텀이 있어야 하는데 그렇지 않은 것이다.

## Reparametrization trick

우리가 어떤 분포 $f_\theta(x)$ 에 대해서 expectation을 구하고 싶은 상황을 생각해 보자.
이때, 해당 expectation을 적분 형태로 나타내면 다음과 같다.

$$
L(\theta) = \mathbb{E}_{p(x)} [f_\theta(x)] = \int f_\theta(x) p(x)dx
$$

우리는 보통 expectation을 maximize하는 것이 목적이기 때문에, 해당 expectation을 $\theta$ 에 대해 미분해야 한다. 당연히 $p(x)$ 에는 $\theta$ 텀이 없기때문에, 우리는 미분의 텀을 $f_\theta(x)$ 에만 적용하면 되고, 이건 $L(\theta)$ 를 미분한 텀도 다시 expectation 형태로 만들 수 있음을 뜻한다.

$$
\triangledown L(\theta) = \nabla_\theta \int f_\theta(x) p(x)dx = \mathbb{E}_{p(x)}[\nabla_\theta f_\theta(x)]
$$

이렇게 적분하는 p(x) 분포가 $\theta$ 에 independent하면, 우리는 몬테카를로 샘플링을 통해서 근사치를 구할 수 있다. 하지만 만약 p(x) 분포가 $\theta$ 에 dependent하면, 우리는 몬테카를로 샘플링을 통해서 근사치를 구할 수 없다. 이 때, 우리는 reparametrization trick을 사용한다. 해당 경우는

$$
L(\theta) = \mathbb{E}_{p_\theta(x)} [f_\theta(x)] = \int f_\theta(x) p_\theta(x)dx
$$

이 경우, 미분하게 되면 다음처럼 expectation 항 하나와 정리할 수 없는 항 하나로 나뉘게 된다.

$$
\nabla_\theta L(\theta) = \nabla_\theta \int f_\theta(x) p_\theta(x)dx = \int \nabla_\theta f_\theta(x) p_\theta(x)dx + \int f_\theta(x) \nabla_\theta p_\theta(x)dx
$$

즉,

$$
\nabla_\theta L(\theta) = \mathbb{E}_{p(x)}[\nabla_\theta f_\theta(x)] + \int f_\theta(x) \nabla_\theta p_\theta(x)dx
$$

두 번째 항은 analytic 하지 않아서 구할 수 없기 때문에, 우리는 $p_\theta(x)$ 를 다시 reparameterize 하여 이 문제를 해결할 수 있다.

우리가 $x=g(\epsilon ; \theta)$ 으로 정의하고, $\epsilon \sim p(\epsilon)$ 즉 가우시안 분포를 따른다고 정의하면, $p(x)$ 는 다음과 같이 표현할 수 있다.

$$
\mathbb{E}_{p_\theta(x)} [f_\theta (x)] = \mathbb{E}_{p(\epsilon)} [f_\theta (g(\epsilon ; \theta))]
$$

해당 항은 미분도 expectation으로 정리 가능하기 때문에, 몬테카를로 샘플링을 통해서 근사치를 구할 수 있다.

$$
\mathbb{E}_{p(\epsilon)} [f_\theta (g(\epsilon ; \theta))] = \frac{1}{N} \sum_{i=1}^N f_\theta (g(\epsilon_i ; \theta))
$$

## Local reparametrization trick

Local reparametrization trick은 논문 [Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557)에서 제안된 방법인데, weight $W$를 reparameterize 했을 때 너무 많은 random sample을 진행해야 해서, weight $W$ 대신 output인 $y$를 바로 reparametrize 하는 방식이다.

우리가 어떤 linear layer의 $W$를 reparametrize 하는 상황은 다음과 같이 표현이 가능하다.

$$
y_i^T = x_i^T W, W\in \mathrm{R}^{1000}\times {1000}, x \in \mathrm{R}^{1000}
$$

이때 $q(W) = \mathcal{N}(\mu, \diag(\sigma))$ 라고 가정하자.

이렇게 계산하면, 우리는 총 $\text{batch_size}\times 1000\times 1000$ 개의 random sample을 진행해야 한다. 하지만 만약 우리가 $y$를 reparametrize 한다면, 우리는 $\text{batch_size}\times 1000$ 개의 random sample만 진행하면 된다. 그리고 가우시안의 합은 다시 가우시안이기 때문에, y는 다음과 같이 나타낼 수 있다.

$$
y_{i,j} = \mathcal{N} (\sum_{k=1}^{1000} x_{i,k} \mu_{k,j}, \sum_{k=1}^{1000} x_{i,k}^2 \sigma_{k,j}^2)
$$

따라서 reparametrize 된 y는 다음과 같이 표현할 수 있게 된다.

$$
y_{i,j} = \sum_{k=1}^{1000} x_{i,k} \mu_{k,j} + \epsilon_{i,j} \cdot\sqrt{\sum_{k=1}^{1000} x_{i,k}^2 \sigma_{k,j}^2}
$$

이 경우, 우리는 노이즈를 큰 매트릭스인 $W$가 아닌 $y$에 대해서 샘플링하기 때문에, 훨씬 효율적으로 학습이 가능하다.

결국 reparametrize를 큰 matrix에 대해서 하면 계산이 너무 비효율적이니, 조금 더 작은 matrix에 대해서 reparametrize 하자는 게 local reparametrization trick 이다. 가우시안의 linearity 성질 (더해도 곱해도 가우시안)을 이용해서, 랜덤 노이즈를 가우시안에서 샘플링하고 이것을 더 작은 메트릭스에서 진행할 수 있게 된다! reparametrization trick과 함께 알아두면 좋은 트릭인 것 같다.
