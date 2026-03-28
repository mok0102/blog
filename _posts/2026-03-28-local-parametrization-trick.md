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

$$
\mathcal{E}_p(x)
$$