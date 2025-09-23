# ProRL V2 - Prolonged Training Validates RL Scaling Laws

Authors: [Jian Hu](https://hijkzzz.notion.site/), [Mingjie Liu](https://research.nvidia.com/person/mingjie-liu), [Shizhe Diao](https://shizhediao.github.io/), [Ximing Lu](https://gloriaximinglu.github.io/), [Xin Dong](https://simonxin.com/), [Pavlo Molchanov](https://www.pmolchanov.com/), [Yejin Choi](https://yejinc.github.io/), [Jan Kautz](https://jankautz.com/), [Yi Dong](https://developer.nvidia.com/blog/author/yidong/) 

First Published: August 11, 2025

---

## Introduction

One of the most compelling questions in AI is whether large language models (LLMs) can continue to improve through sustained reinforcement learning (RL), or if their capabilities will eventually plateau.

ProRLv2 is the latest evolution of our [Prolonged Reinforcement Learning (ProRL)](https://arxiv.org/abs/2505.24864) regime, specifically designed to test the effects of extended RL training on LLMs. Leveraging advanced algorithms, rigorous regularization, and comprehensive domain coverage, ProRLv2 pushes the boundaries well beyond typical RL training schedules. Our experiments systematically explore whether models can achieve measurable progress when subjected to thousands of additional RL steps.

Today, we're excited to announce the release of ProRLv2, building on the foundation of our earlier [ProRL](https://arxiv.org/abs/2507.12507) work. In this update, we'll explore its key innovations, advanced methods, and new empirical results that achieve **new state-of-the-art**—shedding light on how large language models can continue to learn and improve.

## What Sets ProRL Apart?

Most approaches—chain-of-thought prompting, tree search—help models better exploit knowledge they already possess. RL, especially with rigorous, programmatically-verifiable rewards, holds the promise to push models into genuinely new territory. However, traditional short-horizon RL techniques often suffer from instability and quickly diminishing returns, earning a reputation as “temperature distillation” rather than a true enabler of boundary expansion.

**ProRL fundamentally challenges this paradigm:**

- **Extended training:** Over 3,000 RL steps across **five distinct domains**, achieving new state‑of‑the‑art performance among 1.5 B reasoning models.
- **Stability and robustness:** Incorporates KL-regularized trust regions, periodic reference policy resets, and scheduled length regularization.
- **Fully verifiable rewards:** Every reward signal is determined programmatically and is always checkable.
- **Brevity enforced:** Scheduled cosine length penalties ensure outputs remain concise and efficient.

### Comparison Table

| **Conventional RL training** | **What ProRL Does** |
| --- | --- |
| Few-hundred steps, one domain | **3,000+ steps, five domains** |
| Entropy collapse, KL spikes | PPO-Clip, REINFORCE++-baseline, Clip-Higher, Dynamic Sampling, Reference resets |
| Risky reward model drift | *Fully verifiable* rewards |
| Verbose, lengthy outputs | Scheduled cosine length penalty |

*Goal: Move beyond re-sampling familiar solutions to genuinely expanding what the model can discover.*

## Core Techniques: ProRL Algorithms & Regularizers

### 1. Proximal Policy Optimization (PPO-Clip) with [REINFORCE++-baseline](https://medium.com/@janhu9527/reinforce-baseline-is-all-you-need-in-rlvr-f5406930aa85)

At ProRL’s core is the **clipped PPO loss**, which stabilizes policy updates by restricting how much the new policy can diverge from the old ones:

$\mathcal{L}_\mathrm{PPO}(\theta) = \mathbb{E}_\tau\bigg[\min\Big( r_\theta(\tau) A(\tau),\mathrm{clip}\big(r_\theta(\tau), 1 - \varepsilon_\mathrm{low}, 1 +\varepsilon_\mathrm{high}\big) A(\tau) \Big)\bigg]$

where:

- $r_\theta(\tau) = \frac{\pi_\theta(\tau)}{\pi_{\text{old}}(\tau)}$
- $\tilde{R}_\tau = R_\tau - \mu_{\text{group}},\  
\mu_{\text{group}} = \operatorname{mean}_{\text{group}}(R_\tau)$
    
    $A(\tau) = \frac{\tilde{R}\tau - \mu_{\text{batch}}}{\sigma_{\text{batch}}},\ 
    \mu_{\text{batch}} = \operatorname{mean}_{\text{batch}}(\tilde{R}_\tau),\
    \sigma_{\text{batch}} = \operatorname{std}_{\text{batch}}(\tilde{R}_\tau)$
    

*"group"* refers to all generated responses for the same prompt. “batch” refers to the rollout global batch.

**Global Batch Normalization** in the **REINFORCE++-baseline** helps prevent value instability caused by small group sizes: it first subtracts the mean reward of the small group ($\mu_\text{group}$) to reshape the rewards, therefore, the algorithm is not insensitive to reward patterns such as 0 (incorrect) / 1 (correct) / -0.5 (format reward) or -1 (incorrect) / 1 (correct) / -0.5 (format reward), and then applies global batch normalization.

### 2. [Clip-Higher & Dynamic Sampling](https://arxiv.org/abs/2503.14476)

- **Clip-Higher:** Use a higher upper bound of PPO’s clipping range to mitigate policy entropy collapse and promote sampling diversity ($\varepsilon_{\text{high}} > \varepsilon_{\text{low}}$).
    
    PPO-Clip bounds:
    
    $\varepsilon_{\text{low}} = 0.20 \quad \varepsilon_{\text{high}} = 0.28$
    
- **Dynamic Sampling:** Discards prompts with group responses with all 1 (fully correct) or 0 (fully incorrect) rewards to reduce noise in gradient estimates.

### 3. Scheduled [Cosine Length Penalty](https://arxiv.org/pdf/2502.03373)

To promote concise, token-efficient outputs, a scheduled cosine penalty is applied:

$\text{length\_reward}(t) = \eta_{\min} + 0.5 \times (\eta_{\max} - \eta_{\min}) \times [ 1 + \cos ( \pi t / T ) ]$

where:

- $t$  = current output length (tokens)
- $T$  = context token limit
- $\eta_\text{min}$ ,  $\eta_\text{max}$  = reward/penalty boundaries

**Reward update:**

$R'_\tau = R_{\text{correct}} + \lambda_\text{len} \cdot \eta_\text{len}(t)$

The penalty cycles on and off at regular intervals (e.g., 100 updates on, 500 off) to balance informativeness and conciseness.

### 4. KL Regularization & Reference Policy Resets

A KL penalty keeps the policy close to a reference. Periodic resets help prevent overfitting and ensure stability:

$\mathcal{L}_\mathrm{KL\text{-}RL} = \mathcal{L}_\mathrm{PPO} - \beta\, D_\mathrm{KL}(\pi_\theta\ \|\ \pi_\mathrm{ref})$

KL divergence in REINFORCE++-baseline is regularized using a $k_2$ estimator:

$\mathcal{L}_{k_{2}} = \mathbb{E}_{s \sim D,\ a \sim \pi_{\theta_{\text{old}}}(\cdot|s)} \left( \frac{1}{2} ( -\log x )^2 \right)$

with

$x = \exp\left(\mathrm{clamp}\left(\log\frac{\pi_{\text{ref}}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, -10,\, 10\right)\right)$

where the function $\mathrm{clamp}(z, -10, 10)$ limits $z$ to the range $[-10, 10]$ to improve the value stability.

**Reference resets:** Every 200–500 RL steps (or upon KL spikes/stalled validation), the reference policy $\pi_\mathrm{ref}$ is reset to the current policy, optimizer state is not cleared.

By regularly resetting the reference policy, the model avoids being constrained by outdated guidance, allowing it to continue learning effectively.

The scheduled cosine length penalty, applied periodically, also plays an important role. By cycling the penalty on and off, the model avoids becoming trapped in short or fixed context lengths, enabling it to improve both the accuracy and token efficiency of its outputs. Together, these two strategies prevent the model from being limited by either the reference policy or context length, supporting continuous improvements in accuracy and overall performance over time.

## What Did We Discover?

- **New SOTA performance:** Performance continually improves with more RL training steps, ProRL‑3k sets a new SOTA for 1.5B reasoning models.
- **Sustained, non-trivial improvement:** Both Pass@1 and pass@k metrics climb over thousands of RL steps, expanding the base model’s reasoning boundary..
- **Creative and novel solutions:** ProRL outputs show reduced n-gram overlap with pretraining data, indicating true innovation rather than rote memorization.
- **Boundary breakthroughs:** On tasks where base models always failed, ProRL not only achieves strong pass rates, but also demonstrates robust out-of-distribution generalization.

## Comprehensive Results

ProRL was evaluated across math, code generation, and diverse reasoning gym benchmarks. Scores are reported for:

- **Base:** DeepSeek-R1-Distill-Qwen-1.5B
- **ProRL-2k:** 2,000 RL steps (trained with **16k context**)
- **ProRL-3k:** 3,000 RL steps (trained with **8k context**)

As of the time of this blog post, the model is still undergoing continuous training and accuracy improvements. The chart illustrates the performance gains of the 2k steps model over the base model and of the 3k steps model over the 2k steps  model. It shows that, even with the training context length cut in half (16k to 8k) — greatly reducing computational cost — overall model accuracy improves across the tasks.

### Mathematics & GPQA & IFEVAL evaluation results

![image.png](https://hijkzzz.notion.site/image/attachment%3A1fb6308a-0a57-4c63-bcfd-5d9f8d5fc9ee%3Aimage.png?table=block&id=24cd9a33-ecc9-80a6-ab1c-d46f6e0027d2&spaceId=7943afbd-7511-4cc5-9381-e6d435095431&width=2000&userId=&cache=v2)

### Code Generation evaluation results

![image.png](https://hijkzzz.notion.site/image/attachment%3A8d91a205-66be-41b1-979f-17703642ec69%3Aimage.png?table=block&id=24cd9a33-ecc9-80d3-8be8-c5ecf5f7f2df&spaceId=7943afbd-7511-4cc5-9381-e6d435095431&width=2000&userId=&cache=v2)

### Reasoning Gym evaluation results

![image.png](https://hijkzzz.notion.site/image/attachment%3Acbde70d1-c271-4a19-9bc9-6c86e22760d1%3Aimage.png?table=block&id=24cd9a33-ecc9-80c3-a5c2-ce991a48eb3e&spaceId=7943afbd-7511-4cc5-9381-e6d435095431&width=2000&userId=&cache=v2)

### **Average Output Length of the models**

![image.png](https://hijkzzz.notion.site/image/attachment%3Ad170e170-2955-4538-83f4-e61f82030f6a%3Aimage.png?table=block&id=24cd9a33-ecc9-80a3-a602-f2ccab9c4b5d&spaceId=7943afbd-7511-4cc5-9381-e6d435095431&width=2000&userId=&cache=v2)

### Key Takeaways

- **New world’s best 1.5B reasoning model,** our ProRL-3k sets a new state of the art, significantly outperforms its base model, DeepSeek-R1-1.5B, and surpasses the previous SOTA achieved by our own ProRL-2k.
- **ProRL delivers sustained, reliable improvements** across math, code, and reasoning—particularly in domains where base models (even with aggressive sampling) fail outright.
- **More compute > more parameters:** Pushing RL steps further—rather than just scaling up model size—drives substantially more boundary expansion.
- **Gains are robust:** Improvements are not isolated flukes; almost every subtask benefits from continued RL.

## **In summary**

Our empirical results indicate that large language models can achieve sustained improvements in math, code, and reasoning tasks through prolonged reinforcement learning, surpassing the performance typically observed with conventional training routines. Our evaluation demonstrates robust gains across a wide array of benchmarks—including challenging and out-of-distribution tasks—suggesting that extended RL training can meaningfully expand a model’s reasoning capabilities.

For practitioners aiming to push the boundaries of model performance or explore the reasoning potential of LLMs, ProRL offers a reproducible foundation and a practical training recipe. With open-source models and benchmarks available, the community is encouraged to further explore and validate these findings as part of ongoing research into the limits and opportunities of reinforcement learning for large language models.

## Try It Yourself