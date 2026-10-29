# The Math Behind Causal Restructuring

This document explains what the causal restructuring pipeline computes and why. It follows the order of the code. Each section starts with the intuition, then gives the exact formula the code implements, with a pointer to where it lives.

The pipeline has two parts. **Part I** covers the original framework (Steps 1–6 of the design document). **Part II** covers the three extensions added later: least-squares reconstruction, a per-layer sensitivity budget and richer fingerprints. Part II also explains why the best configuration ends up looking the way it does.

---

## 0. Notation and the core idea

| Symbol | Meaning |
|---|---|
| $f$ | the trained teacher network |
| $\ell = 1,\dots,L$ | the layers whose units we compress (Linear layers in MLPs, conv layers in ResNets) |
| $h_\ell(x) \in \mathbb{R}^{n_\ell}$ | the post-activation output of layer $\ell$ on input $x$; one entry per unit (neuron or channel) |
| $z(x) \in \mathbb{R}^{C}$ | the teacher's logits |
| $p_T(x) = \mathrm{softmax}(z(x))$ | the teacher's class distribution |
| $X = \{x_1,\dots,x_N\}$ | a calibration set (a few hundred to a few thousand images) |
| $W_{\ell+1}$ | the weights of the layer that *consumes* $h_\ell$ |

**What compression means here.** We pick a subset $K_\ell \subset \{1,\dots,n_\ell\}$ of units to keep in each layer. Every other unit is removed: its activation is multiplied by 0 (`forward_with_mask`), and its weights are zeroed or folded into the survivors. The **keep fraction** is $\kappa = \sum_\ell |K_\ell| / \sum_\ell n_\ell$.

**The goal.** Choose the $K_\ell$ and fix up the remaining weights so that the small network $g$ behaves like the teacher, i.e. $g(x) \approx f(x)$ on the data distribution, *before* any retraining.

The problem splits into two sub-problems, and every step below belongs to one of them:

1. **Selection:** *which* units to keep (Steps 2–3, extension #2).
2. **Reconstruction:** how to change the remaining weights so the removed units' contribution is not lost (Step 4, extension #1).

---

# Part I — The framework

## Step 1. Teacher training

Training is standard supervised cross-entropy. Nothing in the math depends on how the teacher was trained. In practice, though, better-trained teachers have more redundant units and compress better (compare the 5-epoch and 30-epoch MLP results).

## Step 2. Measuring units: fingerprints and causal scores

Two numbers describe each unit $i$ in layer $\ell$: *what it does* (its fingerprint) and *how much the output depends on it* (its causal score).

### 2a. Activation fingerprint (what the unit does)

The fingerprint is the unit's response across the calibration set:

$$
\phi_i = \big(\bar h_i(x_1), \dots, \bar h_i(x_N)\big) \in \mathbb{R}^N
$$

For an MLP neuron, $\bar h_i(x)$ is just its activation. For a conv channel, the activation is a $H\times W$ map, so we take its **spatial mean**: $\bar h_i(x) = \frac{1}{HW}\sum_{p} h_i(x)_p$.

Two units with nearly parallel fingerprints, $\phi_i \approx c\,\phi_j$, respond to the same inputs. One of them is probably redundant.

*Code:* `compute_causal_fingerprints_and_scores` in [CMI_Method.py](../src/utils/CMI_Method.py).

### 2b. Causal ablation score (how much the output depends on it)

The score comes from an intervention: set unit $i$ to zero everywhere, run the network, and compare the logits to the unablated ones. Let $z^{(i)}(x)$ denote the logits with unit $i$ ablated. Then

$$
S_i \;=\; 1 - \frac{1}{N}\sum_{n=1}^{N} \cos\!\big(z(x_n),\, z^{(i)}(x_n)\big),
\qquad \cos(a,b) = \frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}.
$$

- $S_i \approx 0$: removing the unit barely changes the logits' direction, so the unit is unimportant.
- A larger $S_i$ means the unit carries information the rest of the network does not reproduce.

**This is causal, not correlational.** A unit with a large activation can still have $S_i \approx 0$ if downstream layers ignore it, or if another unit duplicates it. The score measures the *effect of removal*, which is exactly what compression needs to know.

**Limitation.** $S_i$ is a *single-unit* effect. If units $i$ and $j$ are exact duplicates, both get $S \approx 0$ because each covers for the other, yet removing both would hurt. That is why selection doesn't use the scores alone: clustering (Step 3) and the budget profile (extension #2) handle the joint effects.

The scores are computed in fp32. Many scores are $\sim 10^{-4}$, below fp16 resolution.

## Step 3. Selecting units: reserve, cluster, pick a medoid

### 3a. Reserve the top units

The top $r\%$ of units by $S_i$ form the **reserve set** $R$. They are kept unconditionally. The ranking can be done within each layer (`reserve_scope="layer"`, as the document specifies) or across the whole network (`"global"`). The remaining units are the **background** $B$.

### 3b. Drop dead units

A unit with $\lVert\phi_i\rVert \le 10^{-6}\max_j \lVert\phi_j\rVert$ never fires on the calibration data (typically a dead ReLU). It carries no signal, so it is excluded from the background before clustering.

This matters more than it looks. A zero vector has no direction, so its cosine distance to everything is undefined. Each dead unit would end up in its own singleton cluster and use up budget. Excluding them raised MLP accuracy at 10% kept from about 48% to about 80%.

### 3c. Cluster the background by cosine distance

Normalize each background fingerprint, $\hat\phi_i = \phi_i/\lVert\phi_i\rVert$, and run **average-linkage agglomerative clustering** with distance

$$
d(i,j) = 1 - \cos(\phi_i,\phi_j),
$$

into $k_\ell = \mathrm{round}\big(|B_\ell|\,(1-\rho)\big)$ clusters, where $\rho$ is the background reduction rate.

Why cosine distance rather than Euclidean? The next layer's weights can absorb any rescaling. If $\phi_j = c\,\phi_i$, then unit $j$ is replaceable by unit $i$ with its outgoing weights multiplied by $c$. Only the *direction* of a fingerprint matters.

### 3d. Choose each cluster's representative (causal medoid)

From each cluster $C$, exactly one unit $m$ survives. The choice balances being *typical* of the cluster against being *causally important*. With $\mu_C = \frac{1}{|C|}\sum_{j\in C}\hat\phi_j$, the code picks the unit minimizing

$$
J(i) \;=\; (1-w)\,\underbrace{\big(1-\cos(\hat\phi_i,\mu_C)\big)}_{\text{distance to centroid}}
\;+\; w\,\underbrace{\Big(1-\frac{S_i}{\max_{j\in C} S_j}\Big)}_{\text{causal penalty}}
\;+\; 10^{6}\cdot \mathbb{1}[\,i \text{ is dead}\,],
$$

with $w$ = `causal_weight` (default 0.5).

- $w=0$ selects the geometric medoid, which is what DeepAbstract does (it uses Euclidean k-means).
- $w=1$ selects the cluster's highest-scoring unit.
- The dead-unit penalty makes sure a dead unit is never chosen while an active member exists.

*Code:* `cluster_background_units`, `find_causal_medoid` in [CausalRestructuringMethod.py](../src/utils/CausalRestructuringMethod.py).

**Kept set:** $K_\ell = R_\ell \cup \{\text{medoids of layer } \ell\}$.

**Keep-fraction arithmetic.** With reserve $r\%$ and background reduction $\rho$, the kept fraction is

$$
\kappa = \frac{r + (100-r)(1-\rho)}{100}.
$$

The experiments state a target $\kappa$ instead, and pick $r$ and $\rho$ to hit it. With *reserve share* $s$ (the fraction of kept units that come from the reserve), the settings are $r = s\cdot 100\kappa$ and $1-\rho = (100\kappa - r)/(100 - r)$.

## Step 4. Reconstruction: merge and fold the weights (√|m| rule)

Removing units throws away their contribution to the next layer. Step 4 tries to put that contribution back. For each cluster $C$ with medoid $m$:

**Incoming averaging.** The medoid's incoming weights become the cluster average, so it represents the whole cluster:

$$
W_\ell[m,:] \leftarrow \frac{1}{|C|}\sum_{j\in C} W_\ell[j,:]
$$

The same averaging is applied to its bias and to its BatchNorm $\gamma,\beta$.

**Outgoing folding.** The removed members' outgoing columns are added onto the medoid's column, scaled down:

$$
W_{\ell+1}[:,m] \leftarrow W_{\ell+1}[:,m] + \frac{1}{\sqrt{|C|}}\sum_{j\in C\setminus\{m\}} W_{\ell+1}[:,j],
\qquad W_{\ell+1}[:,j]\leftarrow 0 \;\; (j\neq m).
$$

**Why $1/\sqrt{|C|}$?** Consider two extremes.
- If all members were *identical* ($h_j = h_m$), the exact fix is to add the columns at full weight (factor 1). The next layer's input $\sum_j W[:,j]h_j$ is then reproduced exactly.
- If the members were *uncorrelated* with equal variance, the summed signal has $\sqrt{|C|}$ times the standard deviation of one member, so full-weight folding would overshoot.

The $1/\sqrt{|C|}$ factor is a heuristic between these extremes. It is not optimal for any particular data distribution, and this is the weak point extension #1 replaces.

**Residual-stream exception (ResNets).** The output of a residual block's last conv is added into the skip path. Its consumers are the next block's first conv *and* every later block via the skip connection, so there is no single $W_{\ell+1}$ to fold into. These layers are excluded from folding (`get_foldable_layers()` in [resnet_cifar.py](../src/models/resnet_cifar.py)) and are only masked. The foldable layers are the stem conv and the internal convs of each block.

## BatchNorm recalibration (after Steps 3–4, before evaluation)

Masking and folding change the distribution of every downstream pre-activation. The stored BN running statistics $(\hat\mu,\hat\sigma^2)$ are then wrong, and ResNet accuracy collapses if they are left as they are.

The fix: reset the statistics and recompute them as an exact cumulative average (`momentum=None`) over 50 batches of the masked network, with no gradient steps. This is a pure re-estimation, $\hat\mu = \frac{1}{M}\sum_{b}\mu_b$ (and similarly for $\hat\sigma^2$). BN then absorbs any constant shift the reconstruction introduced.

## Step 5. Knowledge-distillation fine-tuning

This step is optional and was not used in the zero-shot comparisons. The loss is

$$
\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\Big(\mathrm{softmax}(z_T/T)\,\Big\|\,\mathrm{softmax}(z_S/T)\Big) + (1-\alpha)\,\mathrm{CE}(z_S, y),
\qquad T=2,\;\alpha=0.85.
$$

Note the direction: the KL is taken *from the teacher*, KL(teacher ‖ student), which is the standard distillation form and what the code computes. The design-doc slide writes it the other way around. The $T^2$ factor keeps gradient magnitudes comparable across temperatures.

Training uses an **EMA** of the weights, $\bar\theta \leftarrow \beta\bar\theta + (1-\beta)\theta$ with $\beta=0.98$. The EMA also averages the BN buffers. The final model is $\bar\theta$.

*Code:* `kd_loss_fn`, `EMAModel` in [FineTuningMethod.py](../src/utils/FineTuningMethod.py).

## Step 6. Evaluation: accuracy and agreement

$$
\text{acc} = \Pr[\arg\max g(x) = y], \qquad \text{agr} = \Pr[\arg\max g(x) = \arg\max f(x)].
$$

Agreement measures *faithfulness to the teacher*, which is the real goal of restructuring. A compressed model can reach the same accuracy while making different mistakes; agreement penalizes that.

**Sanity check used in every experiment.** If the student is right and disagrees with the teacher, the teacher must be wrong. So

$$
\text{acc} \le \text{agr} + \Pr[\text{teacher wrong}] \quad\Longrightarrow\quad \text{acc}-\text{agr} \le 100 - \text{acc}_T.
$$

A result that violates this is a bug. This check is what flagged the ResNet-50 numbers in `experimentOutput.log` as impossible.

---

# Part II — Extensions

## Extension #1. Least-squares reconstruction (`fold_mode="lsq"`)

**The idea.** Instead of guessing a folding factor, *solve* for the next layer's weights that best reproduce what it used to receive.

The next layer computes $W h + b$, where $h = h_\ell(x)$. After removal it can only see the kept units $h_K$. We want new weights $W'$ and bias $b'$ such that

$$
\min_{W',\,b'} \;\mathbb{E}_x \big\lVert\, W h + b - (W' h_K + b') \,\big\rVert^2 .
$$

Since $W$ is fixed, it is enough to find the best *linear prediction of the full layer from its kept units*, $h \approx A h_K + c$, and then set $W' = WA$ and $b' = b + Wc$. This is ordinary linear regression of each unit on the kept ones. With the layer mean $\mu = \mathbb{E}[h]$ and covariance $\Sigma = \mathrm{Cov}(h)$, the solution is

$$
A = \Sigma_{:,K}\,\big(\Sigma_{KK} + \lambda I\big)^{-1}, \qquad c = \mu - A\,\mu_K,
$$

with a small ridge $\lambda = 10^{-3}\cdot\mathrm{tr}(\Sigma_{KK})/|K|$ for numerical stability. In words:
- Row $i$ of $A$ says how to synthesize unit $i$ out of the kept units.
- For a kept unit, that row is (up to the tiny ridge) the indicator $e_i$, so kept units pass through unchanged.
- The constant $c$ captures each dropped unit's mean, which a linear map of zero-mean signals can't produce.

Then

$$
W' = W A \ \ (\text{only the kept columns are nonzero}),\qquad b' = b + W c .
$$

**For conv layers,** $\mu$ and $\Sigma$ are pooled over samples *and* spatial positions, giving channel-by-channel statistics. A conv is linear in its input channels at every kernel offset, so the same $A$ applies to each spatial tap: $W'_{o,k,u,v} = \sum_i W_{o,i,u,v}A_{i,k}$. This minimizes the per-position error. It ignores correlations between neighbouring positions, so for $3\times 3$ kernels it is a close approximation rather than exact.

If the next layer has no bias (conv → BN), the constant $Wc$ is dropped. BN recalibration re-estimates the mean, which absorbs it.

**Why this beats the √|m| rule:**
1. The √|m| rule rebuilds each dropped unit from *one* unit (its medoid) with a fixed factor. LSQ rebuilds it from *all* kept units, with weights estimated from data. A dropped unit split between two kept units is recovered, and so is one correlated with a reserved unit outside its cluster.
2. It is the optimal linear fix for the objective we care about, the next layer's input.
3. It keeps the kept units' incoming weights unchanged. Averaging them (the √|m| rule) changes what the medoid computes, which *adds* error.

Only the mean and covariance of each layer are needed. They come from one pass over 2048 training images (`collect_activation_stats`), followed by one $|K|\times|K|$ solve per layer.

*Code:* `apply_lsq_reconstruction`.

**Effect.** This is the single largest gain:
- MLP5 at 10% kept: 14% → 88%.
- ResNet-18 at 20% kept: 30% → 38%. ResNet gains less because its residual layers cannot be reconstructed, only masked.

## Extension #2. Per-layer sensitivity budget (`budget_mode="sensitivity"`)

**The idea.** A uniform keep ratio treats all layers as equally compressible, and they aren't. In ResNets, early high-resolution layers and the last stage behave very differently. So we *measure* how much each layer can lose, then distribute the global budget accordingly.

**Profile.** For each layer $\ell$ and each keep ratio $q$ on the grid $\{0.05, 0.1, 0.2, \dots, 1.0\}$, compress *only that layer* to ratio $q$ (same selection and reconstruction as the final run, other layers intact) and measure the distortion on 512 calibration images:

$$
D_\ell(q) = \frac{1}{N}\sum_n \mathrm{KL}\big(p_T(x_n)\,\|\,p_{g_{\ell,q}}(x_n)\big).
$$

KL is used rather than accuracy because it changes smoothly even when no prediction flips, which makes it a usable signal at high keep ratios.

Removing more units should never help, so each curve is made monotone: $\tilde D_\ell(q) = \max_{q'\ge q} D_\ell(q')$.

**Allocate.** We want

$$
\min_{q_1,\dots,q_L}\ \sum_\ell \tilde D_\ell(q_\ell) \quad\text{s.t.}\quad \sum_\ell n_\ell\, q_\ell = \kappa \sum_\ell n_\ell .
$$

This assumes the distortions of different layers roughly *add up*, a first-order approximation that holds when each layer's damage is small.

The solver is the classic greedy marginal-cost method. Start with every layer fully kept. At each step, move one grid notch down in the layer with the smallest

$$
\frac{\Delta D}{\Delta \text{units}} = \frac{\tilde D_\ell(q_\text{next}) - \tilde D_\ell(q_\text{cur})}{(q_\text{cur} - q_\text{next})\,n_\ell},
$$

i.e. the cheapest damage per unit removed. Repeat until the budget is reached; the last step is interpolated so the total is hit exactly. When the curves are convex this greedy is optimal. Otherwise it is a good heuristic.

Within each layer, a fraction $s$ (the reserve share) of the $q_\ell n_\ell$ kept units are the top-$S_i$ units. The rest are cluster medoids.

*Code:* `profile_layer_sensitivity`, `allocate_layer_budget`.

**Effect.** It matters most on deep networks at aggressive compression:
- ResNet-18 at 10% kept: 19% → 32% on top of LSQ.
- ResNet-18 at 10% kept with $s=1$: 43%.
- It does little for the 3-layer MLP, which has only two hidden layers to trade between.

## Extension #3. Richer fingerprints (`fp_samples`, `fp_pool`)

Fingerprints were extended to more samples and to a $g\times g$ spatial grid instead of the global spatial mean, so $\phi_i\in\mathbb{R}^{N g^2}$. The aim was to separate channels that respond to the same images but in different locations.

**Effect: none measurable.** The likely reason is that with LSQ, the *exact* cluster membership matters little: the regression rebuilds dropped units from all kept units anyway. Finer fingerprints only change the clustering, which is the part LSQ makes unimportant.

---

## Why the best configuration has no clustering on ResNet

The best ResNet-18 setting was **LSQ + budget with reserve share $s=1$**. Every kept unit is simply a top-$S_i$ unit of its layer, the per-layer counts come from the sensitivity budget, and LSQ rebuilds everything else. Clustering is not used at all.

Here is how the pieces divide the work:

| Component | Role | Evidence |
|---|---|---|
| Causal score $S_i$ | ranks units *within* a layer | top-score + LSQ ≫ k-means + LSQ ≫ random + LSQ |
| Sensitivity budget | decides *how many* units each layer keeps | +13 to +23 points at 10% kept |
| LSQ reconstruction | recovers the dropped units' contribution | the largest gain on MLPs; large on ResNet |
| Clustering + medoid | enforces diversity among kept units | helps slightly on MLPs at 20–30% kept, hurts on ResNet |

Clustering exists to stop the kept set from being full of near-duplicates. But a near-duplicate of a kept unit has a low causal score (the kept copy covers for it, see the Step 2b limitation), so ranking by $S_i$ already avoids duplicates to some extent. Whatever redundancy remains, LSQ resolves by reconstructing from the correlated survivors.

**How DeepAbstract differs.** DeepAbstract (Ashok et al. 2020) clusters activations with Euclidean k-means, keeps the unit nearest each centroid, and sums the outgoing weights with factor 1. It has neither a causal ranking nor a data-fitted reconstruction. On ResNet it can only fold into Linear layers. That explains why it falls behind sharply below 50% kept.

## Summary of results (zero-shot, no fine-tuning)

| Model | Keep | DeepAbstract | Original method (√\|m\|) | Best (LSQ + budget) |
|---|---|---|---|---|
| MLP-5 (UCI digits, 5 seeds) | 10% | 42.6 | 14.3 | **93.4** |
| MLP-5 | 30% | 84.1 | 73.1 | **97.8** |
| ResNet-18 (CIFAR-10, 1 teacher) | 10% | 13.4 | 18.0 | **42.9** |
| ResNet-18 | 30% | 28.6 | 47.3 | **70.7** |
| ResNet-18 | 50% | 66.0 | 73.2 | **84.8** |

The teachers reach 98.2% (MLP-5) and about 90% (ResNet-18) test accuracy.

**Caveats.**
- The ResNet results come from a single teacher, so they have no error bars.
- DeepAbstract still uses the older BN recalibration, which may make its numbers worse than they should be.

## Using it from code

`main.py` does not expose the extensions yet. Call the function directly:

```python
from src.utils.CausalRestructuringMethod import perform_causal_restructuring

sub_model, kept, mask, zs = perform_causal_restructuring(
    teacher_model=model, val_loader=val_loader, test_loader=test_loader,
    top_n_pct=10.0,          # with reserve share 1.0: keep 10% of units, all by causal score
    reduction_rate=1.0,      # background fully removed (no clustering)
    fold_mode="lsq",         # extension #1
    budget_mode="sensitivity",  # extension #2
    calib_loader=train_loader,
    device=device,
)
```

## Code map

| Concept | Function | File |
|---|---|---|
| Fingerprints $\phi_i$, scores $S_i$ | `compute_causal_fingerprints_and_scores` | [CMI_Method.py](../src/utils/CMI_Method.py) |
| Spatial-grid fingerprints | `compute_activation_fingerprints` | [CMI_Method.py](../src/utils/CMI_Method.py) |
| Reserve, cluster, medoid | `select_layer_reserves`, `cluster_background_units`, `find_causal_medoid` | [CausalRestructuringMethod.py](../src/utils/CausalRestructuringMethod.py) |
| √\|m\| merge/fold | `apply_step4_weight_merging` | same |
| $\mu,\Sigma$ and LSQ | `collect_activation_stats`, `apply_lsq_reconstruction` | same |
| Sensitivity budget | `profile_layer_sensitivity`, `allocate_layer_budget` | same |
| BN recalibration, acc/agr | `recalibrate_batchnorm`, `evaluate_masked_model` | same |
| KD + EMA | `kd_loss_fn`, `EMAModel` | [FineTuningMethod.py](../src/utils/FineTuningMethod.py) |
| Foldable ResNet layers | `get_foldable_layers` | [resnet_cifar.py](../src/models/resnet_cifar.py) |
