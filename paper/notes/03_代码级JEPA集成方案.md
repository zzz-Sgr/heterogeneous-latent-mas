# Vision Wormhole 代码级 JEPA 集成方案

> 基于完整源码阅读（14个文件，~7000行代码）

---

## 第一部分：源码架构速览

### 核心数据流

```
Input Question
  │
  ▼
[Planner Agent] ──text prompt──► VLM ──latent rollout──► hidden states z₁
  │                                                         │
  │                                              LatentToUniversalEncoder(z₁) → U₁
  │                                                         │
  │                                              affine_align_out(U₁) → U₁_ref
  │                                                         │
  ▼                                                       
[Critic Agent] ◄── U₁_ref 注入视觉通路 ── VLM ──► z₂
  │                                          
  │                              LatentToUniversalEncoder(z₂) → U₂ → U₂_ref
  ▼
[Refiner Agent] ◄── [U₁_ref, U₂_ref] ── VLM ──► z₃ → U₃ → U₃_ref
  │
  ▼
[Judger Agent] ◄── [U₁_ref, U₂_ref, U₃_ref] ── VLM ──► text answer
```

### 关键代码位置

| 组件 | 文件 | 行号 | 作用 |
|------|------|------|------|
| `LatentToUniversalEncoder` | `methods/vision_latent_mas_codec_new.py` | 539-601 | latent z → universal U |
| `UniversalToVisionDecoder` | 同上 | 604-660 | universal U → delta + gate |
| `_generate_latents` | 同上 | 1630-1718 | 自回归产生 latent embeddings |
| `_inject_memory_into_mm_inputs` | 同上 | 1490-1624 | 注入 U_ref 到接收方视觉通路 |
| `_run_agent_latent` | 同上 | 1919-2021 | 单个 agent 推理主循环 |
| `_train_one_model` | `train_vision_latent_mas_codec_new.py` | 1348-1801 | Codec 训练主循环 |
| `_compute_kl_loss` | 同上 | 201-229 | KL distillation loss |
| 训练 loss 公式 | 同上 | 1691-1713 | `loss = mse_w*loss_mse + kl_w*loss_kl + stats_w*loss_stats` |

### 注入公式（关键）

```python
# 文件: methods/vision_latent_mas_codec_new.py, 行 1530-1540
inputs_embeds[b, pos, :] = base_img + gate * delta
```

- `base_img`: VLM 视觉编码器对 dummy image 的输出
- `delta`: UniversalToVisionDecoder 从 universal tokens 解码出的扰动
- `gate`: sigmoid 学习的标量，控制注入强度

---

## 第二部分：6 个精确的 JEPA 注入点

### 注入点 1：`_generate_latents` 加 JEPA-SCORE 质量评分

**位置**：[`methods/vision_latent_mas_codec_new.py:1717`](methods/vision_latent_mas_codec_new.py#L1717)

**现状**：
```python
# 行 1688-1717
for _ in range(int(latent_steps)):
    latent_vec = wrapper._apply_latent_realignment(last_hidden, text_model)
    latents.append(latent_vec.detach())
    # ... autoregressive forward ...
latent_stack = torch.stack(latents, dim=1)
return past, latent_stack  # ← 只返回 latent，没有质量分
```

**JEPA 改动**：在返回 latent_stack 后，额外返回质量分数：

```python
# 新增: JEPA-SCORE quality scoring
def _compute_jepa_score(self, wrapper, latent_vec, text_model):
    """用 Jacobian 行列式近似计算密度分数"""
    latent_vec.requires_grad_(True)
    # 将 latent 喂回 text_model 的最后一个 transformer block
    # 计算 ∂output/∂latent 的 Jacobian
    # 用随机投影近似 log|det(J)|
    ...

# 在 _generate_latents 返回前:
quality_scores = self._compute_jepa_score(wrapper, latent_stack, text_model)
return past, latent_stack, quality_scores
```

**改动量**：~30 行新增代码，`_generate_latents` 返回值从 2 元组变 3 元组。

**下游影响**：
- `_run_agent_latent` (行 1919) 需要解包 3 个值
- `_encode_latents_to_universal_ref` 需要接收 quality_scores
- 最终在 `_inject_memory_into_mm_inputs` 中 gate 使用 `gate * quality_score`

---

### 注入点 2：`UniversalToVisionDecoder` 输出 (μ, Σ) 替代确定性点

**位置**：[`methods/vision_latent_mas_codec_new.py:604-660`](methods/vision_latent_mas_codec_new.py#L604-L660)

**现状**：
```python
class UniversalToVisionDecoder(nn.Module):
    def forward(self, U):
        # U: [B, K, D]
        kv = self.kv_ln(U)
        q = self.q_img.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        q = self.out_ln(q)
        delta = self.out_proj(q)          # ← 单点输出 [B, K_img, H]
        gate = torch.sigmoid(self.gate_mlp(pooled))  # ← 标量 gate
        return delta, gate
```

**JEPA 改动**：输出 (μ, log_Σ) 而非单点 delta

```python
class GaussianUniversalToVisionDecoder(nn.Module):
    def __init__(self, d_univ, h_out, k_img, ...):
        super().__init__()
        # 原有结构保持不变
        self.out_proj_mu = nn.Linear(d_univ, h_out)      # 均值
        self.out_proj_logvar = nn.Linear(d_univ, h_out)  # log 方差
        # gate 改为基于不确定性自动调节: gate = sigmoid(-mean(logvar))

    def forward(self, U):
        # ... 相同 cross-attention ...
        mu = self.out_proj_mu(q)           # [B, K_img, H]
        logvar = self.out_proj_logvar(q)   # [B, K_img, H]
        
        # gate 与不确定性负相关
        uncertainty = logvar.mean(dim=(1,2), keepdim=True)
        gate = torch.sigmoid(-uncertainty)  # 不确定性高 → gate 小
        
        return mu, logvar, gate
```

**注入公式变为**：
```python
# 原: inputs_embeds[b, pos, :] = base_img + gate * delta
# 新: 
std = torch.exp(0.5 * logvar)
noise = torch.randn_like(mu) * std
inputs_embeds[b, pos, :] = base_img + gate * (mu + noise)
```

**训练改动**：loss 加 KL 正则，防止 Σ 退化：
```python
# 新增 loss: KL(N(μ,Σ) || N(0,I))
loss_kl_gaussian = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
```

**改动量**：新增模块 ~60 行，训练代码 +10 行，推理代码 +5 行。

---

### 注入点 3：训练加 anti-collapse loss

**位置**：[`train_vision_latent_mas_codec_new.py:1691-1713`](train_vision_latent_mas_codec_new.py#L1691-L1713)

**现状**：
```python
loss = loss_mse_w * loss_mse + loss_kl_w * loss_kl + loss_stats_w * loss_stats
```

**JEPA 改动**：加 anti-collapse term

```python
# 新增: anti-collapse loss (VICReg-style)
def _anti_collapse_loss(U):
    """U: [B, K, D] universal tokens"""
    # Variance regularization: 每个维度的方差应 > 1
    std = U.std(dim=0).mean()
    loss_var = F.relu(1.0 - std)
    
    # Covariance regularization: 不同维度应去相关
    U_centered = U - U.mean(dim=0, keepdim=True)
    cov = (U_centered.transpose(1, 2) @ U_centered) / (B * K - 1)  # [D, D]
    cov.diagonal().zero_()
    loss_cov = cov.pow(2).sum() / D
    
    return loss_var + loss_cov

# 新 loss:
loss_anti_collapse = _anti_collapse_loss(U_flat)  # U_flat 来自 encoder 输出
loss = loss_mse_w * loss_mse + loss_kl_w * loss_kl + loss_stats_w * loss_stats + loss_ac_w * loss_anti_collapse
```

**为什么这个位置**：U_flat 在训练代码中已经计算出来了（行 1546 `U_flat = enc(lat_flat)`），anti-collapse loss 零额外前向开销。

**改动量**：~20 行代码，训练参数 +1 (`--vision_codec_loss_anti_collapse`)。

---

### 注入点 4：OOD 检测 — 基于 latent 密度统计

**位置**：[`methods/vision_latent_mas_codec_new.py:1429-1440`](methods/vision_latent_mas_codec_new.py#L1429-L1440)

**现状**：
```python
def _encode_latents_to_universal_ref(self, model_idx, latents):
    U = self.encoders[model_idx](lat)              # [B, K_univ+2, D]
    U_ref = _apply_affine(U, W, b)                  # 仿射对齐到参考空间
    return U_ref.detach().float().cpu()
```

**JEPA 改动**：额外返回 OOD 分数

```python
def _encode_latents_to_universal_ref(self, model_idx, latents):
    U = self.encoders[model_idx](lat)
    U_ref = _apply_affine(U, W, b)
    
    # 新增: OOD detection
    # 计算 U 在训练分布中的密度估计
    # 如果分布显著偏离，标记为 OOD
    density_score = self._estimate_density(U, model_idx)
    ood_flag = density_score < self.ood_thresholds[model_idx]
    
    return U_ref.detach().float().cpu(), density_score, ood_flag
```

**OOD 阈值标定**（在训练完成后、推理前）：
```python
# 用训练数据跑一遍，记录 density_score 分布
# threshold = mean - 3*std（或百分位数）
```

**改动量**：~40 行 + 标定步骤。

---

### 注入点 5：通信链路 Jacobian 可解释性

**位置**：推理后处理，不修改核心代码

**做法**：
```python
@torch.no_grad()
def analyze_communication(U_ref_sequence):
    """分析一次通信中的信息流"""
    # U_ref_sequence: List[Tensor], 每个是 [B, K+2, D]
    results = {}
    for step, U in enumerate(U_ref_sequence):
        # SVD 分析
        U_flat = U.reshape(-1, D)
        _, S, V = torch.svd(U_flat, some=False)
        # 有效秩
        effective_rank = (S / S.max() > 0.01).sum().item()
        # 主方向
        principal_direction = V[:, 0]
        results[f"step_{step}"] = {
            "effective_rank": effective_rank,
            "top5_singular_values": S[:5].tolist(),
            "principal_direction_norm": principal_direction.norm().item(),
        }
    return results
```

**价值**：
- 有效秩下降 → 隐空间可能坍塌
- 奇异值突然变化 → 通信异常
- 主方向漂移 → sender 推理模式改变

**改动量**：~30 行新文件 `analysis.py`。

---

### 注入点 6：JEPA 自监督训练替代 teacher-student

**位置**：[`train_vision_latent_mas_codec_new.py:1348-1801`](train_vision_latent_mas_codec_new.py#L1348-L1801)

**现状**（teacher-student distillation）：
```
teacher(hidden) ──► teacher_logits ──┐
                                     ├── KL loss ──► 优化 codec
student(injected) ──► student_logits ─┘
```

**JEPA 替代方案**：

```python
# 不再需要 teacher forward
# 对同一个 anchor text，用两个不同的 dropout seed 或 noise 做两次 rollout
latents_a = generate_latents_with_noise(wrapper, ids, mask, noise_seed=0)
latents_b = generate_latents_with_noise(wrapper, ids, mask, noise_seed=1)

U_a = enc(latents_a)
U_b = enc(latents_b)  # stop_gradient

# JEPA prediction loss: 从 U_a 预测 U_b
pred_U_b = predictor(U_a)  # 轻量 MLP predictor
loss_pred = F.mse_loss(pred_U_b, U_b.detach())

# Anti-collapse loss
loss_ac = anti_collapse_loss(U_a)

# Total loss
loss = loss_pred + lambda_ac * loss_ac
```

**优势**：
- 完全自监督，不需要 text teacher
- 同时训练了 JEPA-SCORE 密度估计能力
- 训练流程与原始类似，只是替换 loss 计算部分

**改动量**：~80 行（主要是训练 loop 改 loss 部分）。

---

## 第三部分：最小可行实验（推荐先做注入点 1）

### 实验设计

**目标**：验证 JEPA-SCORE 能否区分正常/噪声/对抗消息。

**步骤**：

1. **加载已训练的 Vision Wormhole checkpoint**
   ```python
   ckpt = torch.load("checkpoints/codec_xxx.pt")
   enc = LatentToUniversalEncoder(...)
   enc.load_state_dict(ckpt["encoders"]["model_name"])
   ```

2. **构造三类消息**
   ```python
   # 正常消息: 用训练数据跑 encoder
   normal_U = enc(normal_latents)
   
   # 噪声消息: 随机 latent
   noise_U = enc(torch.randn_like(normal_latents))
   
   # 对抗消息: 对正常 latent 加小扰动
   adv_U = enc(normal_latents + 0.1 * torch.randn_like(normal_latents))
   ```

3. **计算 JEPA-SCORE（简化为方差/余弦距离）**
   ```python
   def jepa_score_simple(U):
       """简化版: 用 U 的各向异性程度作为密度近似"""
       U_norm = F.normalize(U, dim=-1)
       # 余弦相似度矩阵
       sim = U_norm @ U_norm.transpose(1, 2)  # [B, K, K]
       # 同类 token 的相似度应远高于异类
       intra = sim.diagonal(dim1=1, dim2=2).mean()
       inter = (sim.sum(dim=-1) - sim.diagonal(dim1=1, dim2=2)).mean() / (K-1)
       return (intra - inter).item()
   ```

4. **观察三类消息的 score 分布差异**

**期望结果**：
- normal_U 的 score 显著高于 noise_U 和 adv_U
- 如果成立 → JEPA-SCORE 可以用于通信质量门控

**代码位置**：不修改源码，新写一个 `experiments/jepa_score_validation.py`。

---

## 第四部分：技术风险评估

| 风险 | 严重程度 | 缓解方案 |
|------|---------|---------|
| Vision Wormhole UVC 不是严格 JEPA 架构，Jacobian 密度估计可能不准 | 中 | 用简化版 score（方差+余弦距离替代 Jacobian 行列式）；先做注入点 1 验证 |
| Gaussian Codec 训练不稳定（Σ → 0） | 中 | 加 KL 正则 `N(μ,Σ)‖N(0,I)` 防止退化 |
| Anti-collapse loss 与现有 loss 竞争导致 mse/kl 变差 | 低 | 小权重 λ=0.01 起步，warmup 5 steps |
| 训练文本 channel 代码已有收敛问题（BF16 + grad NaN 处理） | 中 | JEPA 自监督理论上比 teacher-student 更稳定（无 logit 数值问题） |

---

## 第五部分：实施优先级

```
第 1 周: 注入点 1 (JEPA-SCORE 质量评分) — 验证可行性
第 2 周: 注入点 5 (可解释性分析) — 零风险，辅助调参
第 3 周: 注入点 4 (OOD 检测) — 基于第 1 周的结果
第 4-6 周: 注入点 2 (Gaussian Codec) — 主要贡献
后续: 注入点 3 (anti-collapse loss) → 注入点 6 (JEPA 自监督训练)
```

---

## 可直接运行的代码片段

### 打印 U 分布的统计量（用于判断是否需要 anti-collapse）

```python
# 在 VisionLatentMASMethodCODECNew.run_item() 中添加
def _debug_universal_stats(self, U_ref, step_name):
    """诊断 universal space 是否坍塌"""
    U = U_ref.float()
    # 各 token 的方差
    token_std = U.std(dim=0).mean().item()
    # 各维度的方差
    dim_std = U.std(dim=-1).mean().item()
    # token 间 cosine similarity
    U_norm = F.normalize(U, dim=-1)
    cos_sim = (U_norm @ U_norm.transpose(1, 2)).mean().item()
    print(f"[JEPA-DEBUG] {step_name}: token_std={token_std:.4f} dim_std={dim_std:.4f} cos_sim={cos_sim:.4f}")
```

**诊断标准**：
- token_std < 0.01 → 所有 token 几乎相同 → collapse
- cos_sim > 0.95 → token 间高度相似 → near-collapse
- dim_std < 0.01 → 某些维度死亡 → representation collapse
