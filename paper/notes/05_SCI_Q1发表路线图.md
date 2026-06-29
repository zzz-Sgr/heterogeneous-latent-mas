# SCI Q1 发表路线图：根治异质VLM隐空间维度坍塌

> 目标期刊：NeurIPS / ICML / ICLR / CVPR / ACL
> 核心贡献：发现并根治 Vision Wormhole UVC 的维度坍塌问题
> 当前状态：小样本验证完成（Eff Rank 2.32 → 75.53），全量训练进行中

---

## 一、论文定位与贡献

**Title（草案）:**
"Dimensional Collapse in Heterogeneous VLM Latent Communication: Diagnosis, Understanding, and a JEPA-Based Cure"

**三项核心贡献：**

| # | 贡献 | 当前状态 |
|---|------|:---:|
| C1 | 首次发现并定量证明 Vision Wormhole UVC 存在严重维度坍塌 | ✅ 已有数据 |
| C2 | 揭示坍塌根因：teacher-student 蒸馏的退化捷径 | ✅ 已有分析 |
| C3 | 提出 I-JEPA 自监督预训练替代 teacher-student，根治坍塌 | 🔄 需补 decoder + 下游 |

---

## 二、完整实验矩阵

### 2.1 必做实验（论文主体，Table 1-4）

```
Table 1: Collapse Diagnosis
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│                     │ Eff Rank │ RankMe   │ CosSim   │ Dead Dims│
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Original (small)    │   2.32   │    —     │  0.999   │    —     │
│ Original (full)     │   3.01   │   1.02   │  0.999   │    2     │
│ VICReg λ=0.1 (sm)   │   1.89   │    —     │    —     │    —     │
│ VICReg λ=0.5 (sm)   │   5.42   │    —     │    —     │    —     │
│ I-JEPA (small)      │  75.53   │    —     │    —     │    —     │
│ I-JEPA (full)       │   ???    │    —     │    —     │    —     │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘

Table 2: Downstream Multi-Agent Communication
┌─────────────────────┬──────────┬──────────┬──────────┐
│                     │ GSM8K    │ ARC-C    │ GPQA     │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Text baseline       │   ???    │   ???    │   ???    │
│ M0 (Original)       │   ???    │   ???    │   ???    │
│ M3 (I-JEPA)         │   ???    │   ???    │   ???    │
└─────────────────────┴──────────┴──────────┴──────────┘

Table 3: Ablation — I-JEPA Components
┌─────────────────────┬──────────┬──────────┐
│                     │ Eff Rank │ GSM8K    │
├─────────────────────┼──────────┼──────────┤
│ Full I-JEPA         │    —     │    —     │
│ - w/o predictor     │    —     │    —     │
│ - w/o target noise  │    —     │    —     │
│ - w/o stop-grad     │    —     │    —     │
│ - w/o AC loss       │    —     │    —     │
└─────────────────────┴──────────┴──────────┘

Table 4: Cross-Architecture Generalization
┌─────────────────────┬──────────┬──────────┐
│                     │ Eff Rank │ GSM8K    │
├─────────────────────┼──────────┼──────────┤
│ Qwen3-VL (Dense)    │    —     │    —     │
│ LFM2.5-VL (Mamba+)  │    —     │    —     │
└─────────────────────┴──────────┴──────────┘
```

### 2.2 加分实验（区分 Q1 和 Q2 的关键）

| 实验 | 价值 | 时间 |
|------|------|:---:|
| **SV decay 曲线**：对比 4 种方法的奇异值衰减 | 直观展示坍塌 | 已有 |
| **Token 可视化**：t-SNE/UMAP of latent tokens | Figure 1 吸引眼球 | 1h |
| **训练过程 Eff Rank 曲线**：step 0→750 的坍塌演化 | 展示坍塌是渐进过程 | 已有 step 150/300 数据 |
| **Multi-seed 统计**：3 seeds 的 Eff Rank 均值±方差 | Reviewer 必问 | ×3 时间 |
| **Cross-task 泛化**：codec 在未见 task 上的表现 | 证明不是过拟合 | 需额外数据 |

### 2.3 实验优先级排序

```
P0（论文必须有，缺了不能投）:
├── Full-data Original + I-JEPA SVD 完整对比 ← 现在就差这个
├── I-JEPA decoder 训练（证明修复后能通信）  ← 现在就差这个
├── Downstream M0 vs M3（3 tasks × 50 samples）← 核心 Table
└── 至少 1 个 ablation（AC loss 有无）

P1（Q1 vs Q2 区分点）:
├── LFM2.5 cross-architecture validation
├── Training dynamics (Eff Rank vs step curve)
├── 3 seeds statistics
└── SV decay 对比图

P2（锦上添花，有时间就做）:
├── t-SNE/UMAP visualization
├── Multi-seed 扩展到全部实验
├── λ grid search for AC loss weight
└── Data size ablation (90/300/3000)
```

---

## 三、时间规划（12周）

```
Week 1-2: 补全 P0 实验
  Day 1-3: I-JEPA full-data encoder+decoder 训练 (6.6h)
  Day 4:   SVD 全量对比分析 (1h)
  Day 5-7: Downstream M0 vs M3 (3 tasks, ~18h)

Week 3-4: P1 实验
  Day 8-10: LFM2.5 M0 + M3 (9.9h)
  Day 11-12: Multi-seed (×3 for key checkpoints)
  Day 13-14: Training dynamics + SV decay

Week 5-6: 写作大纲 + 初稿
  Day 15-17: Abstract + Intro + Related Work
  Day 18-21: Method
  Day 22-28: Experiments + Results

Week 7-8: 修改打磨
  Day 29-35: Internal revision
  Day 36-42: Figure polishing + table formatting

Week 9-10: 投预印本 + 收集反馈
  Day 43-49: arXiv submission
  Day 50-56: 同行反馈 + 修改

Week 11-12: 最终修改 + 投稿
  Day 57-70: Final polish + 投稿
```

---

## 四、论文结构

```
1. Introduction (1 page)
   - 多 agent 通信需要隐空间通道
   - Vision Wormhole 提出 UVC 作为解决方案
   - 我们发现 UVC 存在严重维度坍塌
   - 我们提出 I-JEPA 根治方法

2. Related Work
   - Multi-agent LLM communication
   - Latent space communication / codec
   - Dimensional collapse in representation learning
   - JEPA and self-supervised learning

3. Preliminaries
   3.1 Vision Wormhole UVC recap
   3.2 Dimensional collapse: definition & metrics

4. Collapse Diagnosis (C1)
   4.1 SVD-based collapse measurement
   4.2 Original UVC collapse evidence (Table 1)
   4.3 Training dynamics: collapse is progressive (Figure)
   4.4 VICReg doesn't fix it (Table 1)

5. Why Teacher-Student Distillation Collapses (C2)
   5.1 Degenerate shortcut analysis
   5.2 Gradient analysis of the collapse mechanism
   5.3 Connection to neural collapse literature

6. I-JEPA for Latent Communication (C3)
   6.1 From teacher-student to self-prediction
   6.2 Two-stage training: encoder → decoder
   6.3 Anti-collapse via prediction task diversity

7. Experiments
   7.1 Setup (models, data, metrics, baselines)
   7.2 Main Results (Table 2: downstream tasks)
   7.3 Ablation Studies (Table 3)
   7.4 Cross-Architecture Generalization (Table 4)
   7.5 Analysis: SV spectrum, training dynamics, visualization

8. Discussion
   8.1 Why prediction > distillation for this task
   8.2 Limitations: computation overhead, two-stage complexity
   8.3 Broader impact: what this means for latent communication

9. Conclusion
```

---

## 五、审稿人可能攻击的点及防御

| 攻击 | 风险 | 防御 |
|------|:---:|------|
| "只在 Qwen3 上验证" | 高 | LFM2.5 cross-architecture (P1) |
| "可能 seed 巧合" | 高 | 3 seeds 均值±方差 (P1) |
| "Eff Rank 高 ≠ 通信好" | 中 | Downstream M0 vs M3 直接对比 (P0) |
| "为什么不用其他防坍塌方法" | 中 | 已对比 VICReg (Table 1) |
| "两阶段训练不公平" | 中 | 控制总训练步数相同 |
| "3090 显存限制 bs=4" | 低 | 梯度累积等效。这不是审稿重点 |
| "没有 theoretical guarantee" | 低 | JEPA-SCORE 论文已有理论。我们引用即可 |
| "下游任务太少" | 中 | 3 tasks × 2 methods = 6 evaluations 够用 |

---

## 六、下一步立即执行

**本周必须完成（P0 核心）：**

1. SVD 分析已有 `codec_qwen3_ijepa_full_data.pt`（全量 I-JEPA encoder+decoder）
2. 如果 Eff Rank 仍然 > 50 → P0 实验基本够了
3. 跑 SVD intermediate（step 150/300 + final checkpoints）生成 training dynamics 曲线
4. 汇总所有 SVD 数据到一张对比表

**然后：**
5. Downstream evaluation（3 tasks × M0 vs M3）
6. 完成这些就可以开始写论文

---

## 七、投稿策略

| 会议 | 截稿 | 适合度 | 策略 |
|------|------|:---:|------|
| **NeurIPS** | ~5月 | ★★★★★ | 首选。ML systems + 理论 |
| **ICLR** | ~10月 | ★★★★ | 表示学习 + 多 agent |
| **ICML** | ~1月 | ★★★★ | 核心 ML |
| **ACL** | ~2月 | ★★★ | NLP 偏向，但多 agent 通信相关 |
| **CVPR** | ~11月 | ★★★ | 视觉偏重，但 VLM 相关 |

建议：先投 NeurIPS（最有影响力），reject 后转 ICLR/ICML。

---

## 八、关键 checkpoint 状态

| 文件 | 内容 | Eff Rank | 状态 |
|------|------|:---:|:---:|
| `codec_qwen3_original.pt` | 小样本 M0 | 2.32 | ✅ |
| `codec_qwen3_vicreg.pt` | 小样本 VICReg 0.1 | 1.89 | ✅ |
| `codec_qwen3_vicreg_05.pt` | 小样本 VICReg 0.5 | 5.42 | ✅ |
| `codec_qwen3_ijepa.pt` | 小样本 I-JEPA enc | 75.53 | ✅ |
| `codec_qwen3_ijepa_full.pt` | 小样本 I-JEPA enc+dec | — | ⚠️ 未SVD |
| `codec_qwen3_original_full.pt` | 全量 3000 M0 | 3.01 | ✅ (step 300) |
| `codec_qwen3_vicreg_05_full.pt` | 全量 3000 VICReg 0.5 | — | ⚠️ 未SVD |
| `codec_qwen3_ijepa_full_data_enc.pt` | 全量 3000 I-JEPA enc | — | ⚠️ 未SVD |
| `codec_qwen3_ijepa_full_data.pt` | 全量 3000 I-JEPA enc+dec | — | ⚠️ 未SVD |
| `seg_original_step150.pt` | 分段 step 150 | 3.61 | ✅ |
| `seg_original_step300.pt` | 分段 step 300 | 3.01 | ✅ |
