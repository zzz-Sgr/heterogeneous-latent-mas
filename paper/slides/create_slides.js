const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "组会汇报";
pres.title = "根治异质VLM隐空间通信的维度坍塌";

// Colors
const DARK_BLUE = "1A3A5C";
const WHITE = "FFFFFF";
const LIGHT_GRAY = "F0F2F5";
const ACCENT_GOLD = "E8B830";
const ACCENT_RED = "C0392B";
const ACCENT_GREEN = "27AE60";
const TEXT_DARK = "2C3E50";
const TEXT_MEDIUM = "5D6D7E";

// ============================================================
// Slide 1: Title
// ============================================================
let s1 = pres.addSlide();
s1.background = { color: DARK_BLUE };
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: ACCENT_GOLD } });
s1.addText("根治异质VLM隐空间通信的维度坍塌", {
  x: 0.8, y: 1.5, w: 8.4, h: 1.2,
  fontSize: 36, fontFace: "Arial", color: WHITE, bold: true, align: "center", margin: 0,
});
s1.addText("Vision Wormhole UVC 的诊断与 I-JEPA 修复", {
  x: 0.8, y: 2.7, w: 8.4, h: 0.7,
  fontSize: 20, fontFace: "Arial", color: "BDC3C7", align: "center", margin: 0,
});
s1.addShape(pres.shapes.LINE, { x: 3, y: 3.6, w: 4, h: 0, line: { color: ACCENT_GOLD, width: 1.5 } });
s1.addText("组会汇报 · 2026年6月", {
  x: 0.8, y: 3.9, w: 8.4, h: 0.5,
  fontSize: 14, fontFace: "Arial", color: "95A5A6", align: "center", margin: 0,
});

// ============================================================
// Slide 2: Background
// ============================================================
let s2 = pres.addSlide();
s2.background = { color: WHITE };
s2.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: DARK_BLUE } });
s2.addText("背景：Vision Wormhole 隐空间通信", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: DARK_BLUE, bold: true, margin: 0,
});
s2.addShape(pres.shapes.LINE, { x: 0.6, y: 1.0, w: 2, h: 0, line: { color: ACCENT_GOLD, width: 3 } });

// Left: problem description
s2.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.4, w: 4.2, h: 3.2, fill: { color: LIGHT_GRAY }, rectRadius: 0.1 });
s2.addText([
  { text: "异构VLM多Agent通信", options: { bold: true, fontSize: 18, color: DARK_BLUE, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "UVC (Universal Visual Codec) 将不同VLM的hidden states映射到共享连续隐空间", options: { fontSize: 14, color: TEXT_DARK, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "通过视觉接口注入接收方VLM", options: { fontSize: 14, color: TEXT_DARK, breakLine: true } },
], { x: 0.9, y: 1.6, w: 3.6, h: 2.8, valign: "top", margin: 0 });

// Right: training approach
s2.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.4, w: 4.2, h: 3.2, fill: { color: LIGHT_GRAY }, rectRadius: 0.1 });
s2.addText([
  { text: "训练方式", options: { bold: true, fontSize: 18, color: DARK_BLUE, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "Teacher-Student 蒸馏", options: { fontSize: 15, color: ACCENT_RED, bold: true, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "Teacher: Text pathway 输出", options: { fontSize: 14, color: TEXT_DARK, breakLine: true } },
  { text: "Student: Visual pathway (Codec)", options: { fontSize: 14, color: TEXT_DARK, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "Loss = MSE + KL + Stats", options: { fontSize: 13, color: TEXT_MEDIUM, breakLine: true } },
], { x: 5.5, y: 1.6, w: 3.6, h: 2.8, valign: "top", margin: 0 });

// Bottom question
s2.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.8, w: 8.8, h: 0.6, fill: { color: DARK_BLUE }, rectRadius: 0.05 });
s2.addText("核心问题：这个共享隐空间真的在传递丰富信息吗？", {
  x: 0.8, y: 4.8, w: 8.4, h: 0.6, fontSize: 17, fontFace: "Arial", color: WHITE, align: "center", valign: "middle", margin: 0,
});

// ============================================================
// Slide 3: Core Finding — Collapse
// ============================================================
let s3 = pres.addSlide();
s3.background = { color: WHITE };
s3.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: DARK_BLUE } });
s3.addText("核心发现：UVC存在严重维度坍塌", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: DARK_BLUE, bold: true, margin: 0,
});
s3.addShape(pres.shapes.LINE, { x: 0.6, y: 1.0, w: 2, h: 0, line: { color: ACCENT_GOLD, width: 3 } });

// Table header
const tableY = 1.3;
// Table using shapes
s3.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: tableY, w: 8.8, h: 0.5, fill: { color: DARK_BLUE } });
s3.addText([
  { text: "方法", options: { bold: true, color: WHITE, fontSize: 14 } },
], { x: 0.8, y: tableY, w: 3.2, h: 0.5, valign: "middle", margin: 0 });
s3.addText([
  { text: "小数据 Eff Rank", options: { bold: true, color: WHITE, fontSize: 14 } },
], { x: 4.0, y: tableY, w: 2.6, h: 0.5, align: "center", valign: "middle", margin: 0 });
s3.addText([
  { text: "全量数据 Eff Rank", options: { bold: true, color: WHITE, fontSize: 14 } },
], { x: 6.6, y: tableY, w: 2.6, h: 0.5, align: "center", valign: "middle", margin: 0 });

// Table rows
const rows = [
  ["Original (teacher-student)", "2.32", "4.05"],
  ["+ VICReg λ=0.5", "5.42", "2.94"],
  ["+ I-JEPA (自监督)", "75.53 ★", "14.29→ (进行中)"],
];
rows.forEach((r, i) => {
  let y = tableY + 0.5 + i * 0.5;
  let bg = i % 2 === 0 ? LIGHT_GRAY : WHITE;
  s3.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: y, w: 8.8, h: 0.5, fill: { color: bg } });
  s3.addText(r[0], { x: 0.8, y: y, w: 3.2, h: 0.5, fontSize: 14, color: TEXT_DARK, valign: "middle", margin: 0 });
  let c1 = i === 2 ? ACCENT_GREEN : (i === 0 ? ACCENT_RED : TEXT_DARK);
  s3.addText(r[1], { x: 4.0, y: y, w: 2.6, h: 0.5, fontSize: 14, color: c1, bold: i === 2 || i === 0, align: "center", valign: "middle", margin: 0 });
  let c2 = i === 2 ? DARK_BLUE : TEXT_DARK;
  s3.addText(r[2], { x: 6.6, y: y, w: 2.6, h: 0.5, fontSize: 14, color: c2, bold: i === 2, align: "center", valign: "middle", margin: 0 });
});

// Key findings
s3.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.6, w: 8.8, h: 1.7, fill: { color: LIGHT_GRAY }, rectRadius: 0.08 });
s3.addText([
  { text: "关键发现", options: { bold: true, fontSize: 16, color: DARK_BLUE, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "▸ 512维隐空间实际只使用 2-3 维（Eff Rank = 2.32）", options: { fontSize: 13, color: TEXT_DARK, breakLine: true } },
  { text: "▸ Token Cosine Similarity = 0.999 — 所有 token 几乎相同", options: { fontSize: 13, color: TEXT_DARK, breakLine: true } },
  { text: "▸ 训练越久越差：step 150 (3.61) → step 300 (3.01)", options: { fontSize: 13, color: TEXT_DARK, breakLine: true } },
  { text: "▸ VICReg 正则化只能缓解，不能根治（Eff Rank 5.42）", options: { fontSize: 13, color: TEXT_DARK } },
], { x: 0.9, y: 3.75, w: 8.2, h: 1.5, valign: "top", margin: 0 });

// ============================================================
// Slide 4: Why Collapse?
// ============================================================
let s4 = pres.addSlide();
s4.background = { color: WHITE };
s4.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: DARK_BLUE } });
s4.addText("为什么坍塌？Teacher-Student 蒸馏的退化捷径", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.7,
  fontSize: 26, fontFace: "Arial", color: DARK_BLUE, bold: true, margin: 0,
});
s4.addShape(pres.shapes.LINE, { x: 0.6, y: 1.0, w: 2, h: 0, line: { color: ACCENT_GOLD, width: 3 } });

// 3 boxes
const boxes4 = [
  {
    title: "退化捷径",
    body: "Student 学会对所有输入输出相同向量即可让 MSE 很低。Teacher 的 latent 本身多样性有限，student 记住\"平均表示\"就是局部最优。",
  },
  {
    title: "梯度集中化",
    body: "Loss landscape 退化 — 梯度只更新极少数方向。大量维度从未被有效训练，形成\"死维度\"（Dead Dimensions）。",
  },
  {
    title: "正则化不够",
    body: "VICReg（方差+协方差正则）只能缓解 5.42 → 不足以恢复。需要改变训练范式，而非加正则项。",
  },
];
boxes4.forEach((b, i) => {
  let x = 0.6 + i * 3.1;
  s4.addShape(pres.shapes.RECTANGLE, { x: x, y: 1.4, w: 2.8, h: 2.8, fill: { color: LIGHT_GRAY }, rectRadius: 0.08 });
  // Accent top line
  s4.addShape(pres.shapes.RECTANGLE, { x: x, y: 1.4, w: 2.8, h: 0.06, fill: { color: i === 0 ? ACCENT_RED : i === 1 ? "E67E22" : DARK_BLUE } });
  s4.addText(b.title, { x: x + 0.2, y: 1.65, w: 2.4, h: 0.4, fontSize: 16, color: DARK_BLUE, bold: true, margin: 0 });
  s4.addText(b.body, { x: x + 0.2, y: 2.1, w: 2.4, h: 1.8, fontSize: 12, color: TEXT_MEDIUM, valign: "top", margin: 0 });
});

// Bottom arrow
s4.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.6, w: 8.8, h: 0.7, fill: { color: DARK_BLUE }, rectRadius: 0.05 });
s4.addText("结论：Teacher-Student 蒸馏 = 结构性问题，不是调参能解决的 → 需要新训练范式", {
  x: 0.8, y: 4.6, w: 8.4, h: 0.7, fontSize: 15, color: WHITE, align: "center", valign: "middle", margin: 0,
});

// ============================================================
// Slide 5: I-JEPA Fix
// ============================================================
let s5 = pres.addSlide();
s5.background = { color: WHITE };
s5.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: DARK_BLUE } });
s5.addText("修复方案：I-JEPA 自监督替代蒸馏", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: DARK_BLUE, bold: true, margin: 0,
});
s5.addShape(pres.shapes.LINE, { x: 0.6, y: 1.0, w: 2, h: 0, line: { color: ACCENT_GOLD, width: 3 } });

// Two-stage diagram
// Stage 1
s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.4, w: 4.2, h: 3.0, fill: { color: LIGHT_GRAY }, rectRadius: 0.08 });
s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.4, w: 4.2, h: 0.06, fill: { color: ACCENT_GREEN } });
s5.addText("Stage 1: I-JEPA Encoder 预训练", {
  x: 0.9, y: 1.6, w: 3.6, h: 0.4, fontSize: 16, color: DARK_BLUE, bold: true, margin: 0,
});
s5.addText([
  { text: "不模仿 teacher，而是预测被 mask 的 token", options: { fontSize: 13, color: TEXT_DARK, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "anchor+noise_a → encoder → U_a → predictor → 预测 U_b", options: { fontSize: 11, color: TEXT_MEDIUM, breakLine: true } },
  { text: "anchor+noise_b → encoder → U_b (stop_grad)", options: { fontSize: 11, color: TEXT_MEDIUM, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "Loss = MSE(pred, target) + AC loss (方差+协方差)", options: { fontSize: 12, color: TEXT_DARK, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "▸ 预测任务天然要求 encoder 对不同输入产生不同输出", options: { fontSize: 12, color: ACCENT_GREEN, bold: true } },
], { x: 0.9, y: 2.05, w: 3.6, h: 2.1, valign: "top", margin: 0 });

// Stage 2
s5.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.4, w: 4.2, h: 3.0, fill: { color: LIGHT_GRAY }, rectRadius: 0.08 });
s5.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.4, w: 4.2, h: 0.06, fill: { color: DARK_BLUE } });
s5.addText("Stage 2: 冻结 Encoder，训练 Decoder", {
  x: 5.5, y: 1.6, w: 3.6, h: 0.4, fontSize: 16, color: DARK_BLUE, bold: true, margin: 0,
});
s5.addText([
  { text: "冻结已防坍塌的 encoder", options: { fontSize: 13, color: TEXT_DARK, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "随机初始化 decoder → teacher-student 蒸馏", options: { fontSize: 11, color: TEXT_MEDIUM, breakLine: true } },
  { text: "Decoder 学习 \"把 U 翻译成有效 VLM 注入\"", options: { fontSize: 11, color: TEXT_MEDIUM, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "Loss = MSE + KL + Stats (同原始)", options: { fontSize: 12, color: TEXT_DARK, breakLine: true } },
  { text: "", options: { fontSize: 6, breakLine: true } },
  { text: "▸ Encoder 保持高 Eff Rank，Decoder 只做翻译", options: { fontSize: 12, color: DARK_BLUE, bold: true } },
], { x: 5.5, y: 2.05, w: 3.6, h: 2.1, valign: "top", margin: 0 });

// Bottom results
s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.6, w: 8.8, h: 0.8, fill: { color: DARK_BLUE }, rectRadius: 0.05 });
s5.addText([
  { text: "小数据 Eff Rank 2.32 → 75.53（32× 提升）", options: { bold: true, color: ACCENT_GOLD, breakLine: true } },
  { text: "全量重训中：step 600 Eff Rank = 14.29，趋势持续上升", options: { color: WHITE } },
], { x: 0.8, y: 4.6, w: 8.4, h: 0.8, fontSize: 14, align: "center", valign: "middle", margin: 0 });

// ============================================================
// Slide 6: Next Steps
// ============================================================
let s6 = pres.addSlide();
s6.background = { color: WHITE };
s6.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: DARK_BLUE } });
s6.addText("下一步计划", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: DARK_BLUE, bold: true, margin: 0,
});
s6.addShape(pres.shapes.LINE, { x: 0.6, y: 1.0, w: 2, h: 0, line: { color: ACCENT_GOLD, width: 3 } });

const plans = [
  { title: "I-JEPA 全量训练完成", desc: "重训至 2 epoch，每 200 步 SVD 验证 Eff Rank 趋势", time: "进行中", color: ACCENT_GREEN },
  { title: "下游任务评估", desc: "GSM8K / ARC-C / GPQA 三任务 M0 vs M3 通信质量对比", time: "2周", color: DARK_BLUE },
  { title: "梯度因果分析", desc: "梯度SVD + Hessian谱 + CKA 揭示坍塌的优化动力学机制", time: "4-5周", color: "E67E22" },
  { title: "跨架构验证", desc: "LFM2.5-VL Mamba混合架构上验证坍塌和修复的通用性", time: "6-7周", color: "8E44AD" },
  { title: "投稿 ICLR 2027", desc: "12周写作+修改，目标10月截稿", time: "10-12周", color: ACCENT_RED },
];
plans.forEach((p, i) => {
  let y = 1.3 + i * 0.85;
  s6.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: y, w: 8.8, h: 0.75, fill: { color: LIGHT_GRAY }, rectRadius: 0.06 });
  s6.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: y, w: 0.08, h: 0.75, fill: { color: p.color } });
  s6.addText(p.title, { x: 1.0, y: y + 0.05, w: 5.0, h: 0.3, fontSize: 15, bold: true, color: DARK_BLUE, margin: 0 });
  s6.addText(p.desc, { x: 1.0, y: y + 0.38, w: 6.0, h: 0.3, fontSize: 12, color: TEXT_MEDIUM, margin: 0 });
  s6.addText(p.time, { x: 7.8, y: y, w: 1.4, h: 0.75, fontSize: 13, color: p.color, bold: true, align: "center", valign: "middle", margin: 0 });
});

// ============================================================
// Slide 7: Summary
// ============================================================
let s7 = pres.addSlide();
s7.background = { color: DARK_BLUE };
s7.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: ACCENT_GOLD } });
s7.addText("总结", {
  x: 0.6, y: 0.4, w: 8.8, h: 0.7,
  fontSize: 30, fontFace: "Arial", color: WHITE, bold: true, margin: 0,
});

const summaries = [
  { icon: "1", text: "首次发现并量化 Vision Wormhole UVC 的维度坍塌", sub: "512 维隐空间仅用 2-3 维，Eff Rank = 2.32" },
  { icon: "2", text: "Teacher-Student 蒸馏是根因", sub: "退化捷径 + 梯度集中化。VICReg 正则化不够" },
  { icon: "3", text: "I-JEPA 自监督替代方案有效", sub: "Eff Rank 2.32 → 75.53（32× 提升），全量重训确认中" },
  { icon: "→", text: "下一步", sub: "下游通信验证 + 梯度因果分析 → ICLR 2027" },
];
summaries.forEach((s, i) => {
  let y = 1.4 + i * 1.0;
  s7.addShape(pres.shapes.OVAL, { x: 1.0, y: y + 0.1, w: 0.55, h: 0.55, fill: { color: ACCENT_GOLD } });
  s7.addText(s.icon, { x: 1.0, y: y + 0.1, w: 0.55, h: 0.55, fontSize: 16, color: DARK_BLUE, bold: true, align: "center", valign: "middle", margin: 0 });
  s7.addText(s.text, { x: 1.8, y: y, w: 7.2, h: 0.35, fontSize: 18, color: WHITE, bold: true, margin: 0 });
  s7.addText(s.sub, { x: 1.8, y: y + 0.4, w: 7.2, h: 0.3, fontSize: 13, color: "BDC3C7", margin: 0 });
});

s7.addShape(pres.shapes.LINE, { x: 1, y: 5.1, w: 8, h: 0, line: { color: ACCENT_GOLD, width: 1 } });
s7.addText("组会讨论：因果机制分析方向 + 投稿策略", {
  x: 0.6, y: 5.2, w: 8.8, h: 0.4, fontSize: 14, color: "95A5A6", align: "center", valign: "middle", margin: 0,
});

// ============================================================
// Write file
// ============================================================
pres.writeFile({ fileName: "/Users/rui/Documents/Workspace/Project_01_组会读Paper/slides/组会汇报_维度坍塌.pptx" })
  .then(() => console.log("PPT saved!"))
  .catch(err => console.error(err));
