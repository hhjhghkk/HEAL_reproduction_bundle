# HEAL 复现包：下载、安装与使用说明

> Windows 用户和实际复现实验请优先阅读
> [`REPRODUCTION_CN.md`](REPRODUCTION_CN.md)。该文档包含已验证命令、每一步的作用、
> HEAL 专用评分方式，以及论文与公开数据数量不一致的说明。
>
> 如果是第一次阅读代码并准备亲自跑模型，请阅读
> [`HEAL_CODE_AND_RUN_GUIDE_CN.md`](HEAL_CODE_AND_RUN_GUIDE_CN.md)。其中包含代码结构、
> 数据流、核心接口、样例走读和逐命令实操说明。

## 1. 这个包是什么

HEAL（EMNLP 2025 Findings）研究 LLM 驱动具身智能体在 **scene–task inconsistency** 下的幻觉。论文公开的是 **HEAL probing set 数据集**；其基础执行/评测代码来自 **Embodied Agent Interface (EAI)**。

因此最稳妥的复现组合是：

```text
HEAL probing set
      +
EAI goal_interpretation pipeline
      +
LLM / VLM inference
      ↓
object/state hallucination evaluation
```

本压缩包提供一键脚本，把完整 EAI 源码和 HEAL 数据拉到 `workspace/`。

---

## 2. 目录

```text
HEAL_reproduction_bundle/
├── README.md
├── UPSTREAM_SOURCES.md
├── requirements_reproduction.txt
├── paper/
│   └── HEAL_EMNLP2025.pdf
├── upstream/
│   └── EAI_LICENSE.txt
├── scripts/
│   ├── fetch_all.sh
│   ├── fetch_eai_sdist.py
│   ├── download_heal_dataset.py
│   ├── setup_env.sh
│   ├── run_eai_goal_interpretation.sh
│   └── inspect_heal.py
├── examples/
│   ├── load_heal_hf.py
│   └── local_qwen_inference_template.py
└── workspace/
    ├── embodied-agent-interface/   # fetch_all 后生成
    └── HEAL_dataset/               # fetch_all 后生成
```

---

## 3. 最短使用路径

### Step 1：下载完整上游源码与 HEAL 数据

```bash
cd HEAL_reproduction_bundle
bash scripts/fetch_all.sh
```

脚本会：

1. `git clone` EAI 官方仓库；
2. 如果 git clone 失败，下载并校验 `eai-eval==1.0.5` PyPI source distribution；
3. 从 Hugging Face 下载 HEAL 数据集。

### Step 2：创建环境

官方 EAI 使用 Python 3.8 路线，建议保持一致：

```bash
bash scripts/setup_env.sh
conda activate heal-repro
```

### Step 3：确认 HEAL 数据

```bash
python scripts/inspect_heal.py
```

### Step 4：先跑 EAI Goal Interpretation

生成 VirtualHome prompts：

```bash
eai-eval \
  --dataset virtualhome \
  --eval-type goal_interpretation \
  --mode generate_prompts
```

或者：

```bash
bash scripts/run_eai_goal_interpretation.sh virtualhome generate_prompts
```

BEHAVIOR：

```bash
bash scripts/run_eai_goal_interpretation.sh behavior generate_prompts
```

### Step 5：接入模型

HEAL 本质上是在 task description / structured scene information 上施加受控扰动，再检查模型是否生成不被场景支持的目标。

优先只跑一个本地模型，例如：

```text
Qwen2.5-7B-Instruct
```

模板见：

```bash
python examples/local_qwen_inference_template.py
```

实际实验时，应把 EAI 生成的 prompt 输入模型，并保存原始回复，再交给 evaluator。

### Step 6：评测模型输出

```bash
eai-eval \
  --dataset virtualhome \
  --eval-type goal_interpretation \
  --mode evaluate_results \
  --llm-response-path <YOUR_RESPONSE_PATH>
```

---

## 4. HEAL 四类关键扰动

建议按以下顺序复现：

| 扰动 | 含义 | 建议优先级 |
|---|---|---:|
| Distractor Injection | 任务描述中加入场景不存在的干扰对象 | 3 |
| Task Relevant Object Removal | 从场景信息删除任务关键对象 | 2 |
| Synonymous Object Substitution | 场景对象替换为同义名称 | 4 |
| Scene Task Contradiction | 用不相关场景与原任务构造整体冲突 | 1 |

对我们当前研究，优先级最高的是 **SceneTaskCon + ObjectRemoval**：它们最接近“是否知道当前任务不可执行、是否应该拒绝规划/主动验证”的问题。

---

## 5. 推荐复现路线

### Phase A：只复现论文核心趋势

```text
HEAL data
→ VirtualHome structured prompt
→ Qwen2.5-7B
→ hallucinated object/state extraction
→ CHAIR / POPE / rejection behavior
```

此阶段不需要先把 VirtualHome/BEHAVIOR 全部物理执行跑通。

### Phase B：接入 EAI evaluator

```text
HEAL perturbation
→ EAI goal interpretation
→ LTL / symbolic goal
→ hallucination + planning quality
```

### Phase C：连接我们自己的 Checker

```text
HEAL offline inconsistency
→ model response
→ HEAL/EAI metrics
→ our Checker
→ reject / verify / replan
```

### Phase D：迁移到在线具身闭环

HEAL 的一个重要局限是论文主要评价 symbolic goal，而不是完整物理执行。因此真正与我们研究衔接的工作是：

```text
HEAL scene-task inconsistency
→ 在线 robot episode
→ memory / plan
→ Checker
→ System-1 execution
→ consequential hallucination
```

---

## 6. 不建议一开始做什么

不要一开始：

- 复现论文所有 12 个模型；
- 同时安装并跑满 VirtualHome 与 BEHAVIOR；
- 直接做 VLM image+text 全量实验；
- 把 self-correction 当作主要方法；
- 先追求论文所有表格逐数字一致。

更有效的是先用 **一个本地模型 + VirtualHome + 4 类 perturbation** 跑通完整链路。

---

## 7. 与我们研究的接口

HEAL 最适合作为外部 baseline，而不是我们的最终环境。

```text
HEAL
scene-task inconsistency
       ↓
证明 LLM 会在证据不足时继续规划
       ↓
我们的 P1
memory-conditioned / physical consequential hallucination
       ↓
我们的 P2
trace-grounded SFT / DPO / GRPO
       ↓
Checker + online execution
```

HEAL 的价值主要在于：**标准化诱发错误**；我们的工作应继续回答：错误是否进入记忆、改变计划并真正造成物理后果。

---

## 8. 常见问题

### Q1：为什么包里最初 `workspace/` 是空的？

HEAL 论文没有独立 GitHub code repo；完整代码依赖 EAI。运行 `bash scripts/fetch_all.sh` 后，官方 EAI 源码和 HEAL 数据会自动进入 `workspace/`。

### Q2：必须装 BEHAVIOR/iGibson 吗？

不是。先做 HEAL 离线/goal-interpretation 复现时可以暂不安装；只有需要完整 BEHAVIOR simulation 时再安装。

### Q3：应该先用哪个模型？

建议 Qwen2.5-7B-Instruct，方便与你现有本地 System 2 实验衔接。

### Q4：需要严格复刻作者所有模型吗？

不需要。先验证 4 类扰动的基本趋势，再扩展 2–3 个代表模型即可。

---

## 9. 官方来源

详见 `UPSTREAM_SOURCES.md`。上游 EAI 代码遵循其原始 MIT License；本包未改变其许可证要求。
