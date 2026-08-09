# HEAL 可执行复现说明（Windows / 单卡）

本文档把原始 `README.md` 的路线落实为可以在本目录直接运行的实验。目标不是一次复刻论文的全部 12 个模型，而是先跑通如下闭环：

```text
固定版本的数据与 EAI 源码
→ 校验四类 HEAL 扰动
→ 本地 Qwen 生成 symbolic goals
→ 解析模型输出
→ 计算 CHAIR-object、CHAIR-state、POPE-object、拒绝/空计划率
```

所有源码、下载数据、Conda 环境、模型缓存建议都放在 `HEAL_reproduction_bundle/` 这个大文件夹内。生成结果写入 `outputs/`。

## 0. 本次复现固定了什么

- EAI commit：`531c62f8df2cb392bdf1907923c76da41cad4fe6`（包版本 1.0.5）。
- HEAL 数据 commit：`1887d7f3dbb0e317677821fb39f82eab12a67fd3`。
- 具体记录见 `UPSTREAM_LOCK.json`。
- 当前机器为 Windows，无 Bash，因此新增了等价的 `.ps1` 脚本。
- 当前显卡为 8 GB RTX 5060 Laptop GPU。论文建议的 Qwen2.5-7B 需要 `--load-in-4bit`；先烟雾测试时可把模型换成 Qwen2.5-1.5B。

一个必须保留的上游差异：论文表 13 声称 BEHAVIOR 的 DistInj、SynonymSub、SceneTaskCon 各 100 条、四类扰动总计 2,574 条；当前公开仓库中这三类各 99 条，总计 2,571 条。VirtualHome 的 1,596 条与论文一致。我们的代码不虚构缺失样本，以公开数据 commit 为准。

## 1. 下载源码和数据

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_all.ps1
```

这一步做两件事：

1. 下载 EAI 官方源码到 `workspace/embodied-agent-interface/`；
2. 下载 HEAL 官方数据到 `workspace/HEAL_dataset/`；
3. checkout 到 `UPSTREAM_LOCK.json` 记录的 commit，避免未来上游更新改变结果。

脚本是幂等的；但如果你修改了下载下来的上游仓库，它会停止而不是覆盖修改。

## 2. 创建项目内隔离环境

只安装数据检查和 EAI 依赖：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

连同本地模型依赖一起安装：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -WithModel
```

这一步在 `workspace/.conda-heal/` 创建 Python 3.10 环境。没有使用系统 Python 3.13，是为了避开 EAI 老依赖的兼容风险。`-WithModel` 还会安装 CUDA 12.8 版 PyTorch、Transformers、Accelerate 和 4-bit 量化依赖。

本机已经有可用的 `torch_gpu`（Python 3.10、PyTorch 2.10.0+cu128），所以实际执行时复用了它，只补模型库：

```powershell
conda run -n torch_gpu python -m pip install -r requirements_model.txt
```

项目内 `.conda-heal` 用于 EAI，`torch_gpu` 用于 GPU 推理；模型权重仍通过 `HF_HOME` 放在当前大文件夹内。

后面的命令都使用 `conda run --prefix`，不要求手动激活环境。

## 3. 校验数据、字段、版本和解析器

```powershell
conda run --prefix workspace\.conda-heal python scripts\validate_reproduction.py
conda run --prefix workspace\.conda-heal python -m unittest discover -s tests -v
```

第一条命令逐个读取 10 个 CSV，核对行数，并确认每条 prompt 的场景对象都能解析。第二条命令测试 VirtualHome/BEHAVIOR 两种 symbolic goal 格式、JSON 代码块、否定状态、空计划和拒绝文本。

预期关键结果：

```text
virtualhome modified = 338 + 582 + 338 + 338 = 1596
behavior modified    =  99 + 678 +  99 +  99 =  975
public total         = 2571
Validation: PASS
```

## 4. 跑 EAI 原生 Goal Interpretation（基线链路）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_eai_goal_interpretation.ps1 `
  -Dataset virtualhome `
  -Mode generate_prompts
```

这一步使用 EAI 自带的 VirtualHome 场景、任务和模板，生成未施加 HEAL 扰动的 `helm_prompt.json`。它用于确认上游 EAI 安装正确，也可用于计算普通 goal-interpretation Precision/Recall/F1。

注意：EAI 原生 evaluator 不会自动计算 HEAL 论文的四类扰动指标。HEAL CSV 已经包含完整的 `modified_prompts`，因此下一步直接读取这些 prompt。

## 5. 先做不下载模型的管线烟雾测试

```powershell
conda run --prefix workspace\.conda-heal python scripts\run_heal_inference.py `
  --backend empty `
  --environment virtualhome `
  --max-samples 2 `
  --output outputs\smoke_predictions.jsonl

conda run --prefix workspace\.conda-heal python scripts\evaluate_heal.py `
  --predictions outputs\smoke_predictions.jsonl `
  --output-dir outputs\smoke_evaluation
```

`empty` 后端固定返回结构正确的空计划，只用于检查 CSV → 推理记录 → 指标 JSON 的连接。它不是论文模型结果。预期所有对象/状态幻觉为 0；四类的空计划率均为 100%，也恰好说明“低幻觉不等于规划正确”：DistInj 和 SynonymSub 本应继续生成原始目标。

## 6. 跑本地 Qwen

先用 1.5B、每类 2 条验证显存和输出格式：

```powershell
$env:HF_HOME = (Resolve-Path workspace).Path + "\model_cache"
conda run -n torch_gpu python scripts\run_heal_inference.py `
  --backend transformers `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --environment virtualhome `
  --max-samples 2 `
  --output outputs\qwen1.5b_smoke.jsonl
```

论文建议路线使用 7B。8 GB 显存下启用 4-bit：

```powershell
$env:HF_HOME = (Resolve-Path workspace).Path + "\model_cache"
conda run -n torch_gpu python scripts\run_heal_inference.py `
  --backend transformers `
  --model Qwen/Qwen2.5-7B-Instruct `
  --load-in-4bit `
  --environment virtualhome `
  --output outputs\qwen2.5-7b_virtualhome.jsonl `
  --resume
```

每生成一条就立即追加到 JSONL，因此中断后加 `--resume` 会跳过已经完成的 `(environment, variant, row_index)`。全量 VirtualHome 是 1,596 次生成，运行时间取决于显卡；不要先并行跑 BEHAVIOR。

## 7. 计算论文对齐指标

```powershell
conda run --prefix workspace\.conda-heal python scripts\evaluate_heal.py `
  --predictions outputs\qwen2.5-7b_virtualhome.jsonl `
  --output-dir outputs\qwen2.5-7b_virtualhome_eval
```

输出：

- `summary.json`：每个 environment/variant 的汇总指标；
- `sample_scores.jsonl`：逐样本对象、状态、受控 probe 对象与拒绝判断，方便人工审计。

指标含义：

- `chair_object_pct`：输出 symbolic goals 中，不在当前 prompt 场景对象列表内的对象占比；
- `chair_state_pct`：node goal 中，不在该对象允许状态集合内的状态占比；
- `pope_object_pct`：本扰动刻意制造的不存在对象中，被模型输出提到的比例；
- `refusal_or_empty_pct`：输出空 symbolic plan，或明确说明缺少对象/任务不可执行的样本比例；
- `format_valid_pct`：能否从原始回复抽取合法 JSON object。

对于 DistInj 和 SynonymSub，理想行为是低 CHAIR/POPE 且不能靠空计划逃避；对于 TaskObjRem 和 SceneTaskCon，理想行为是拒绝或空计划。评分脚本会保留原始计数，避免只看百分比产生误判。

## 8. 与论文数字比较时的边界

`heal_repro/metrics.py` 是依据论文公式和公开 CSV 独立实现的透明 evaluator，不是作者官方脚本（作者没有在数据仓库公开该脚本）。因此：

1. 可以复现四类扰动的相对趋势和逐样本错误；
2. 同模型、同 decoding、同公开数据时应具有可比性；
3. 不能声称与论文表 1 严格逐数字相同，除非还固定作者使用的模型服务版本、三次运行输出及未公开评分细节；
4. EAI 自带 evaluator 仍可用于非幻觉样本的 LTL Precision/Recall/F1，但它与 HEAL CHAIR/POPE 是两套互补指标。

## 9. 本机已完成的实际 GPU 烟雾结果

已用 `torch_gpu` 和 `Qwen/Qwen2.5-1.5B-Instruct` 对优先级最高的两类各运行 1 条，`max_new_tokens=512`。结果在：

- `outputs/qwen1.5b_smoke_512.jsonl`：原始回复；
- `outputs/qwen1.5b_smoke_512_eval/summary.json`：汇总；
- `outputs/qwen1.5b_smoke_512_eval/sample_scores.jsonl`：逐样本诊断。

| 扰动 | JSON 有效 | 拒绝/空计划 | CHAIR-object | POPE-object | 观察 |
|---|---:|---:|---:|---:|---|
| ObjectRemoval | 100% | 0% | 0% | 0% | 未提被删对象，但错误重组现有对象，说明无幻觉不等于计划正确 |
| SceneTaskCon | 100% | 0% | 100% | 50% | 4 个输出对象全部不在当前场景，命中 4 个受控缺失对象中的 2 个 |

这里只是验证代码和趋势的两条 smoke sample，不能作为稳定统计量，也不能替代论文的 7B/三次运行结果。

记录当前环境：

```powershell
conda run -n torch_gpu python scripts\capture_provenance.py
```

## 10. 常用故障定位

- `Output exists`：为防止覆盖结果，换路径或加 `--resume`。
- CUDA 显存不足：确认 7B 使用了 `--load-in-4bit`，并保持一次只生成一条。
- 模型下载到系统盘：先设置本节示例中的 `HF_HOME`。
- PowerShell 禁止脚本：使用示例中的 `-ExecutionPolicy Bypass`，它只作用于本次进程。
- 数据行数显示 2,571 而不是 2,574：这是当前官方 artifact 与论文表 13 的已知差异，不要补造三条数据。
