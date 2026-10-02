---
model_name: Canopy-Jev-27B
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: canopy-jev
tags:
  - decision-model
  - probabilities
  - lora
  - text-only
---

# Canopy-Jev-27B model card

**Artifact license:** Apache-2.0 for base-derived inference components. **Original project source:** MIT. See the complete licenses and attribution shipped alongside the files.

## Identity

- Base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Release source checkpoint: `506410609a6ba6eb422182fddda095b81fa4923011ceaa3717e85a74817237f4`.
- Continuation: 64 Dirichlet samples, actor weight 0.1, step 1200.
- Adaptation: 6,258,688 FP32 LoRA parameters on independent copies of text layers 60–63. The base text model and copied base weights stay frozen. The ten-letter numerical prior is supplied as a separate safe tensor file.
- Export: adapter and prior, 25,244,396 bytes total. No original base or raw training/evaluation data is included.

## Intended use

Text-based finite decisions with explicitly supplied candidates: routing, classification, evidence checks and ordinal rubrics. One request can contain a shared state and several isolated questions. The output is a numerical probability distribution, not a generated explanation.

Choice supports 1–10 candidates. Noul is a two-candidate adaptation; Score is an ordered-candidate adaptation with at most ten levels. This checkpoint does not provide separate Noul/Score heads, a 255-candidate head, vision inference, an action executor or autonomous closed-loop control. Decisions and probabilities can be wrong; application-specific calibration and policy selection are separate tasks.

## Training and selection

The project uses supervised calibration objectives and an actual sampled Dirichlet leave-one-out policy-gradient continuation. The continuation comparisons used a frozen eligible pool of 15,042 examples; the release endpoint consumed 9,600 ordered records through update 1,200. Earlier v4 and prior training are separate exposures. Only adapter parameters were updated; CE, Brier and a fixed-reference KL term accompany the actor objective. Original weights remain read-only.

Training sources include repurposed MMLU subject-test and SuperGPQA material plus auxiliary replay. These sources are training material here and must not be advertised as unseen evaluation benchmarks. The original question-text overlap guard protected MMLU-Pro and then-existing decision benchmarks. A [retrospective input screen](docs/TRAINING_INPUT_SCREEN.md) found zero normalized exact matches for the new public Test/OOD and JF100 against the prior/v4 actual training pools and current continuation exposure. This does not detect paraphrases, translations, semantic overlap or base-model pretraining contamination; pilot snapshots and games are outside this additional screen.

The release endpoint was chosen for reporting after inspecting all thirteen saved full-Pro endpoints at the user's request. It is not a blind development-selected result. Matched CE at the same step is only three questions behind; stable RL improvement is not established. The earlier independent development selection remains an independent historical record.

## Evaluations

| Evaluation | Result | Scope |
|---|---:|---|
| Full MMLU-Pro | 7,761/12,032, 64.503% | Historical tree-cache execution; five official category demonstrations; no target CoT or answer generation |
| JevBench public | 202/231, 87.446% | Fresh zero-shot tree inference; strict upstream replay and independent CPU audit |
| JevBench family mean | 87.788% | Eighteen equal-weight families |
| JF100 rotations | 216/300, 72.00% | One hundred items, three fixed rotations; all 300 valid |
| Expanded public Test hard | 33,457/42,738, 78.284% | 384 unsupported rows counted wrong; 387 soft targets separate |
| Expanded public OOD hard | 69,154/83,867, 82.457% | 864 unsupported rows counted wrong; 400 soft targets separate |

JF100 has 59/100 items correct under all three rotations and 65/100 semantic choices consistent across rotations. Expanded public Open-Jev Test/OOD completed and passed independent audits. All supported and unsupported rows are accounted for; public projection denominators differ from the original complete Open-Jev report. Publication does not add benchmark answers to training or calibrate this checkpoint on those benchmarks.

Generated reasoning controls scored 457/681 on Test and 450/750 on OOD; OOD temporal control is 32/108. These are subsets of the expanded evaluation. Complete weak and strong domains remain in [the control summary](results/openjev_reasoning_controls.json).

The default release API is `zero_shot_choice`. The historical Pro result used the explicitly separate five-demonstration protocol and historical cache numerical domain. A fresh arbitrary application prompt does not automatically inherit that score.

The exact historical Open-Jev pilot snapshots also completed: workflow raw 53/96 and gated 62/96, browser proposal 113/120 and drone complete 12/120. Five bounded local games and paired baselines were independently replayed. Trex and tiny-Wiki had 3/3 success; Snake collected no food, Tile had 0/3 completion and Doom mean reward was -29. These weak results are retained in [the full aggregate](results/openjev_pilot.json), with exact metric definitions in [evaluation protocols](docs/EVALUATION.md). No browser action or drone flight occurred.

## Runtime and validation

Validated model environment: Linux, Python 3.10, PyTorch 2.6.0/CUDA 12.4, Transformers 5.8.0, FLA 0.5.2 and Triton 3.2.0. A matching version string alone does not bypass source checks. At least 64 GiB free CUDA memory is required before loading. Full-model parity with the research loader, safe tensor conversion, typed payload validation and CPU package checks are documented in the validation artifacts. An isolated fresh CUDA install passed eleven CPU tests and three-case installed-package GPU checks; an initial probability-hash failure remains unexplained and is preserved in [clean_install_validation.json](clean_install_validation.json).

The default zero-shot latency study used one strongly shared H200, three warmups plus twenty measured requests for each of nine fixed state/query combinations. Full local request P50 ranges from 272 to 741 ms, P95 from 282 to 762 ms. It includes tokenization, fresh full64 work and JSON, excludes loading/network, and does not claim exclusive-device speed. Loading measured 37.58 seconds separately. All 180 timings and 27 warmups had identical outputs and passed independent CPU audit; see [latency_zero_shot.json](results/latency_zero_shot.json). The historical five-demo study is retained separately in [latency.json](results/latency.json); differing device loads prevent controlled cross-study speed claims.

## 中文

Canopy-Jev-27B 是冻结 Qwen3.8-27B 上的文本决策适配器。一次读取共享状态与隔离问题，直接返回概率，不生成解释。Choice 最多十项；Noul 与 Score 是候选适配，不是独立训练头。基座、视觉模块、原始数据、动作执行器和游戏环境不在发布包中。

完整 Pro 为 7761/12032、64.503%；JevBench 为 202/231、87.446%，任务族平均 87.788%。Pro 检查点是在查看全部十三个端点后指定报告的，不能称为盲测开发选择。与同预算 CE 的差距很小，没有稳定 RL 增益结论。原始 MMLU/SuperGPQA 中被用于训练的材料不能再次当作未见测试。

公开扩展Test/OOD已完整审计，含不支持项的硬准确率为78.284%/82.457%；有效子集分数另列。七类生成推理控制为67.11%/60.00%，时间控制OOD仅29.63%，全部弱项保留报告。

源码采用 MIT；基座派生成分保留 Apache-2.0。实际部署需要独立验证校准、兼容环境和业务决策规则。

## Citation

```bibtex
@software{qiu2026canopyjev,
  author  = {Qiu, Ruoxi},
  title   = {Canopy-Jev: Shared-State Numerical Decision Inference},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Camellia86/Canopy-Jev}
}
```
