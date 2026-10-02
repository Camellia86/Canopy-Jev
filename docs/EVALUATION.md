# Evaluation protocols / 评测协议

Every result uses the fixed step-1200 adapter and inherited probability policy. No benchmark is used here for training, temperature fitting, retries or checkpoint selection. The reported MMLU-Pro endpoint was chosen after inspecting thirteen saved endpoints; that historical selection limitation remains part of its interpretation.

## Completed evaluations

| Evaluation | Population | Result | Execution |
|---|---:|---:|---|
| MMLU-Pro | 12,032 questions | 7,761 correct; 64.503% | Historical five-demonstration tree cache |
| Native Qwen3.8-27B, direct letter readout | Same 12,032 questions | 7,539 correct; 62.658% | Same historical Pro input protocol; no generated reasoning |
| JevBench | 231 tasks, 18 families | 202 correct; 87.446%; family mean 87.788% | Fresh full-model, zero-shot application-state tree |
| JF100 | 100 items × three rotations | 216/300; 72.000% | Fresh full-model, fixed rotations |

Historical cache execution and fresh request execution are separate numerical domains. The Pro score is not an evaluation of arbitrary new prompts or the native base's generative thinking mode.

### JevBench comparison

Canopy-Jev's 202/231 is compared with published Open-Jev-27B-v1.1 197/231 and NeoHorse-Jev-4B 75.32%. Competitors were not rerun. Open-Jev's 86.469% family mean is recomputed from its [official family counts](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1/blob/28cf73067d5b337860bbef3c85b8b82ba8730956/verification/jevbench.json); NeoHorse's 75.73% family mean is from its [fixed model card](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B/blob/434cb21d3a994a953d3ae5788405fcb2c4970554/README.md).

The three task files and four upstream scoring files match the pinned Open-Jev corpus byte for byte. Canopy-Jev has Original/Easy/Hard counts 70/72, 48/48 and 84/111; Noul/Choice/Score counts 66/74, 121/139 and 15/18. The original scorer and its lexicographic-key tie rule were retained. All 231 probability outputs were valid, with no repair or renormalization. Brier is 0.190145, NLL is 0.372925 and Score expected-level MAE is 0.227584.

Different model prompts, training data, candidate order and model sizes prevent attributing the published-score difference solely to architecture.

### JF100

The pinned [Jev Frontier 100](https://github.com/softpudding/jev-frontier-100/tree/9abacec47394f3b393f81fbe3cdd524f028bc088) has 100 items, three fixed option rotations and 50 counterfactual pairs. All 300 probabilities were valid. 59/100 items were correct under every rotation; 65/100 retained a consistent semantic choice. Counterfactual pair correctness was 89/150 pair-rotations; 23/50 pairs had all six decisions correct.

Rotations are correlated observations. The descriptive source-stratified pair bootstrap interval is 65.67–77.67% for aggregate accuracy, not an interval based on 300 independent questions. The historical Open-Jev 27B 100-step pilot reports 196/300; it is a different checkpoint from final v1.1. See the [upstream protocol](https://github.com/Zefan-Cai/Open-Jev/blob/b718c4beaaf13af7d37504a23e99ef599e216e67/docs/frontier-eval.md) and [audited aggregate](../results/jf100.json).

## Expanded public Test and OOD

The fixed [Open-Jev-v1.1 public projection](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1/tree/10ad6888333fa97f8c948192797bad3de3040802) contains 43,125 Test rows and 84,267 OOD rows. It omits 395 Wiki rows from the original internal report. Our projection cannot be assigned that report's denominator or compared as an identical population.

| Split | Planned rows | Candidate-supported rows | Explicit K=16 failures | Hard / soft targets |
|---|---:|---:|---:|---:|
| Test | 43,125 | 42,741 | 384 | 42,738 / 387 |
| OOD | 84,267 | 83,403 | 864 | 83,867 / 400 |

The model supports at most ten candidates; K=16 painting rows remain planned failures. Candidates are never truncated. Token preflight found no supported row over the 4,096-token joint-prompt cap. This cap is distinct from an upstream per-candidate limit. The shared-state batch budget is 6,000 tokens and at most sixteen questions; JF100 uses a 16,384-token single-input cap.

Inference reads only seven whitelisted input fields and hashes reference bytes without parsing targets. Independent CPU scoring begins after every shard has completed and source, artifact, prompt, runtime, probability and coverage checks pass. All-planned hard accuracy counts unsupported or overlength failures as incorrect. Valid-only accuracy discloses its smaller denominator. Soft targets remain distributions; NLL, distribution Brier and expected one-hot Brier cover valid probabilities only. Group macro weights each `(source, group_id)` equally. The seven reasoning-control families are subsets of this same evaluation, not additional independent benchmarks.

The expanded runs completed and passed independent audits. Test hard accuracy is 33,457/42,738=78.284%; OOD hard accuracy is 69,154/83,867=82.457%. Valid-only hard accuracy is 78.994%/83.315%, with probability coverage 99.110%/98.975%. No partial accuracy was used for claims or model changes. The additional upstream historical snapshot and game protocols completed and were independently audited. Their metrics are documented below; snapshot correctness must not be described as real browser execution, drone flight or game completion.

## Audited expanded results

All **127,392 public rows** completed and passed independent source, prompt, probability and coverage audits. The unsupported sixteen-candidate cases remain failures in all-planned hard accuracy; soft targets are reported as distributions.

| Public projection | Full hard accuracy | Valid-only hard accuracy | Probability coverage | Equal-group hard mean |
|---|---:|---:|---:|---:|
| Test | **33,457/42,738 — 78.28%** | 78.99% | 42,741/43,125 — 99.11% | 68.88% |
| OOD | **69,154/83,867 — 82.46%** | 83.32% | 83,403/84,267 — 98.97% | 71.22% |

Test has 384 unsupported rows and 387 soft targets; OOD has 864 unsupported rows and 400 soft targets. No supported row exceeded the joint-prompt limit. Valid-probability NLL / distribution Brier are **0.556895 / 0.294701** for Test and **0.472864 / 0.249299** for OOD. See [Test](../results/openjev_public_test.json), [OOD](../results/openjev_public_ood.json) and [full protocols](EVALUATION.md).

Open-Jev-27B-v1.1's [original internal report](https://github.com/Zefan-Cai/Open-Jev/blob/b718c4beaaf13af7d37504a23e99ef599e216e67/docs/internal-full-evaluation.md) reports 41,357/42,789 (96.65%) on expanded Test and 80,934/83,924 (96.44%) on expanded OOD. Our observed scores are lower. Its complete population includes Wiki rows omitted from the public projection, and candidate scoring and training protocols differ; these are not identical-denominator comparisons.

The seven generated reasoning-control families are subsets of these same runs: **457/681 (67.11%) Test** and **450/750 (60.00%) OOD**. OOD temporal control is only **32/108 (29.63%)**. All seven families are retained in [the complete control summary](../results/openjev_reasoning_controls.json); these programmatic controls are not external general-reasoning benchmarks.

## Historical pilot snapshots and local games

The exact historical pilot snapshot cases and upstream typed scorer were retained. All **336 snapshots / 2,249 decision heads** were valid; no real browser action or drone flight was executed.

| Snapshot benchmark | Canopy-Jev 27B | Published Open-Jev 27B 100-step pilot |
|---|---:|---:|
| Workflow raw exact / 96 | **53 (55.21%)** | 41 (42.71%) |
| Workflow after fixed policy gate / 96 | **62 (64.58%)** | 43 (44.79%) |
| Browser complete proposal / 120 | **113 (94.17%)** | 104 (86.67%) |
| Drone complete three-head decision / 120 | 12 (10.00%) | 21 (17.50%) |

The comparison is with historical pilots, not final Open-Jev-27B-v1.1. Fixed upstream postprocessing, including typed distribution normalization and workflow gates, is retained; native model probabilities are separately checked. See the [protocol](https://github.com/Zefan-Cai/Open-Jev/blob/b718c4beaaf13af7d37504a23e99ef599e216e67/docs/model-evaluation.md) and [audited aggregate](../results/openjev_pilot.json).

Five real local game engines were also run with seeds 10001–10003, a forty-step budget per episode and paired random and upstream teacher/oracle baselines. All **15 model episodes** and thirty baseline trajectories completed as planned and passed independent replay. These are simplified local fixtures, including genuine ViZDoom basic assets, rather than general game competence.

| Local game metric | Canopy-Jev | Random baseline | Upstream teacher/oracle |
|---|---:|---:|---:|
| Snake: mean food collected | 0.00 | 0.00 | 7.67 |
| Trex: all twelve obstacles cleared | 3/3 | 0/3 | 3/3 |
| Tile platformer: completed | 0/3 | 0/3 | 3/3 |
| Tiny Wiki graph: target reached | 3/3 | 0/3 | 3/3 |
| ViZDoom basic: mean engine reward | -29.00 | 46.00 | 80.33 |

The teacher policies have different information and computation: Snake uses a visible-state heuristic, Trex a physics oracle, Tile a local BFS solver, and tiny-Wiki a shortest-path oracle with access to the full graph. They are reference policies, not competing learned models.

One model Doom episode reached the step cap and is explicitly truncated; no success threshold is invented. Browser routing and the local Trex/Wiki fixtures are strengths in this evaluation; drone maneuvering, Snake and Tile completion and Doom remain weak. No teacher fallback was used.

## Reproduce public projection inference

Use the complete checkout and its supplied artifacts, and prepare data separately. Original datasets retain their licenses and are not bundled or relicensed as MIT. Commands and shard rules are in [PUBLIC_EVALUATION.txt](../tools/PUBLIC_EVALUATION.txt). Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting numerical libraries. Do not merge receipts made under a different runtime or prompt contract.

For JF100, pass `--max-input-tokens 16384`; for the expanded public Test/OOD, use 4096. On several GPUs, use distinct output directories and shard indexes with the same shard count. Score only once every shard has a completion receipt. The lightweight public scorer verifies coverage and byte identities; its generic first-candidate tie rule does not replace JevBench's upstream scorer or the separate research audit.

## Other advertised upstream scopes

The archived provider comparison reused 82 original rows, including 76 hard targets, for historical Open-Jev 2B/9B checkpoints. It contains four Wiki rows (one hard and three soft) absent from the public projection. The exact 76-hard slice therefore cannot be completely evaluated from this public release; Canopy-Jev is not assigned a score under that denominator. Replacing the missing row or reporting a smaller slice as 76 would change the benchmark.

Upstream final-model JF100, TREC, the other provider panels and the fixed V3 panels are still pending or prepared in the pinned upstream [benchmark status](https://github.com/Zefan-Cai/Open-Jev/blob/b718c4beaaf13af7d37504a23e99ef599e216e67/docs/benchmarks.md). TREC's reported Jev/Luna/Astra runs are not Open-Jev measurements. These scopes are not presented as completed comparable Open-Jev model results. Old/expanded Test and OOD populations overlap, so they should not be counted as independent additional studies.

## 中文

所有本轮补测固定使用 1200 步检查点、原温度和概率策略；不训练、不重试、不据评测答案调参。Pro 的 64.503% 来自已查看十三个端点后的指定报告，不能当作盲测开发选择结果；原生 Qwen3.8 的 62.658% 是相同历史输入下的直接字母读出，尚未测原生生成式思考。

历史 provider 对比的 76 个硬样本中有一个 Wiki 样本不在公开投影，不能冒用原分母给本模型打分。最终模型 JF100、TREC、其他 provider/V3 面板在固定上游版本仍是 pending/prepared；不包装成 Open-Jev 已完成的对比成绩。

JevBench 使用全部 231 个公开任务、原评分与独立审计，202/231 正确，任务族均分 87.788%。竞品分数来自固定公开版本，各模型输入适配和训练不同。JF100 为 100 题三次轮换，216/300 正确，59 题三次全对，65 题语义选择一致；轮换不能当作 300 个独立问题。历史 Open-Jev 百步 pilot 与最终 v1.1 分开标注。

公开扩展 Test/OOD 共 127,392 行，含 1,248 个十六候选不支持项，全部保留在失败分母；公开投影比原内部报告少 395 个 Wiki 行。先完整推理、审核所有分片和身份，再读取答案评分。硬标签、软分布、有效概率覆盖和等权组指标分别报告，不能把有效子集分数冒称完整成绩。上游 workflow、browser、drone 快照与五种本地游戏已经完成并独立审计；快照分类不代表真实网页动作、飞行或游戏通关。
