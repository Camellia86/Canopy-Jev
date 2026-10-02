# Canopy-Jev（冠层 Jev）

**JevBench 87.45%：超过Open-Jev-27B-v1.1 2.16个百分点。Hard 75.68%：提升3.60个百分点。**

[English](README.md#english) · [Hugging Face权重](https://huggingface.co/Camellia86/Canopy-Jev-27B) · [MIT](LICENSE)

![Canopy-Jev与Open-Jev的JevBench对比](docs/assets/benchmark_comparison.png)

Canopy-Jev基于冻结Qwen3.8-27B，一份应用状态供多个隔离问题分支共享，直接输出候选概率，不生成答案token。名字来自“一条主干支撑多个决策分支”的树形计算。

### 亮点

- JevBench **87.45%**；Hard **75.68%**；任务族平均 **87.79%**。
- 完整MMLU-Pro **64.50%**，达到60+目标。
- 最后四层使用 **626万参数LoRA**，适配器和概率先验共 **25.24MB**。
- 默认零样本API在共享H200上，完整请求P50为 **272–741ms**。

### 更多决策成绩

JF100 **72.00%**，workflow门控后 **64.58%**，browser完整方案 **94.17%**，均超过Open-Jev-27B的**百步历史pilot**。[完整评测](docs/EVALUATION.md) · [图表数据](docs/BENCHMARK_FIGURES.md)。

### 安装与调用

Linux、NVIDIA CUDA，载入前至少64GiB空闲显存。已验证Python3.10、PyTorch2.6.0/CUDA12.4、Transformers5.8.0、FLA0.5.2、Triton3.2.0。

```bash
git clone https://github.com/Camellia86/Canopy-Jev.git
cd Canopy-Jev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cuda]'
hf download Camellia86/Canopy-Jev-27B --include 'artifacts/*' --local-dir .
export CUBLAS_WORKSPACE_CONFIG=:4096:8
canopy-jev doctor
canopy-jev predict --artifacts artifacts \
  --base-model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --request examples/request.json --output response.json
```

基座单独从固定官方版本下载，也可指定兼容的本地目录并加 `--local-files-only`。接口支持Choice 1–10候选、Noul二候选适配、Score 1–10有序候选适配；问题共享前缀且互相隔离，完整打包上限16384tokens。

### 时延

默认零样本接口，共享H200；单位ms，P50/P95：

| State tokens | 1个问题 | 4个问题 | 8个问题 |
|---:|---:|---:|---:|
|31|275 / 285|283 / 312|302 / 318|
|511|272 / 282|279 / 296|340 / 348|
|2048|575 / 595|613 / 633|741 / 762|

计时包含分词、完整模型、概率解码和JSON；模型加载另计37.58秒。[时延汇总](results/latency_zero_shot.json)。

### 架构与训练

前60层冻结编码共享树，最后四层复制底座并加LoRA。普通注意力通过因果树掩码和分支位置编号隔离；Gated DeltaNet分别复制每个分支的前缀卷积和递归状态。数值先验读取分支端点，返回固定策略的候选概率。

训练使用Dirichlet采样策略梯度，配合CE、Brier和参考KL。方法、真实离线W&B曲线及模型选择历史见 [训练记录](docs/TRAINING.md) 和 [模型说明](MODEL_CARD.md)。

### 许可证

原创代码与文档采用完整 **MIT**；基座派生成分保留 **Apache-2.0**。完整条款见 [LICENSE](LICENSE)、[WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 引用

Author / 作者：**Ruoxi Qiu**。

```bibtex
@software{qiu2026canopyjev,
  author  = {Qiu, Ruoxi},
  title   = {Canopy-Jev: Shared-State Numerical Decision Inference},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Camellia86/Canopy-Jev}
}
```

Machine-readable citation / 机器可读引用：[CITATION.cff](CITATION.cff)。
