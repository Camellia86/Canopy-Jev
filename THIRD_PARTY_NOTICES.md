# Third-party notices

The MIT license covers original Canopy-Jev code and documentation. Dependencies are installed separately; no third-party library source or binary runtime is vendored here.

| Component | Role | Upstream terms |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | Frozen pretrained text backbone and base-derived prior | Apache-2.0; complete copy in `licenses/QWEN_APACHE_2_0.txt` |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Tensor and CUDA execution | Upstream BSD-style terms and contributor notices |
| [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) | Qwen model and tokenizer implementation | Apache-2.0; Copyright the Hugging Face team |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE) | Gated DeltaNet kernels | MIT; Copyright Songlin Yang, Yu Zhang, Zhiyuan Li |
| [Triton](https://github.com/triton-lang/triton/blob/main/LICENSE) | Kernel runtime | Upstream MIT license |
| [Safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE) | Non-executable inference tensor format | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Numerical result handling | Upstream BSD-style terms |

The architecture is inspired by the shared-state and isolated-question design discussed in the supplied Jev architecture notes. The notes are a technical reference and are not redistributed. The project does not claim knowledge of TypeSafe's unpublished implementation or official RLCD recipe.

Benchmarks are evaluation inputs, not part of the inference package. Preparation tools download pinned public sources separately. JevBench, MMLU-Pro, Jev Frontier 100, and Open-Jev's public evaluation projection retain their own licenses and attribution. Open-Jev's projection has mixed record provenance, including CC0 generated examples, WANLI CC BY 4.0 examples, and documented source-specific restrictions. Benchmark names and published competitor scores are credited in the results documentation. No competitor model weights are bundled.

## 中文

原创代码使用 MIT；基座派生成分、独立安装的依赖和下载的数据均保留各自许可。仓库不包含第三方运行库副本、原始基座、竞品权重、架构笔记 PDF 或原始评测题集。评测数据的来源、版本和许可由准备工具记录，不能以本仓库的 MIT 覆盖。
