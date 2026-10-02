# Inference artifact licensing

The repository's original source code and documentation use the complete MIT license in `LICENSE`, as requested by the project owner.

The adapter and numerical-prior artifacts are derived from **Qwen/Qwen3.8-27B**, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Base-derived components retain **Apache-2.0**, whose complete upstream text is provided in `licenses/QWEN_APACHE_2_0.txt`; attribution and the modification notice are in `NOTICE`. The MIT code license does not replace those upstream terms.

Only the inference adapter and small prior are exported. The original base checkpoint, vision weights, vocabulary output matrix and raw training/evaluation records are not bundled. Obtain the exact base from its [official repository](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).

The source configuration, tensor index, model card and license were compared byte for byte with that official revision. `base_provenance.json` records their identities. Downloaded datasets retain their original licenses and are not relicensed by this repository.

## 中文

本仓库原创代码与文档采用完整 **MIT** 许可证。模型适配器与小型概率先验来源于上述固定版本的 Qwen3.8-27B；其中基座派生成分保留 **Apache-2.0** 条款、原始署名和修改说明。MIT 不替换基座、数据或依赖的许可证。仓库不打包原始基座、视觉权重、全词表输出矩阵或原始训练/评测数据。
