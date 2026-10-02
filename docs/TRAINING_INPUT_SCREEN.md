# Retrospective training-input overlap scope

The fixed release checkpoint (`506410609a6ba6eb422182fddda095b81fa4923011ceaa3717e85a74817237f4`) continues the original v4 checkpoint (`171197c2a7112f1dac183044a4e8472ba968d967c05528f9cda79f2a44720282`). Both use the v2 linear prior (`4620b120f1887a2240dc0cc748de9631b701e8db16a200450d3b41b4e25d6bee`). The v3 head is not an ancestor; the v3 candidate pool supplied a selected auxiliary replay subset to v4.

| Input population | Rows | Sources |
|---|---:|---|
| Prior v2 training | 10,000 | MMLU auxiliary 10,000 |
| v3 candidate pool, broader than actual ancestor training | 89,218 | MMLU auxiliary 89,218 |
| v4 cached training | 15,488 | MMLU subject 5,488; SuperGPQA 8,000; auxiliary replay 2,000 |
| v4 eligible training pool | 15,042 | MMLU subject 5,401; SuperGPQA 7,653; auxiliary replay 1,988 |
| Current continuation through update 1,200 | 9,600 | MMLU subject 3,418; SuperGPQA 4,926; auxiliary replay 1,256 |

Raw file and manifest hashes, the independently reproduced eligibility filter, the locked shuffled batch order, and checkpoint hashes are recorded in the accompanying JSON. The 15,042-row pool is the full eligible population; the current continuation checkpoint has consumed its first 9,600 ordered records. Earlier v4 training and the prior are separate exposures. Screening the full v4 pool is conservative; it does not claim that every row affected every ancestor checkpoint.

The screen covered all 43,125 public Test rows and 84,267 public OOD rows, including all 1,248 unsupported K=16 rows, plus the 100 original JF100 items. It compared each training question with the evaluation instructions, serialized state, state plus instructions, and string leaves of structured state. Normalization follows the existing preparation code: Unicode casefold and `\w+` tokens joined by spaces. A stronger check also requires the normalized, order-independent option multiset to match. Neither evaluation references nor model output files were read; target/answer fields were not used.

There were **zero question-component matches against the prior v2 training, v4 cached pool, v4 eligible pool, or the current continuation exposure**, for all three evaluation suites. The broader v3 candidate-pool screen also found zero matches on public Test/OOD. Four JF100 items had a one-character, one-word state leaf equal to the same short candidate-pool question; none matched the option multiset, none met the eight-word threshold, and that question was absent from the v4 replay/eligible pools. This is not evidence of a duplicated complete problem.

This is a retrospective exact-text screen, not proof of decontamination. It does not detect paraphrases, substring copies, translations, semantic overlap, or base-model pretraining contamination. Official in-context demonstrations, restricted Wiki rows, and archived workflow/browser/drone/game pilot inputs are outside the screen. The original v4 overlap guard did not include the newly added Open-Jev or JF100 suites; those suites must not be described as prospectively decontaminated.

Evidence JSON SHA-256: `5f61c51b93a61db78344eaab9eaa2d747ca221ad7ed3b5c96f894b2141b51401`.
