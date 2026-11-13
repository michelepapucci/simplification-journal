# Improving, and interpreting, readability control in the GFFF models

*Working report for the journal extension of "Evaluating Italian Readability-Controlled Text Simplification" (CLiC-it 2026). November 2026.*

The models discussed here are the Qwen3 0.6B / 1.7B / 4B models in `GFFF/` of the old `text-simplification-experiments` repository. They are conditioned with one token per strategy and level, such as `<|readability_syntax_40|>`. The CLiC-it 2025 and LREC 2026 papers are used as related work whose code can be reused.

**Dataset.** The journal version uses the Wikipedia part of the IMPaCTS release on Hugging Face ([`mpapucci/impacts`](https://huggingface.co/datasets/mpapucci/impacts), configs `wikipedia` and `wikipedia_profiling`, pinned at revision `489c78e` in [`../scripts/impacts.py`](../scripts/impacts.py)). It holds 764,061 pairs from 106,680 original sentences. Section 0 sets out how it relates to the data the CLiC-it 2026 models were trained on. The public-administration part is left out for now.

**Objective.** The objective throughout is *simplification only*: the model should never be asked to make a text harder. That choice drives most of the design below.

Every number comes from [`analysis/controllability_evidence.py`](analysis/controllability_evidence.py) (output in [`analysis/controllability_evidence.json`](analysis/controllability_evidence.json)), run as:

```sh
python reports/analysis/controllability_evidence.py --gfff ../GFFF --lrec ../LREC/data/dataset_profiled.zip
```

The release is downloaded at the pinned revision (`--hf-dir` reads a local snapshot instead). `--lrec` adds the check of how the release was derived from the LREC 2026 corpus. Nothing in the old repository was modified.

---

## Summary: what to change, in order

1. **Fix the pipeline first.** Use the data rebuilt from the release ([`../scripts/build_dataset_v0.py`](../scripts/build_dataset_v0.py), dataset v0 in [`../data/v0/`](../data/v0/)): correct lexical tokens, no "no overlap" filter, no `n_sentences`. Hold out test **originals**, not just test pairs. Fix the embedding initialisation of the new tokens. Evaluate the tokens the paper says it evaluates. All four are cheap, and every later comparison depends on them.
2. **Train on the whole release, not on the CLiC-it selection.** The CLiC-it selection (original READ-IT global > 70, no READ-IT score higher after simplification, two noise filters) keeps 55,877 of the release's 764,061 Wikipedia pairs. With test originals held out, the release offers 740,563 pairs from 104,105 originals, 17× the 42,907 left in the selection. The extra pairs bring sources at every readability level, the trade-off pairs, and examples at the rare levels.
3. **Make the evaluation match the task.** A simplifier should only be asked for levels at or below the source. When the source is already at the requested level, leaving it (almost) unchanged is the correct answer. Most of the lexical "failure" in the paper is requests above the source.
4. **Always give three target tokens, one per READ-IT dimension (base, lexical, syntax), plus the source's levels.** Train with an `any` option so that one model can target one dimension or all three, either letting the others vary freely or holding them fixed (Section 2.3).
5. **Keep the trade-off pairs and label them.** 45% of the release's pairs are harder than the original in at least one dimension, mostly lexical, although never globally. The CLiC-it selection threw them all away. Labelled with `any` for the dimension that went up, they are valid simplification examples and the main source of independent variation between dimensions.
6. **Add pairs inside each original's cluster, in the simplifying direction.** Up to 2.6 million more ordered pairs between simplifications of the same original, including 290,000 near no-ops. They supply "same input, different target" contrasts. With the whole release, data volume is no longer the constraint, so subsample them by design.
7. **Re-run the noise filters on the whole release.** The langdetect and "double filter" lists only cover the 141,984 pairs the CLiC-it selection kept. The release still contains 3,154 pairs langdetect flagged, and 622,077 pairs were never checked. Decide whether the double filter, which removes half of the hard originals, is a quality filter worth re-implementing (Section 0).
8. **Leave true complexification (simplification → original) out of the main runs.** It is not needed for a simplifier and teaches the model to add content (Section 2.4).
9. **Make the control space learnable and close the loop.** Use quantile bins, ordinal token embeddings, and smoothed labels for the noisy syntax score. Then use classifier-free guidance and best-of-N reranking (no training), hindsight relabelling, and preference optimisation.
10. **Interpretability.** The learned token embeddings already show an ordinal "level" axis. Section 5 lays out how to find where it is read and whether the three dimensions act separately.

---

## 0. The dataset: IMPaCTS Wikipedia on Hugging Face

The release is a strict subset of the LREC 2026 corpus, with identical texts and scores. Its cleaning rule reproduces membership exactly (100% of rows): **drop every pair whose READ-IT global score went up, and every simplification identical to its original.** The CLiC-it 2026 data is a further selection from it:

| Step | Pairs | Originals |
|---|---|---|
| LREC 2026 corpus, Wikipedia | 1,058,960 | 108,371 |
| − global score higher after simplification (275,849) − exact copies (19,050): **the release** | **764,061** | **106,680** |
| original READ-IT global > 70 | 237,201 | 26,930 |
| no READ-IT score (base, lexical, syntax, global) higher: the CLiC-it selection | 141,984 | 25,739 |
| − langdetect list − double-filter list | 58,632 | |
| − test pairs: dataset v0 (`data/v0/training_set_v0.zip`) | 55,877 | 11,657 |
| − every pair of a test original (grouped split, 2.1) | 42,907 | 9,145 |

The GFFF source file (`dataset_profiled_filtered_with_4_control_tokens.zip`) is exactly the CLiC-it selection plus 4,146 pairs that the release drops because the simplification is identical to the original. Profiling-UD features in `wikipedia_profiling` match the GFFF file value for value, and still include the constant `n_sentences` columns, which the build drops.

The release differs from the CLiC-it training data in four ways:

1. **It has sources at every readability level.** 75% of its originals (79,750) have READ-IT global ≤ 70. The quartiles of source global are 29 / 49 / 70, against 75 / 81 / 87 for the hard subset. For syntax they are 28 / 48 / 69 against 73 / 79 / 86.
2. **It keeps the trade-off pairs.** 45.2% of its pairs are harder than the original in base (15.1%), lexical (37.3%) or syntax (4.2%), while never harder globally. The CLiC-it selection removed them: 40.1% of the hard subset.
3. **It has no identity pairs.** The GFFF source file contained 4,146 pairs whose simplification is the original, unchanged, and 1,598 of them (3.5%) reached the CLiC-it 2026 training file. Without them dataset v0 has 55,877 training pairs per strategy instead of 57,720. The three test sets are unchanged.
4. **It has not been through the GFFF noise filters.** 3,154 pairs flagged by langdetect and 80,198 removed by the double filter are in the release. 622,077 pairs lie outside the CLiC-it selection and were never checked by either filter.

**What the double filter seems to be.** No code in the old repository produces `list_dropped_idx_double_filter.txt`. It acts on original sentences: it removes every simplification of 50.9% of the selected originals and part of them for only 2.4%. The removed originals are longer (median 250 characters, against 199). 92.3% of its decisions match a simple proxy: "the original contains quotes or parentheses, or is longer than 400 characters". That is consistent with dropping originals the sentence splitter breaks into several sentences, the same texts for which Profiling-UD averages its features over sub-sentences. This needs confirming. If it is a quality rule, re-implement it as a rule and apply it to the whole release; on the hard subset it would remove about half the originals.

---

## 1. Where the current models lose control

| Finding | Evidence | Consequence |
|---|---|---|
| **The models never saw an output harder than its source.** | The CLiC-it selection keeps only pairs with no READ-IT score above the source. Of the release's 237,201 pairs for originals with READ-IT global > 70, that removed 95,217 (40.1%), mostly because lexical went up (34.1% of pairs). The release had already removed every pair whose global score went up. | Any request at or above the source is out of distribution, and the model cannot learn trade-offs between dimensions. |
| **Lexical requests are often above the source.** | Test sources have lexical scores of 39–47 (quartiles). 45% of the paper's lexical requests target a bin *above* the source, rising to 85–99% at CT ≥ 50. Base and syntax requests are 97–98% feasible. | For lexical (4B), RMSE is 6.7 when the source already lies inside the requested bin, 19.2 for ordinary simplification requests and 21.9 when the bin is above the source. For those requests above the source, the three models copy the input 25–31% of the time and make it harder in only 11–14% of cases. The paper's lexical RMSE curve is largely a feasibility curve. |
| **Little within-input contrast.** | Each training original appears with 2.3 distinct base levels, 2.9 syntax and 1.5 lexical (dataset v0). 52% of the originals have a single lexical level. In the whole release: 2.8, 3.3 and 2.1, and 20% have a single lexical level. | The model rarely sees the same input with different targets, which is the signal that teaches it what the token changes. |
| **The lexical scale is compressed.** | 98.8% of lexical targets in the CLiC-it selection fall in levels 20–40. `lexical_60` has 31 examples and `lexical_70` 5. In the whole release it is still 95.4%, so this is a property of READ-IT lexical on this corpus. The release does have 3,463 examples at lexical ≥ 60, against 39. | 10-point bins waste most lexical tokens, and rare tokens are barely trained. |
| **Strategies are three measurements of one output, not three operations.** | Every pair appears three times, once under each strategy's token, with the same target text. The other two scores are never stated. | `lexical_30` means "lexical ≈ 30, plus whatever length and syntax usually come with it". That is why the CLiC-it 2026 lexical outputs were also shorter and structurally simpler. |
| **The syntax label is noisy.** | Among 20,993 near-identical (character similarity ≥ 0.9, not identical) simplifications of the same original in the training set, READ-IT syntax differs by ≥ 20 points in 6.8% of pairs (median 3.2), against 0.8% for base and 0.0% for lexical. Over the whole release (350,441 pairs): 5.3%, 0.9% and 0.1%. | Part of the weak syntax control (ρ 0.38–0.41 and RMSE 22–28 on 98%-feasible requests) is label noise. |
| **Train/test leakage inflates the larger model.** | 93–98% of test originals also appear in training with other simplifications. 4B reproduces a training simplification of the same original verbatim in 8–16% of outputs (near-duplicates 19–33%). For 0.6B/1.7B it is 2–7%. | Comparisons across sizes are confounded until the split is by original. |
| **Identity pairs in training.** | 1,598 CLiC-it training pairs (3.5%) are the original unchanged, labelled with the source's own levels. | They teach copying under whatever tokens the source happens to have. The release drops them. With source tokens (2.2), the same signal comes back in a controlled form: near-no-op pairs whose target levels equal the source's (2.1). |
| **The token embeddings start unevenly.** | `init_control_tokens_embeddings` (`GFFF/controlled_text_simplification_ft.py`, ~line 203) computes `old_n = rows − num_added`. After `resize_token_embeddings(..., pad_to_multiple_of=8)` (~line 182) the matrix has 5 padding rows *after* the new tokens. So the mean initialisation lands on the last 25 control tokens plus padding, and `base_0`–`base_40` keep the resize default. | Minor, but it biases comparisons between tokens. Initialise by token id instead (Section 2.5). |
| **Evaluation details.** | The paper's RMSE tables (16–18), Figure 4 and Spearman values (Table 5) were computed on tokens **0–60** but labelled CT 10–70. On tokens 0–60, the Spearman values of all three models reproduce Table 5 exactly. Generation samples at temperature 0.7 (`evaluate_model_against_original.py`). Pooled Spearman understates per-input ordering (base 0.70 pooled vs 0.74–0.75 within-original; lexical 0.28–0.33 vs 0.37–0.46). | Fix the labels, decode greedily (or average samples), and report within-input metrics. |

The CLiC-it 2026 models were also trained with the lexical-token bug: lexical tokens carried base levels. The embeddings show it, since `base_L` and `lexical_L` moved in similar directions during training (cosine 0.51–0.60, against 0.17–0.31 for `base_L` vs `syntax_L`).

---

## 2. Design

### 2.1 Training data

1. **Hold out test originals entirely.** Reuse the grouped split from the LREC code (`LREC/tuning/controlled_text_simplification_ft.py`, `load_datasets_from_path`, which splits by `original_sentence_idx`). In the CLiC-it selection, holding out the 2,575 test originals removes 12,970 training pairs (23%), leaving 42,907 pairs from 9,145 originals. In the whole release it leaves 740,563 pairs from 104,105 originals.

   The current test sets only contain originals with READ-IT global > 70. If training uses every source level, add test originals across the source range, drawn by original from the release, and request a grid of targets *at or below the source*. Each test original's other simplifications in the release can serve as multi-level references.
2. **Train on the whole release.** The CLiC-it selection uses 19% of the release's pairs before its noise filters and 7% after. Relaxing it adds:
   - **sources at every level.** 75% of the release's originals have READ-IT global ≤ 70. A model serving several reader levels must see medium sources, and they supply the cases where the target is already met.
   - **the trade-off pairs** (item 3).
   - **data at the rare levels.** Lexical ≥ 60 has 3,463 examples instead of 39. Base and syntax have all ten levels with at least 1,000 examples, instead of seven and nine.

   Without the trade-off pairs, the release still has 402,897 held-out pairs with no dimension harder. Every run from E3 on (Section 4) needs the noise filters re-run first (item 5).
3. **Keep the trade-off pairs, and label them.** With the labelling rule of 2.3, a pair whose lexical score went up is a valid example for base and syntax, with lexical set to `any`. These pairs are also the main source of independent variation between dimensions (2.3). For single-token models, a pair can serve strategy *s* only if *s* did not go up: that leaves 627,145 / 461,785 / 708,960 pairs for base / lexical / syntax. That lexical goes up in 37% of the pairs deserves a look of its own. One plausible cause is that shorter sentences have higher lexical density and type-token ratio, which READ-IT lexical counts as harder.
4. **Use the cluster: pairs between simplifications of the same original.** Each original has several simplifications at different levels (7.2 on average in the release). Ordered pairs *sᵢ → sⱼ* give many contrasts for the same input. Keep those that would pass the release's own rule (global score not higher, different text):

   | Pairs, test originals held out | CLiC-it selection | Release, global > 70 | Release, all sources |
   |---|---|---|---|
   | original → simplification | 42,907 | 213,703 | 740,563 |
   | *sᵢ → sⱼ*, global not higher | 106,569 | 865,588 | 2,614,990 |
   | … of which simpler in base / lexical / syntax | 72,482 / 62,779 / 96,393 | 616,346 / 499,135 / 792,167 | 1,774,167 / 1,494,732 / 2,350,199 |
   | … of which all three scores in the same 10-point bin (near no-op) | 13,082 | 77,629 | 289,827 |

   The near-no-op pairs teach the model when to stop: minimal edits when the target is already met. On the whole release the cluster pairs outnumber what any run can use, so subsample them: cap pairs per original, balance over target cells, and favour near no-ops and pairs that change one dimension.
5. **Re-run the noise filters on the whole release.** langdetect is cheap to re-run: it flagged 2.2% of the CLiC-it selection. The double filter is the open question of Section 0.

### 2.2 Condition on the source as well as the target

Three parametrisations carry the same information once the source score is known, and READ-IT on the input is cheap at inference:

| Control | Example | Pros | Cons |
|---|---|---|---|
| Target only (current) | `<tgt_syn_30>` | Absolute: matches a reader profile. | The model must infer the source level from the text. Requests at or above the source are invisible. |
| Delta | `<delta_syn_-50>` | Expresses the edit size directly, like ACCESS ratios (Martin et al., 2020). | Loses the absolute anchor. The same delta means different things at different sources. |
| **Source + target** (recommended) | `<src_syn_80><tgt_syn_30>` | Subsumes the delta and keeps the absolute target. Makes "target ≈ source → minimal edit" learnable. | More tokens; needs READ-IT at inference. |

The literature points the same way:
- Agrawal & Carpuat (2023) show that controls set per instance, from the source and the target grade, beat corpus-level controls.
- Farajidizaji et al. (2024) find that readability after rewriting stays correlated with the source's readability.
- Participants in the TSAR 2025 shared task report performance tied to the gap between source and target level (OUNLP, 2025).

Source tokens matter more on the release than on the CLiC-it selection, since its sources span the whole scale. Test it as an ablation (E1 vs E0, Section 4).

### 2.3 Three target tokens, one per dimension

**Format.** Every example carries the source's three levels and three target tokens, in a fixed order:

```
<|src_base_80|><|src_lex_40|><|src_syn_80|><|tgt_base_30|><|tgt_lex_any|><|tgt_syn_20|>
{source sentence}
{target sentence}
```

Each pair is used once, not three times. That is 3 dimensions × (10 levels + `any`) target tokens, plus 30 source tokens (or the quantile bins of 2.5). On the whole release that is 740,563 examples, 5.5× the 135,627 (45,209 × 3) of a CLiC-it run.

**Why it improves specificity.** With one token, the model sees only one coordinate of the output and fills in the other two from co-occurrence. With all three, every change in the output is described. Each token can take credit only for what the other two don't explain, so the lexical token no longer has to carry the length and syntax changes that usually accompany lexical simplification.

**How far the data allows it.** The three scores must vary independently in the data:

| | CLiC-it selection (141,984) | Release, global > 70 (237,201) | Release, all sources (764,061) |
|---|---|---|---|
| Correlation of changes, base–syntax | 0.54 | 0.56 | 0.53 |
| Correlation of changes, base–lexical / lexical–syntax | 0.30 / 0.29 | 0.26 / 0.16 | 0.31 / 0.08 |
| Syntax drops ≥ 10 while base and lexical stay within ±5 | 4.8% | 8.2% | 10.0% |
| Lexical drops ≥ 10 while base stays within ±5 | 0.3% (489) | 0.3% (792) | 0.5% (3,440) |
| Populated (base, lexical, syntax) 10-point cells (≥ 20 pairs), of 1,000 | 252 (99.0% of pairs) | 319 (99.3%) | 387 (99.8%) |

- **Base and syntax** are separable, more so with the trade-off pairs and with all sources: one pair in ten of the release changes syntax alone.
- **Lexical** almost never drops without the sentence also getting shorter. LLaMAntino rarely swaps vocabulary at constant length, so a "lexical only" request is mostly outside the data. The release has 3,440 such pairs: enough to evaluate the request, thin for training it. Three tokens make the gap visible rather than fixing it. Fixing it needs targeted data, for example prompting the generator for word substitution only.
- **The scores are nested by design.** READ-IT lexical includes the base features, and syntax includes both, so the three can never be fully independent.
- **Combinations** outside the ~250–390 populated cells are unsupported. The inference code should check requests against the cells observed for sources like the input.

**Labelling rule (simplification only).** For each training pair:
1. Source tokens are the source's bins.
2. For each dimension, the target token is the target's bin if it is at or below the source's bin; otherwise it is `any`. No example ever asks the model to raise a dimension; a trade-off pair simply says nothing about the dimension that went up.
3. Independently replace each remaining target token with `any` with probability ≈ 0.3 (attribute dropout, in the spirit of classifier-free guidance; Ho & Salimans, 2022). Occasionally replace all three, which keeps a generic "simplify" mode.

**Inference: targeting one dimension or all of them.**

| Mode | Tokens (want lexical 30) | Meaning | Works when |
|---|---|---|---|
| **Free** | `<tgt_base_any><tgt_lex_30><tgt_syn_any>` | Reach lexical 30; the others move as they usually do. Reproduces today's single-token behaviour. | Always (within the data) |
| **Keep** | `<tgt_base_80><tgt_lex_30><tgt_syn_85>`, i.e. the others set to the source's own levels | Change vocabulary, hold length and syntax. The specific request. | Syntax-only: yes (10% of release pairs). Lexical-only: rarely (0.5%) |
| **Guided** | Free or keep, plus classifier-free guidance on the one dimension | At each step `logits = any + w·(target − any)`, with *target* = logits with `lex_30` and *any* = logits with `lex_any`, everything else identical. `w > 1` amplifies that token's specific effect (Sanchez et al., 2024). | Costs two forward passes per token; fluency drops at large `w` |
| **Full** | All three set | A reader profile defined on every dimension. | Inside the populated cells |

**Measuring specificity.** In keep mode, request one dimension and measure how much the other two scores, and their Profiling-UD features, move anyway (off-target movement). Report it for every strategy, compared with the free mode and with the single-token models.

### 2.4 Complexification: not in the main runs

For a simplifier, true complexification (simplification → original) is not needed:
- **The lexical failure is an evaluation problem.** For requests above the source, the right output is the input unchanged, or the request should not be made.
- **It would not help base or syntax.** Their requests are already 97–98% feasible, and their control is still weak.
- **The useful contrast comes without it.** Cluster pairs in the simplifying direction (2.1) and the release's medium sources supply it.
- **It teaches content addition.** Going from a simplification back to its original often means restoring content the simplification dropped. That is the hallucination the CLiC-it 2026 manual evaluation already found in 8–18% of outputs.

What a simplifier *does* need is in the main design: knowing when to stop (source tokens + near-no-op pairs) and seeing dimensions trade off (trade-off pairs labelled with `any`).

Complexification is justified only if the journal model should change readability in both directions, as in Farajidizaji et al. (2024). It could also be a late ablation, testing whether harder targets sharpen the level scale. Its precedents are MUSS, which trains on mined paraphrases in both directions with ACCESS-style relative controls (Martin et al., 2022), and back-translation in MT (Sennrich et al., 2016). If it is tried, keep reverse pairs only when content is preserved in both directions (bidirectional NLI or BERTScore threshold, length-ratio bound), and track named entities and numbers absent from the source.

### 2.5 A learnable control space

- **Bins:** replace 10-point bins with per-dimension quantile bins (equal frequency), or finer bins where data is dense. Lexical has three levels with at least 1,000 examples in the CLiC-it selection and six in the whole release, with 95% of targets in 20–40 either way. Quantile bins are needed whatever the data.
- **Ordinal embeddings:** the trained embeddings already order levels along their first principal component (Section 5.1), so make that structure explicit. Parametrise `e(dim, level) = a_dim + level·d_dim + r_(dim,level)` with a small free residual, or initialise level tokens by interpolating between two anchors. Rare and unseen levels then share statistical strength. Compare against plain-text numbers (`lexical 30`), which reuse the model's pretrained number representations; Li & Shardlow (2024) compare control-token formats.
- **Initialisation fix:** take ids from `tokenizer.convert_tokens_to_ids(control_tokens)` instead of `rows − num_added`.
- **Label smoothing for noisy scores:** at training time, draw the control bin from a distribution centred on the measured score (for example, a neighbouring bin with probability proportional to proximity). This is most useful for syntax, whose label noise is several times that of base.
- **Rebalancing:** temperature-sample bins or cap the dominant ones. The release already fills the extreme base and syntax bins (Section 2.1).

### 2.6 Closing the loop on the model's own outputs

1. **Decoding only (no training).**
   - **Classifier-free guidance** on the dimension of interest (2.3).
   - **Best-of-N reranking:** sample N outputs, score with READ-IT, and keep the closest to the target among those above a meaning-preservation threshold. This is a cheap upper bound and a strong baseline. The TSAR 2025 findings report that reliable control often needed multi-step or iterative pipelines (Alva-Manchego et al., 2025).
   - **Guided decoding:** Kew & Ebling (2022) reach target levels with FUDGE-style guided decoding.
2. **Hindsight relabelling:** generate for many (input, request) pairs, score each output, relabel it with the tokens of the levels it *actually* reached (any dimension that went up becomes `any`), filter for meaning and fluency, and retrain. This is Hindsight Instruction Relabeling (Zhang et al., 2023) applied to readability, and it fills rare levels with on-policy data.
3. **Preference optimisation:** build DPO pairs (Rafailov et al., 2023). "Chosen" is close to the target with meaning preserved; "rejected" is far from the target, a copy when change was requested, or a hallucination. The release's clusters already provide candidates: for a given request, a sibling simplification that meets it versus one that does not. Ribeiro et al. (2023) control summary readability with a readability-distance RL reward and lookahead decoding, and their code is public. Yanamoto et al. (2022) use RL for grade-level simplification. READ-IT needs parsing, so offline DPO is more practical than online RL.

---

## 3. Evaluation protocol for the journal

- **Splits and decoding:** grouped split by original; greedy decoding (or the mean over k samples) for control metrics.
- **Test originals:** keep the current test sets for comparison with CLiC-it 2026, and add test originals across the release's source range (2.1).
- **Request grid:** for each test original and dimension, request only bins at or below the source's bin. When the source already lies in the requested bin, an (almost) unchanged output is correct and should be scored as such.
- **Per-level metrics:** RMSE, signed bias and **within-original** Spearman, plus a dose-response slope (achieved vs requested score per input).
- **Specificity:** off-target movement in keep mode (2.3), for each dimension.
- **Joint success:** |error| < 10 *and* meaning preserved (BERTScore or SBERT threshold), plus copy rate on requests that need change, a hallucination proxy, and memorisation rate against training targets of the same original.
- **An external check:** READ-IT is both the label and the metric, so validate on a human-rated sample as in the LREC 2026 study, or with a second readability model.

---

## 4. Experiment plan

Run ablations on Qwen3-1.7B, then scale the best two configurations to 0.6B and 4B. All runs use the held-out split and the initialisation fix. From E3 on they use the release after re-running langdetect (and the double filter, if kept).

| ID | Change | Training pairs | Compare with | Question |
|---|---|---|---|---|
| E0 | Fixed baseline: one target token per strategy (current format), CLiC-it selection | 42,907 × 3 strategies | CLiC-it 2026 | Clean reference |
| E1 | + source tokens (E1b: delta tokens) | as E0 | E0 | Does knowing the source help? |
| E2 | Three target tokens (+ source tokens if E1 helps), `any` dropout, each pair once | 42,907 | E1 | Specificity; free vs keep modes |
| E3 | E2 on the whole release, pairs with no dimension harder | 402,897 | E2 | Do medium sources and rare-level data help? |
| E4 | E3 + trade-off pairs (a dimension that went up is labelled `any`) | 740,563 | E3 | Separating the dimensions |
| E5 | E4 + subsampled cluster pairs, including near no-ops | + up to 2.6M | E4 | Within-input contrast; knowing when to stop |
| E6 | Quantile bins / ordinal embeddings / syntax label smoothing | best of E4–E5 | that run | Rare levels; syntax |
| E7 | Decoding: CFG on one dimension; best-of-N reranking | — | best model | How far decoding alone goes |
| E8 | Hindsight relabelling (1–2 rounds) | + self-generated | best model | Rare levels |
| E9 | DPO toward the target | preference pairs | E8 | Precision |
| A1 | *Optional:* filtered complexification pairs (simplification → original) | + ≤ 740k | E4/E5 | Only if bidirectional modification becomes a goal |

**Priorities.**
- **Minimum for the journal:** E0–E4, the format change and the move to the release. E1 is cheap.
- **E4:** needed if the paper keeps claiming distinct base / lexical / syntax *strategies*.
- **E5:** if within-input contrast is still weak after E4.
- **E7:** cheap and should run on every final model.
- **Compute:** E4 has 5.5× the examples of a CLiC-it run, so fix the number of training steps across E2–E5, or subsample to match, before attributing gains to the data.

**Dataset v1** ([`../data/v1/README.md`](../data/v1/README.md)) is the first implementation of the three-token format: 78,462 pairs, trade-off pairs included. It differs from E4 in four ways:
- it keeps complex originals only (READ-IT global > 70, like the test sets) and applies the double-filter proxy of Section 0;
- a dimension gets a target only when it drops by at least one control token and by more than READ-IT's noise (the 90th percentile of score differences between near-identical simplifications: 5.7 base, 3.8 lexical, 17.6 syntax);
- there are no source tokens, although the source levels are stored;
- `any` dropout is left to the training script.

---

## 5. How the model uses the control token

### 5.1 What the trained embeddings already show

For the three CLiC-it 2026 checkpoints, comparing each control token's trained embedding with its initialisation (`modules_to_save` vs `original_module` in the adapter):

- **An ordinal axis per strategy.** The first principal component of the ten level embeddings orders the levels: Spearman with the level is 0.77–0.92 for base, 0.98–0.99 for lexical and 0.95–0.98 for syntax. PC1 explains 40–63% of the variance. The model discovered ordinality from data alone, which supports ordinal embeddings (2.5) and makes steering along that axis plausible (5.2).
- **Aliasing shows up in the weights.** `base_L` and `lexical_L` moved in similar directions (mean cosine 0.51–0.60), unlike `base_L` and `syntax_L` (0.17–0.31). This is the signature of the lexical bug and a useful sanity check for the retrained models, where it should disappear.
- **Adjacent levels are similar but not interchangeable** (cosine 0.47–0.66), so the bins are not merged.

### 5.2 Experiments, from cheapest to most involved

| Question | Method | What would count as an answer |
|---|---|---|
| Does each token change *what* is edited, beyond the overall score? | **Counterfactual dose-response.** Same input, every request, greedy decoding. Profile outputs with Profiling-UD and compare *outputs under request A vs request B*, reusing the Wilcoxon / rank-biserial code in [`../scripts/build_webview_data.py`](../scripts/build_webview_data.py). | The effect of the token itself, isolated from original-vs-simplified differences. |
| Are the three dimensions controlled separately? | **Specificity matrix.** In a three-token model, move one target token at a time (keep mode) and measure the change in all three output scores. A 3 × 3 matrix: diagonal = intended effect, off-diagonal = leakage. Mechanistically, estimate each dimension's level direction (from embeddings or activations) and check how orthogonal the three are. | Separate dimensions show a dominant diagonal and near-orthogonal directions. For lexical, expect leakage into base given the data (2.3). |
| What does a token push the model to write? | **Guidance differences.** At each step, the logit difference between `lex_30` and `lex_any` (the CFG term) lists the words the lexical token promotes. Aggregate by NBIV membership, frequency class and POS. | E.g. whether the lexical token promotes basic-vocabulary words and the syntax token promotes sentence boundaries and main-clause verbs. |
| Is the level read once, early, or throughout generation? | **Attention to the token positions** across layers and decoding steps, plus **attention knockout**: block attention to the tokens from layer ℓ onward (Geva et al., 2023). | The layers after which blocking no longer removes control: where the information has been copied into the residual stream. |
| Which layers and heads carry the control? | **Activation patching** (Meng et al., 2022; Zhang & Nanda, 2024; Heimersheim & Nanda, 2024). Run the same input with two requests and patch residual-stream states from one run into the other, per layer and position. Metric: log-probability difference of the two reference continuations under teacher forcing, which is cheap and needs no READ-IT. The release's clusters supply such pairs of references for the same input. Then patch individual heads. | A layer × position map of where control is causally used, and possibly a small set of "control-reading" heads. |
| Does the model plan the output's linguistic profile before writing it? | **Linear probes** on the hidden state of the last prompt token, predicting the eventual output's features (length, tree depth, subordination, NBIV share) and READ-IT scores. Use control tasks for selectivity (Hewitt & Liang, 2019), following the lab's own profiling-probe work (Miaschi et al., 2020). | High selectivity before any token is generated means planning. Comparing probes across requests on the same input shows what each token sets. |
| Is control a linear direction? | **Steering.** Take d = mean activation(level 70) − mean activation(level 10) at layer ℓ, add α·d while keeping the tokens fixed, and measure achieved READ-IT and quality (contrastive activation addition, Rimsky et al., 2024; linear-representation view, Park et al., 2024). Steering readability in free-form summaries works but costs quality at high strength (Braun et al., 2025). | A monotonic dose-response in α means control lives on a direction. That enables continuous control and interpolation to rare levels. |
| Which generated words depend on the tokens? | **Contrastive attribution** with Inseq (Sarti et al., 2023; Yin & Neubig, 2022). Attribute each generated token to each control token vs the source tokens, aggregated by POS and dependency relation. | E.g. whether subordinators disappear and auxiliaries appear because of the syntax token: a direct link to the linguistic-transformation results. |

**Practical setup.**
- **Merge the adapter first.** Merge each LoRA adapter into its base model (`peft` `merge_and_unload`) and work with plain PyTorch hooks or nnsight, which handle any Hugging Face model.
- **Replace the t-SNE plots.** The existing t-SNE plots in `GFFF/test_qwen.ipynb` distort distances and cluster sizes (Wattenberg et al., 2016). Use PCA, linear probes and representational similarity analysis for claims about geometry.
- **Check sanity first.** Run every experiment first on the CLiC-it checkpoints, where base and lexical are known aliases, before the retrained models.

---

## References

- Agrawal, S. & Carpuat, M. (2023). Controlling Pre-trained Language Models for Grade-Specific Text Simplification. *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.790
- Alva-Manchego, F., Stodden, R., Imperial, J. M., Barayan, A., North, K. & Tayyar Madabushi, H. (2025). Findings of the TSAR 2025 Shared Task on Readability-Controlled Text Simplification. *TSAR 2025*. https://aclanthology.org/2025.tsar-1.8
- Braun, J. et al. (2025). Beyond Multiple Choice: Evaluating Steering Vectors for Adaptive Free-Form Summarization. arXiv:2505.24859. https://arxiv.org/abs/2505.24859
- Farajidizaji, A., Raina, V. & Gales, M. (2024). Is It Possible to Modify Text to a Target Readability Level? An Initial Investigation Using Zero-Shot Large Language Models. *LREC-COLING 2024*. https://aclanthology.org/2024.lrec-main.815
- Geva, M., Bastings, J., Filippova, K. & Globerson, A. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. *EMNLP 2023*. https://arxiv.org/abs/2304.14767
- Heimersheim, S. & Nanda, N. (2024). How to Use and Interpret Activation Patching. arXiv:2404.15255. https://arxiv.org/abs/2404.15255
- Hewitt, J. & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP-IJCNLP 2019*. https://aclanthology.org/D19-1275
- Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. arXiv:2207.12598. https://arxiv.org/abs/2207.12598
- Kew, T. & Ebling, S. (2022). Target-Level Sentence Simplification as Controlled Paraphrasing. *TSAR 2022*. https://aclanthology.org/2022.tsar-1.4
- Li, Z. & Shardlow, M. (2024). How Do Control Tokens Affect Natural Language Generation Tasks like Text Simplification. *Natural Language Engineering* 30(5). https://doi.org/10.1017/S1351324923000566 (extends Li, Shardlow & Hassan, *TSAR 2022*, https://aclanthology.org/2022.tsar-1.14)
- Martin, L., de la Clergerie, É., Sagot, B. & Bordes, A. (2020). Controllable Sentence Simplification. *LREC 2020*. https://aclanthology.org/2020.lrec-1.577
- Martin, L., Fan, A., de la Clergerie, É., Bordes, A. & Sagot, B. (2022). MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases. *LREC 2022*. https://aclanthology.org/2022.lrec-1.176
- Meng, K., Bau, D., Andonian, A. & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. https://arxiv.org/abs/2202.05262
- Miaschi, A., Brunato, D., Dell'Orletta, F. & Venturi, G. (2020). Linguistic Profiling of a Neural Language Model. *COLING 2020*.
- OUNLP (2025). OUNLP at TSAR 2025 Shared Task: Multi-Round Text Simplifier. arXiv:2511.04495. https://arxiv.org/abs/2511.04495
- Papucci, M., Venturi, G. & Dell'Orletta, F. (2026). Controllable Sentence Simplification in Italian: Fine-Tuning Large Language Models on Automatically Generated Resources. *LREC 2026*, 7178–7191. Dataset: https://huggingface.co/datasets/mpapucci/impacts
- Park, K., Choe, Y. J. & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICML 2024*. https://arxiv.org/abs/2311.03658
- Rafailov, R. et al. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS 2023*. https://arxiv.org/abs/2305.18290
- Ribeiro, L. F. R., Bansal, M. & Dreyer, M. (2023). Generating Summaries with Controllable Readability Levels. *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.714 (code: https://github.com/amazon-science/controllable-readability-summarization)
- Rimsky, N. et al. (2024). Steering Llama 2 via Contrastive Activation Addition. *ACL 2024*. https://arxiv.org/abs/2312.06681
- Sanchez, G., Spangher, A., Fan, H., Levi, E. & Biderman, S. (2024). Stay on Topic with Classifier-Free Guidance. *ICML 2024*. https://arxiv.org/abs/2306.17806
- Sarti, G. et al. (2023). Inseq: An Interpretability Toolkit for Sequence Generation Models. *ACL 2023 System Demonstrations*. https://arxiv.org/abs/2302.13942
- Sennrich, R., Haddow, B. & Birch, A. (2016). Improving Neural Machine Translation Models with Monolingual Data. *ACL 2016*. https://aclanthology.org/P16-1009
- Wattenberg, M., Viégas, F. & Johnson, I. (2016). How to Use t-SNE Effectively. *Distill*. https://distill.pub/2016/misread-tsne/
- Yanamoto, D. et al. (2022). Controllable Text Simplification with Deep Reinforcement Learning. *AACL-IJCNLP 2022*. https://aclanthology.org/2022.aacl-short.49
- Yin, K. & Neubig, G. (2022). Interpreting Language Models with Contrastive Explanations. *EMNLP 2022*. https://arxiv.org/abs/2202.10419
- Zhang, F. & Nanda, N. (2024). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. *ICLR 2024*. https://arxiv.org/abs/2309.16042
- Zhang, T., Liu, F., Wong, J., Abbeel, P. & Gonzalez, J. E. (2023). The Wisdom of Hindsight Makes Language Models Better Instruction Followers. *ICML 2023*. https://proceedings.mlr.press/v202/zhang23ab.html
