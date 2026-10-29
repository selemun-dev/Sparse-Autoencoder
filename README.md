# Sparse Autoencoder on GPT-2 Small — Group 9 (SAE-only track)

**iCog Labs internship — Mechanistic Interpretability focused training**
Author: [your name] · Mentor: [mentor name] · Notebook: `[notebook file name].ipynb`

---

## 1. Summary

I trained and evaluated two small sparse autoencoders (SAEs) on activations of a frozen GPT-2 Small, at one fixed site (the block-6 post-residual state), using ordinary text from WikiText-2. The two SAEs are identical except for their sparsity penalty (λ = 0.0003 and λ = 0.003).

| | λ = 0.0003 (low) | λ = 0.003 (high) |
|---|---|---|
| Held-out reconstruction MSE | 0.0696 | 0.2610 |
| Mean-activation baseline MSE | 0.5262 | 0.5262 |
| Variance accounted for (held-out) | 0.867 | 0.503 |
| Mean active latents per token (L0, threshold 1e-6) | 543.4 of 1,024 | 53.9 of 1,024 |
| Latents never active on the held-out sample | 0.0 % | 0.0 % |

A ten-times larger penalty made the code about ten times sparser (L0 543.4 → 53.9) and raised the reconstruction error about 3.75 times (MSE 0.0696 → 0.2610). Both runs reconstruct better than the mean-activation baseline. Three latents of the high-penalty SAE were inspected and tested on new text; of 12 written predictions, 8 held and 4 did not, and every failed prediction led to a revised description (Section 7).

This is a small, single-seed experiment on a limited corpus. It is meant to train and critically evaluate a small SAE, not to recover a research-quality GPT-2 dictionary (see the limitations in Section 10).

---

## 2. Task and scope

**Source.** The "Research Assignment — Activation Patching & Sparse Autoencoders" Notion page, with the study chapter "Sparse Autoencoders" and the overview "Mechanistic Interpretability — Focused Training".

**Group 9 scope.** The mentor's note for Group 9: *the causal intervention and activation patching are removed, and the task is only about sparse autoencoders.* Everything about the IOI patching prompts, patch sites, patching baselines and controls is therefore out of scope here. Only the "Train and evaluate your own SAE" part and its deliverables are covered.

### Requirement checklist

| # | Required by the assignment | Status | Evidence (cell) |
|---|---|---|---|
| 1 | Collect about 10,000–20,000 token-position activation vectors at one fixed site; remove padding; split passages into train and held-out **before** extraction; record site, provenance, seeds, shapes | Done | 13,103 train + 3,276 held-out vectors; 208 / 52 passages split before extraction (Cells 2, 3, 11) |
| – | Frozen GPT-2 in evaluation mode, gradients off while extracting, LM passes separate from SAE training | Done | Cells 2, 3 |
| – | Consistent activation scaling | Done | One scalar fit on training data only (Cell 4) |
| – | Encoder/decoder as specified, 1,024 latents, MSE + L1 loss, decoder norms constrained, reduction conventions stated | Done | Cell 5, Section 4 |
| 2 | Two SAEs, same architecture/split/steps/init seed, two nonzero penalties with a clear contrast, 300–600 steps, batch 128, learning curves, checkpoints | Done | λ = 0.0003 and 0.003; 500 steps; Cells 7, 8, 10, 11 |
| 3 | Held-out MSE (stated normalization), L0 with explicit threshold, % never-active latents, mean-activation baseline | Done | Cells 8, 8b |
| 4 | Inspect 2–3 latents (top contexts), tentative description, test on ≥ 2 new counterexamples; report unclear semantics honestly | Done | Latents 428, 685, 449; Cells 9, 9b; 12 tests |
| 5 | Explain the trade-off; one figure comparing the two runs | Figure done, explanation in Section 8 | Cell 10 |
| – | Notebook, saved outputs, 700–1,000-word report, reproducibility record | Notebook, outputs and record done. The report draft is generated (Cell 12) and its `[WRITE]` parts must be completed by the author | Cells 11–13 |

---

## 3. Environment and reproducibility record

| Item | Value |
|---|---|
| Model | `gpt2` (GPT-2 Small, 12 blocks, d_model = 768), frozen, evaluation mode, gradients disabled |
| Site | Hugging Face `hidden_states[7]` = output of block index 6 = TransformerLens `blocks.6.hook_resid_post` (zero-based). The convention was **not cross-checked against TransformerLens** in this run |
| Corpus | WikiText-2 raw, train split (`Salesforce/wikitext`); paragraphs longer than 300 characters, not headings; 13,290 usable paragraphs, 260 sampled at random (seed 0) |
| Passages | 208 train, 52 held-out (80/20 split **by passage, before extraction**), first 64 tokens of each |
| Vectors | 13,103 train, 3,276 held-out (16,379 total), 768 dimensions; padding removed; token position 0 excluded |
| Activation scale | 0.3120692 (train mean vector norm becomes √768 = 27.71; held-out mean norm 27.79) |
| SAE | 1,024 latents; 1,574,656 trainable parameters; written in NumPy with hand-derived gradients |
| Training | Adam, learning rate 1e-3, 500 steps, batch size 128, penalties 0.0003 and 0.003 |
| Seeds | data split / sampling seed 0; SAE initialization seed 0; batch-order seed 0 (identical in both runs) |
| Device | CPU |
| Libraries (as recorded by the notebook) | python 3.13.18, numpy 2.1.3, torch 2.11.0+cpu, transformers 5.20.2, datasets 4.8.5 |
| Active-latent threshold | f > 1e-6 (sensitivity at 1e-2 and 1e-1 in Section 6) |

The saved files contain everything needed to inspect or regenerate the work: the exact passages (`passages.json`), token ids per passage (`tokens.json`), both SAE checkpoints and `results_and_record.json`. The activation cache itself is not saved because it can be regenerated from the passages and seeds.

---

## 4. Method

### 4.1 Activation collection
Each paragraph is tokenized (GPT-2 tokenizer, end-of-text token used as padding, right padding), truncated to 64 tokens, and run through the frozen model in batches with gradients off. The residual-stream state at the site is read for each token. Padding positions are dropped using the attention mask. For every kept vector the code records the passage id and the token position, so activations can be traced back to text.

**Token position 0 is excluded.** Measured on the first 16 training passages, its mean activation norm was **3034.4** against **88.9** at later positions (about 34 times larger). Because squared error grows with the square of the norm, one such vector would carry as much squared size as roughly a thousand ordinary vectors and would dominate both training and the MSE. This was checked on 16 passages only, not on all of the data.

### 4.2 Scaling
One scalar is fit on the training vectors only so that their mean norm becomes √768, and the same scalar is applied to the held-out vectors. All MSE values in this document are in these scaled units.

### 4.3 SAE
```
f(h)  = ReLU((h − b_dec) · W_enc + b_enc)        h ∈ R^768, f ∈ R^1024
ĥ     = f · W_dec + b_dec
loss  = mean over batch of [ (1/768)·‖h − ĥ‖² + λ·‖f‖₁ ]
```
- Reduction convention: the squared error is summed over the 768 dimensions and divided by 768; the L1 term is summed over the 1,024 latents; both are averaged over the batch.
- Decoder rows (the feature directions) are renormalized to unit length after every optimizer step, which closes the scale loophole in the L1 penalty.
- Matrix shapes in code: `W_enc` is 768×1024 and `W_dec` is 1024×768 because examples are stored as rows (the assignment's equations, transposed). `W_enc` starts as the transpose of the random unit-norm decoder; `b_enc` starts at zero; `b_dec` starts at the training mean.
- The same initialization seed and batch order are used in both runs.

### 4.4 Implementation check
The SAE and the Adam optimizer are implemented in NumPy with hand-written gradients. A gradient check on a tiny random problem (5-dimensional inputs, 8 latents, 6 examples, float64) compares every analytical gradient with a finite-difference estimate: **maximum relative error 5.46e-09**. This check validates the gradient code; it was not run on the GPT-2 data.

### 4.5 Choosing the two penalties
A pilot trained five penalties for 150 steps on **training data only** (the held-out set was not touched). Training baseline MSE (always predict the training mean): 0.5224.

| λ | Active latents (L0) of 1,024 | Train MSE | Share of baseline error removed |
|---|---|---|---|
| 0.0001 | 580.4 | 0.0698 | 86.6 % |
| 0.0003 | 490.9 | 0.0870 | 83.3 % |
| 0.001 | 302.0 | 0.1569 | 70.0 % |
| 0.003 | 90.1 | 0.3207 | 38.6 % |
| 0.01 | 6.9 | 0.5078 | 2.8 % |

Selection rule (fixed before any held-out result was seen): **high λ = the largest value whose training MSE removed at least 30 % of the baseline error; low λ = ten times smaller.** This gives λ = 0.003 (38.6 %) and λ = 0.0003. λ = 0.01 was rejected because it uses about 7 latents and removes only 2.8 % of the error (close to collapse). The 30 % threshold is a choice made by the author. [CONFIRM: this is the rule you applied.]

---

## 5. Cell-by-cell documentation

| Cell | What it does | Key output |
|---|---|---|
| 1 | Imports; constants (`SEED = 0`, `D_MODEL = 768`, `N_LATENTS = 1024`) | none |
| 2 | Loads frozen GPT-2 Small and its tokenizer; sets site, 64 tokens per passage, 260 passages, position-0 exclusion; loads WikiText-2; filters paragraphs; splits 80/20 **by passage before extraction** | 13,290 usable passages; 208 train / 52 held-out |
| 3 | `collect_batch` (drops padding and position 0, records passage id and position) and `extract` (batched, no-grad forward passes reading `hidden_states[7]`); norm check at position 0 | train (13103, 768); held-out (3276, 768); norm 3034.4 at position 0 vs 88.9 later |
| 4 | Fits one scale on training vectors and applies it to both sets | scale 0.3121; mean norm 27.71 (train), 27.79 (held-out) |
| 5 | Defines `SparseAutoencoder` (encode, decode, loss, hand-written gradients, decoder normalization) and `Adam` | none |
| 6 | Gradient check against finite differences | max relative error 5.46e-09 |
| 7 | `train_sae`, `evaluate`, `lambda_pilot`; runs the pilot on training data | pilot table in Section 4.5 |
| 8 | Trains the two SAEs (identical seeds, 500 steps, batch 128) and evaluates both on the held-out vectors | table in Section 6 |
| 8b | Train-vs-held-out error and sensitivity of L0 / never-active to the threshold | Section 6 |
| 8c | Top contexts for three latents of the low-penalty SAE | Section 7.3 |
| 9 | Picks three high-penalty latents by a pre-set rule and prints their strongest held-out contexts | latents 428, 685, 449 (Section 7.1) |
| 9b | Tests the latent descriptions on 12 new sentences with written predictions | 8 of 12 predictions held (Section 7.2) |
| 10 | Four-panel comparison figure of the two runs | Section 6 and `two_runs.png` |
| 11 | Saves passages, tokens, checkpoints, figure and `results_and_record.json`; lists file sizes | six files, zip of 11.9 MB |
| 12 | Builds `report_draft.md` from the measured numbers (interpretation left as `[WRITE]` notes) | draft with word count |
| 13 | Re-zips the outputs and downloads `outputs_small.zip` | download completed |

---

## 6. Results

### 6.1 Held-out evaluation (3,276 vectors; "active" means latent > 1e-6)

| | λ = 0.0003 | λ = 0.003 |
|---|---|---|
| MSE (held-out) | 0.0696 | 0.2610 |
| MSE (full training set) | 0.0525 | 0.2212 |
| Mean-activation baseline MSE | 0.5262 | 0.5262 |
| Variance accounted for, held-out | 0.867 | 0.503 |
| Variance accounted for, train | 0.899 | 0.577 |
| Mean active latents, L0 | 543.4 (53.1 % of dictionary) | 53.9 (5.3 % of dictionary) |
| Mean L1 norm of the code | 260.6 | 30.2 |
| Latents never active on the held-out sample | 0.0 % | 0.0 % |

The baseline MSE (training mean) and the variance-accounted-for figures agree with each other (1 − 0.0696/0.5262 = 0.868; 1 − 0.2610/0.5262 = 0.504), as expected when the training and held-out means are close.

### 6.2 Sensitivity to the "active" threshold (held-out)

| Threshold | L0, λ = 0.0003 | L0, λ = 0.003 | Never-active latents (both) |
|---|---|---|---|
| 1e-6 | 543.4 | 53.9 | 0.0 % |
| 1e-2 | 535.1 | 52.5 | 0.0 % |
| 1e-1 | 461.7 | 42.1 | 0.0 % |

The ratio of the two L0 values stays near ten at every threshold (10.1× at 1e-6, 11.0× at 1e-1). Every latent of both SAEs is active at least once on the held-out sample even at a threshold of 0.1. "Never active" refers to this finite sample only.

### 6.3 Generalization gap
Held-out MSE is higher than training-set MSE in both runs: 0.0696 vs 0.0525 for the low penalty (33 % higher) and 0.2610 vs 0.2212 for the high penalty (18 % higher). The held-out set is only 52 passages, and there is a single seed, so this gap carries no confidence interval.

### 6.4 Training dynamics (Cell 8 log and Cell 10 figure; plot values are approximate readings)

| Step | Low λ: batch MSE / L0 | High λ: batch MSE / L0 |
|---|---|---|
| 1 | 0.2763 / 509.8 | 0.2883 / 509.8 |
| 100 | 0.1121 / 489.1 | 0.3716 / 106.5 |
| 300 | 0.0632 / 509.1 | 0.2538 / 62.8 |
| 500 | 0.0551 / 525.8 | 0.2260 / 46.2 |

- **Low λ:** MSE falls quickly and flattens around 0.055 by about step 400; L0 slightly **rises** (about 489 at step 100 to about 526 at step 500).
- **High λ:** the training MSE **first rises** from about 0.26 to about 0.36 (peak near step 85) while L0 collapses from about 510 to about 100 within the first 100 steps, and only then declines. At step 500 both its MSE and L0 are still decreasing, so this run is **not converged**.
- Final-20-step means: MSE 0.055 / 0.227 and L0 526 / 49 (low / high λ). These are single noisy mini-batches, so the full-set values in 6.1 are the ones to report.

Figure: `two_runs.png` (training MSE over steps, training L0 over steps, held-out MSE with the baseline as a dashed line, held-out L0).

---

## 7. Feature inspection

Latents were chosen by a rule fixed before any context was read: among latents active on 0.5 %–20 % of held-out vectors, take the three with the largest maximum activation. Contexts show 12 preceding tokens (visible to the model), the token where the latent fires in `[[double brackets]]`, and 3 following tokens that are **not visible to the model at that position** (GPT-2 reads left to right).

### 7.1 High-penalty SAE (λ = 0.003): initial descriptions from top contexts

| Latent | Strongest held-out activations | Initial description |
|---|---|---|
| 428 | 12.41 `… 9 metres ( 30[[ ft]]`; 12.02 `… 31[[ ft]]`; 11.32 `… 9 @.@ 4[[ m]]`; 9.29 `centimet[[res]]`; 8.48 `3 @.@ 9[[ in]]`; 8.32 `10[[ cm]]` | A length unit right after a number |
| 685 | 12.29 / 11.80 / 11.78 / … all on the second `@` of the WikiText marker `@-@` (`middle @-[[@]]`, `well @-[[@]]`, `Anglo @-[[@]]`) | The second `@` of WikiText's `@-@` hyphen marker, i.e. a corpus formatting artifact |
| 449 | 11.35 `in[[ March]]`; 9.92 `on[[ May]]`; 9.68 `its[[ June]]`; 8.37 `In[[ December]]`; 8.22 `of[[ February]]` | A month name in a date expression |

### 7.2 New-text tests (12 sentences, predictions written before running)

A latent "fires" on a sentence if its peak activation (position 0 ignored) is at least **25 % of its largest held-out activation**. This 25 % rule is a choice made for this analysis; the cut-offs are about 3.10 (latent 428), 3.07 (685) and 2.84 (449). [CONFIRM: the predictions were not changed after seeing the results.]

| Latent | Sentence | Predicted | Peak (token) | Result |
|---|---|---|---|---|
| 428 | The bridge is 120 metres long and the tower is 300 ft tall. | fire | 11.08 (` ft`) | as predicted |
| 428 | The poem is written in an irregular metre. | quiet | 2.95 (` metre`) | as predicted (borderline: 95 % of the cut-off) |
| 428 | It took 30 minutes to walk there. | quiet | 4.85 (` minutes`) | **surprise** (fired) |
| 428 | The room is 5 m wide and 8 m long. | fire | 9.48 (` m`) | as predicted |
| 685 | He was a well @-@ known author. | fire | 13.01 (`@`) | as predicted |
| 685 | He was a well-known author. | quiet | 0.00 | as predicted (no activation at all) |
| 685 | The value was 3 @.@ 5 percent. | quiet | 10.23 (`@`) | **surprise** (fired) |
| 685 | Contact me at john@example.com. | quiet | 6.56 (`@`) | **surprise** (fired) |
| 449 | The film was released in March 2010 and sold well by June. | fire | 11.54 (` March`) | as predicted |
| 449 | You may go home now, she said. | quiet | 0.00 | as predicted |
| 449 | The soldiers began to march at dawn. | quiet | 2.02 (` dawn`) | as predicted |
| 449 | We met on Monday and again in the summer. | quiet | 5.09 (` Monday`) | **surprise** (fired) |

**8 of 12 predictions held** (428: 3 of 4; 685: 2 of 4; 449: 3 of 4).

### 7.3 Revised descriptions (hypotheses supported by 4 tests each, not established facts)

- **Latent 428** responds to a unit-of-measure word after a number, strongest for length units (ft, m, metres, cm, in). It also fired on "minutes" (39 % of its maximum), so "length unit" is too narrow; the word "metre" without a number stayed just below the cut-off.
- **Latent 685** responds to the **`@` token** rather than to a hyphen: strongest in `@-@` (13.01), also in `@.@` (10.23, 83 % of its maximum) and in an email address (6.56, 53 %), and silent on an ordinary hyphen. In the held-out data its strongest contexts are all the WikiText `@-@` marker, so it is partly a corpus-formatting feature.
- **Latent 449** responds to calendar words: month names most strongly, and a weekday ("Monday", 45 % of its maximum). The verb "march" did not reach the cut-off.

### 7.4 Low-penalty SAE (λ = 0.0003): top contexts (Cell 8c)

Latents were picked with the same kind of rule but a higher activity threshold (1.0), because this code is dense, so the two selections are not identical.

| Latent | Strongest contexts | Description |
|---|---|---|
| 685 | 6.54 `middle @-[[@]]`; 6.42 `well @-[[@]]`; 6.34 `Class @-[[@]]` | The `@` of `@-@` (same index and behaviour as in the high-penalty run) |
| 428 | 6.04 `9 @.@ 4[[ m]]`; 4.57 `30[[ ft]]`; 4.29 `800[[ m]]`; 4.07 `cent[[imet]]` | Length units after a number (same index and behaviour) |
| 231 | 5.87 `in[[ 1942]]`; 5.77 `in[[ 1942]]`; 5.46 `in[[ 1914]]`; 5.21 `late[[ 1890]]`; 4.73 `In the[[ 1920]]` | A year or decade number, usually after "in" |

The same latent indices (685, 428) appear in both runs with matching top contexts. Both runs share their random initialization, so the indices may correspond to the same features, but this was **not verified** (for example by comparing decoder directions). The maximum activations of these latents are about half as large in the low-penalty run (6.54 vs 12.29 for latent 685), which may be because its reconstruction is shared among many more active latents; this explanation is untested.

---

## 8. Trade-off discussion

**Does the stronger penalty reduce activation density?** Yes. A tenfold larger penalty reduced the mean number of active latents per token from 543.4 to 53.9 (about 10×), consistently across thresholds, while the mean L1 norm of the code fell from 260.6 to 30.2.

**What happens to reconstruction?** It gets worse: held-out MSE rose from 0.0696 to 0.2610 (3.75×) and variance accounted for fell from 0.867 to 0.503. Even the sparse code still removes about half of the baseline error (50.4 % against 86.8 % for the low penalty). In the high-penalty run the training MSE also rose temporarily during the first ~85 steps before it declined (Section 6.4).

**Can a model reconstruct well without discovering semantically clean features?** The evidence here only partly answers this. The low-penalty SAE reconstructs 86.7 % of held-out variance while using about 543 of 1,024 latents per token, a dense code in which each token is described by hundreds of features, so good reconstruction does not by itself imply a small set of clean features per token. The three low-penalty latents that were inspected were readable from their top contexts, but only three latents per run were examined, chosen by maximum activation among moderately frequent latents, so the sample is likely optimistic and says nothing about the other ~1,000 latents. Cleanliness was judged only from top contexts and 4 tests per high-penalty latent.

---

## 9. Observations that may have run against expectations

These are facts from this run; whether any of them "challenged an initial expectation" depends on what the author expected beforehand. [Choose the one that is true for you and say so in the report.]

1. **No dead latents.** Both runs had 0.0 % never-active latents, even at a threshold of 0.1, and even for the strongly penalized run (a stronger penalty might have been expected to switch some latents off permanently within 500 steps).
2. **Non-monotonic training of the high-penalty run.** The training MSE rose before it fell (about 0.26 → 0.36 near step 85) while the code became much sparser.
3. **A "clean" feature was a corpus artifact.** Latent 685 looked like a crisp feature in its top contexts, but it tracks the WikiText `@-@` marker and also fires on other `@` characters.
4. **Descriptions were too narrow in 4 of 12 tests.** Length units also covered "minutes", the `@` latent fired on `@.@` and an email address, and the month latent fired on "Monday".
5. **The low-penalty code is not very sparse.** About 53 % of the dictionary is active per token, and its L0 rose slightly during training.

---

## 10. Limitations

- **Small data and short training:** 13,103 training vectors, 500 steps, one corpus (WikiText-2), only the first 64 tokens of each paragraph, one site (block 6), one seed, and no confidence intervals. The held-out set is 52 passages.
- **The high-penalty run is not converged** (MSE and L0 were still decreasing at step 500).
- **Coarse penalty grid and short pilot:** five penalties spaced by about a factor of three, trained for 150 steps, while the final runs use 500 steps; a different pilot length could select different values.
- **Threshold dependence:** L0 and "never active" depend on the activity threshold (checked at 1e-6, 1e-2, 1e-1). "Never active" only describes this sample.
- **Feature inspection is small and informal:** three latents per run, selected by maximum activation, descriptions from top contexts and 12 test sentences (4 per latent). The 25 % "fires" rule is an arbitrary choice. Labels are tentative hypotheses.
- **Not checked:** the site indexing against TransformerLens; whether same-index latents in the two runs are the same feature; whether the SAE preserves the language model's behaviour when spliced in (next-token loss); other layers or seeds; dead-latent resampling.
- **Position 0 was excluded** after a norm check on 16 passages, so nothing here applies to the first token of a text.
- **Implementation choice:** the SAE is in NumPy (gradients checked numerically on a tiny problem), while the study chapter's example uses PyTorch. [CONFIRM with the mentor that this is acceptable.]

---

## 11. Files and how to reproduce

`outputs_small.zip` (11.9 MB) contains:

| File | Contents |
|---|---|
| `results_and_record.json` | Model, site, corpus, split, counts, scale, parameters, settings, seeds, versions, pilot table, held-out results, threshold sensitivity, counterexample tests, gradient-check value |
| `sae_lambda_0.0003.npz`, `sae_lambda_0.003.npz` | SAE checkpoints (`W_enc`, `b_enc`, `W_dec`, `b_dec`) |
| `passages.json` | The exact train and held-out passages |
| `tokens.json` | Token ids of every passage (for locating contexts) |
| `two_runs.png` | The four-panel comparison figure |
| `report_draft.md` | Report draft with measured numbers filled in |

**To reproduce:** open the notebook in Colab (CPU is enough), run Cells 1 → 13 in order (after a runtime reset use *Run all*). Cells 8c and 9b define or use helpers that must exist first; if a `NameError` appears, run Cell 9 before them. Seeds are fixed, so the split, initialization and batch order repeat; small numerical differences between machines are possible.

---

## 12. AI-assistance disclosure

Parts of the code (SAE class, evaluation helpers, feature-inspection and report-draft cells) were written with the help of an AI assistant (Claude, by Anthropic), and so were drafts of the test sentences and predictions in Cell 9b. I ran every cell, checked the hand-written gradients numerically, and can explain each step. All numbers in this document come from the outputs of my own runs.

---

## 13. Open questions for the mentor

1. Does "block-6 post-residual" mean the output of the zero-indexed block 6 (`blocks.6.hook_resid_post`, used here) or the sixth block counted from one?
2. Is a NumPy implementation of the SAE acceptable, or should it be PyTorch?
3. With the patching items removed for Group 9, does the 700–1,000-word report length still apply (and do tables count)?
