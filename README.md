# Review Star Rater: fine-tuning Qwen2.5 with QLoRA to predict Yelp ratings

Given a Yelp review, predict the star rating (1–5) the reviewer gave.

I fine-tuned **Qwen2.5-0.5B** and **Qwen2.5-1.5B** with **QLoRA** (4-bit quantization + LoRA adapters) on a free Kaggle T4 GPU. I ran 7 training runs, changing one thing at a time, to find out what actually improves the model.

**Result:** accuracy went from **42.7% → 65.0%**, and predictions within 1 star from **78% → 98.3%**.

---

## Results

Evaluated on **300 unseen test reviews** (the same 300 for every model).

| Model | Base | Training reviews | Epochs | Accuracy ↑ | Within 1 star ↑ | MAE ↓ | Val loss ↓ |
|---|---|---|---|---|---|---|---|
| No fine-tuning | 0.5B | – | – | 0.427 | 0.780 | 0.889 | – |
| Run 1: r=16, lr=2e-4, attention | 0.5B | 2,000 | 1 | 0.590 | 0.957 | 0.467 | – |
| Exp A: bigger adapter (r=32) | 0.5B | 2,000 | 1 | 0.590 | 0.943 | 0.487 | 0.337 |
| Exp B: smaller learning rate (1e-4) | 0.5B | 2,000 | 1 | 0.580 | 0.950 | 0.497 | 0.329 |
| Exp C: attention + MLP layers | 0.5B | 2,000 | 1 | 0.573 | 0.947 | 0.490 | 0.320 |
| Exp D: 2 epochs | 0.5B | 2,000 | 2 | 0.597 | 0.960 | 0.453 | 0.318 |
| Exp E: 5,000 reviews | 0.5B | 5,000 | 2 | 0.607 | 0.967 | 0.440 | 0.302 |
| No fine-tuning | 1.5B | – | – | 0.560 | 0.927 | 0.547 | – |
| **Exp F: Qwen2.5-1.5B (best)** | **1.5B** | **5,000** | **2** | **0.650** | **0.983** | **0.383** | **0.271** |

- **Accuracy:** exact star predicted correctly
- **Within 1 star:** prediction is at most 1 star off
- **MAE (mean absolute error):** average distance from the true rating, in stars
- **Val loss:** loss on 200 held-out validation reviews (Run 1 was trained in a different notebook, so its loss isn't comparable)

---

## What I learned

1. **Tuning the adapter didn't help.** A bigger adapter (r=32), a lower learning rate, or adapting more layers (MLP) gave no gain on 2,000 reviews. Capacity was not the bottleneck.
2. **More training and more data helped a little.** 2 epochs: +0.7 points. 5,000 reviews instead of 2,000: another +1 point, and the best validation loss for the 0.5B model.
3. **A smarter base model helped the most.** Switching to Qwen2.5-1.5B with the same settings gave **+4.3 points**. The untrained 1.5B already scored 56%, close to the fine-tuned 0.5B.
4. **Lower loss ≠ higher accuracy.** Exp C had a lower validation loss than A and B but the lowest accuracy. Always check the real metrics.
5. **Watch training vs validation loss.** In no run did validation loss keep rising while training loss fell (no overfitting). Validation loss flattened during epoch 2, so 2 epochs was about the sweet spot.
6. **Save before you evaluate.** I lost a trained adapter once because a runtime reset before it was saved. Every adapter here is pushed to Hugging Face right after training, before testing.

### Where the model still struggles
The confusion matrix shows almost all mistakes are only 1 star off. The hardest cases are **3-star (mixed) reviews** and telling **4 from 5 stars**, which people also disagree on. For context, the best published models on Yelp 5-class reach roughly 70–73% accuracy.

---

## Method

| | |
|---|---|
| Dataset | [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) (650k train / 50k test reviews) |
| Splits | Training: 2,000 or 5,000 from the train split · Validation: 200 from the train split, never used for training · Test: 300 from the test split |
| Prompt | `Review: <first 600 characters>\n\nHow many stars (1-5) did the reviewer give?\nStars:` → completion ` 4` |
| Quantization | 4-bit NF4 with double quantization (bitsandbytes), compute in float32 (the T4 has no real bf16) |
| LoRA | r=16, alpha=32, dropout=0.05, on `q_proj, k_proj, v_proj, o_proj` (0.28–0.44% of parameters trained) |
| Training | TRL `SFTTrainer`, completion-only loss, batch 4 × gradient accumulation 4 = 16, lr 2e-4, cosine schedule, 3% warmup, `paged_adamw_32bit`, max length 192 |
| Evaluation | Greedy generation of up to 3 tokens; the first digit 1–5 is the prediction |
| Hardware | Free Kaggle T4 GPU: 6 min (0.5B, 2k reviews, 1 epoch) to 89 min (1.5B, 5k reviews, 2 epochs) |

---

## Files

| File | What it is |
|---|---|
| `review_star_rater_kaggle_experiments.ipynb` | Main notebook: resume check, confusion matrix, Experiments A–F, losses and final comparison (with outputs) |
| `review_star_rater.ipynb` | Clean version of the Run 1 pipeline: baseline, training, saving to Drive + Hugging Face, reloading the adapter |

## Models on Hugging Face

| Adapter | Repo |
|---|---|
| **Best (Exp F, Qwen2.5-1.5B)** | [`shivkrchy9876/review-star-rater-exp-f-1.5b`](https://huggingface.co/shivkrchy9876/review-star-rater-exp-f-1.5b) |
| Best 0.5B (Exp E) | [`shivkrchy9876/review-star-rater-exp-e-5k`](https://huggingface.co/shivkrchy9876/review-star-rater-exp-e-5k) |

Each repo also has a `results.json`; the best one has `final_comparison.json` with every run.

## Use the best model

```python
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
ADAPTER = "shivkrchy9876/review-star-rater-exp-f-1.5b"

tokenizer = AutoTokenizer.from_pretrained(BASE)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=quant,
                                            dtype=torch.float32, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

def predict_stars(review: str) -> int | None:
    prompt = f"Review: {review[:600]}\n\nHow many stars (1-5) did the reviewer give?\nStars:"
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=3, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.match(r"^[1-5]", answer)
    return int(m.group()) if m else None

print(predict_stars("The food was amazing and the staff were so friendly. Will come back!"))
```

Needs a CUDA GPU (bitsandbytes 4-bit). Tested with `transformers==4.57.6`, `trl==0.25.1`, `bitsandbytes==0.48.2`, `peft`.

---

## Next steps

- More training data (10k–20k reviews) and longer review text (the verdict is often at the end of a review)
- Score the five answers by probability instead of generating text, and predict the expected star value to lower MAE
- A larger test set (1,000+ reviews) so small differences between runs are statistically clearer
- A live Gradio demo on Hugging Face Spaces
