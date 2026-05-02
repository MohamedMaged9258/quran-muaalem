# MODEL — Architecture & Adaptation

This document explains the model we are building: what it is, how it processes audio, and how we adapt it from Quranic recitation to Modern Standard Arabic (MSA).

---

## 1. What the Model Does

Given a raw audio waveform of Arabic speech, the model produces a **sequence of phonemes** — the basic sound units of the language. From phonemes we can derive transcription, alignment, and pronunciation feedback.

```
Audio (16 kHz WAV)  ─────►  Model  ─────►  Phoneme sequence: "د َ ر َ س"
```

We do not predict words directly. Phonemes are language-agnostic units that generalize better across speakers and accents than character-level transcription.

---

## 2. Base Architecture: Wav2Vec2-BERT + Multi-Level CTC

The original Quran Muaalem model (`obadx/muaalem-model-v3_2`) is built on top of Meta's **Wav2Vec2-BERT** encoder, with multiple parallel CTC heads on top.

```
                    ┌──────────────────────────────────┐
   Audio waveform   │   Wav2Vec2-BERT Encoder          │
   (16 kHz, mono)   │   (pre-trained on 53k hours      │
        │           │   of multilingual speech)        │
        ▼           └──────────────┬───────────────────┘
                                   │ hidden states
                                   │ (batch, time, 1024)
                                   ▼
                    ┌──────────────────────────────────┐
                    │   Dropout + ModuleDict of heads  │
                    │                                  │
                    │   ├── phonemes head    (Linear) │
                    │   ├── tajweed head     (Linear) │
                    │   ├── sifat head       (Linear) │
                    │   └── ... other levels          │
                    └──────────────┬───────────────────┘
                                   │ logits per level
                                   ▼
                          CTC decoding per level
```

### Key components

| Component | Role |
|---|---|
| **Feature extractor** | Converts 16 kHz waveform into 160-dim mel-style features (one frame ≈ 10 ms). |
| **Wav2Vec2-BERT encoder** | Self-supervised speech encoder. Outputs contextualized 1024-dim hidden states at ~50 Hz (one vector every 20 ms). |
| **`level_to_lm_head`** | A `nn.ModuleDict` mapping each "level" (phonemes, tajweed, sifat, etc.) to its own `nn.Linear(1024, vocab_size)` classifier. |
| **CTC loss** | Trains each head to align variable-length audio frames with variable-length label sequences without explicit per-frame labels. |

The `Wav2Vec2BertForMultilevelCTC` class is defined in [src/quran_muaalem/modeling/modeling_multi_level_ctc.py](src/quran_muaalem/modeling/modeling_multi_level_ctc.py).

### Why CTC?

Audio frames (~50 per second) outnumber phonemes (~10 per second). CTC handles this by introducing a "blank" token and learning a soft alignment, so we never need to label which frame corresponds to which phoneme.

---

## 3. Our Adaptation: MSA Phoneme-Only Head

The pre-trained model has **43 Quranic phoneme classes** (including tajweed-specific marks like elongation, ghunna, qalqalah). For Modern Standard Arabic we only need a smaller, plain phonetic inventory.

### Goal

Replace the 43-class Quranic phoneme head with a **35-class MSA phoneme head**, while keeping the entire encoder frozen.

```
   Original                          Adapted
   ────────                          ────────
   Encoder (frozen)                  Encoder (frozen, identical)
        │                                 │
        ▼                                 ▼
   phonemes head: Linear(1024, 43)   phonemes head: Linear(1024, 35)
   (Quranic + tajweed marks)         (MSA phonemes only)
```

### MSA phoneme inventory (35 tokens)

Defined in [src/quran_muaalem/modeling/msa_vocab.py](src/quran_muaalem/modeling/msa_vocab.py):

| Group | Count | Tokens |
|---|---|---|
| Consonants | 28 | `ء ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي` |
| Vowels / diacritics | 5 | `َ` (fatha), `ُ` (damma), `ِ` (kasra), `ْ` (sukun), `ة` (ta marbuta) |
| Special | 2 | `[PAD]`, `[UNK]` |

The 28 consonants include the four emphatic (pharyngealized) consonants `ص ض ط ظ`, which are distinct phonemes in MSA. An earlier 31-token version of the inventory dropped these and silently mapped them to `[UNK]` during training — that version is fixed: any checkpoint trained against the old vocab must be re-adapted and re-trained against the current 35-class head.

### Tokenizer

The `MSATokenizer` in [src/quran_muaalem/modeling/msa_tokenizer.py](src/quran_muaalem/modeling/msa_tokenizer.py) converts whitespace-separated phoneme strings (e.g. `"د َ ر َ س"`) into integer IDs and back. `[PAD]=0` is also the CTC blank.

### How the resize works

The script [src/quran_muaalem/modeling/adapt_model_for_msa.py](src/quran_muaalem/modeling/adapt_model_for_msa.py) performs a one-time surgery on the pre-trained checkpoint:

1. Load `obadx/muaalem-model-v3_2` (43-class phonemes head).
2. Read the head's `in_features` (1024) and `out_features` (43).
3. Build a new `nn.Linear(1024, 35)`. The size is read from `MSA_PHONEME_COUNT`, so updating the inventory automatically updates the head.
4. Initialize weights with small Gaussian noise, biases at zero.
5. **Warm-start**: copy rows `0..34` of the old weight matrix and bias into the new layer (the first 35 Quranic phoneme rows overlap with MSA phonemes by index).
6. Replace `model.level_to_lm_head["phonemes"]` with the new layer.
7. Update `config.level_to_vocab_size["phonemes"] = 35`.
8. Save the resulting checkpoint to `checkpoints/msa_model_adapted/` — including the feature extractor (`preprocessor_config.json`) so the directory is self-contained for `AutoFeatureExtractor.from_pretrained`.

The other heads (tajweed, sifat, etc.) are left in place but ignored during MSA training — only the `phonemes` head receives gradients.

### Why warm-start instead of random init?

The Linear layer mixes the same 1024-dim encoder features regardless of what the output classes mean. Copying overlapping rows preserves the encoder's already-good "what does this frame sound like?" signal for shared phonemes, so training converges faster than from scratch.

---

## 4. Tensor Shapes End-to-End

For an input of `T_audio` waveform samples at 16 kHz, with the feature extractor's 10 ms hop:

| Stage | Tensor shape | Notes |
|---|---|---|
| Raw audio | `(batch, T_audio)` | `T_audio = 16000 × duration` |
| Feature extractor output | `(batch, T_feat, 160)` | `T_feat ≈ T_audio / 160` |
| Encoder hidden states | `(batch, T_enc, 1024)` | `T_enc ≈ T_feat / 2` (~50 Hz) |
| Phoneme logits | `(batch, T_enc, 35)` | one distribution over 35 classes per ~20 ms frame |

For a 15-second clip: `T_audio = 240,000` → `T_feat ≈ 1500` → `T_enc ≈ 187` frames → 187 phoneme distributions.

---

## 5. Why This Design

| Decision | Rationale |
|---|---|
| **Freeze the encoder** | It was pre-trained on 53k hours of speech. Fine-tuning it on ~17 hours of MSA would mostly hurt generalization. The CTC head has all the capacity we need to learn the new vocabulary. The trainer enforces this in [`load_model_for_msa`](src/quran_muaalem/training/train_msa.py): every parameter is set to `requires_grad=False`, then only the `phonemes` head is re-enabled. AdamW is built from the trainable subset, so no optimizer state is allocated for frozen weights — fits on a 4 GB GPU. |
| **Phoneme-only output** | The other levels (tajweed, sifat) are Quran-specific. MSA doesn't need them, and dropping them simplifies labels, loss, and evaluation. |
| **Reuse the multi-level class** | We keep the original `Wav2Vec2BertForMultilevelCTC` and just resize one head, so the engine/inference code keeps working unchanged. |
| **35-class inventory** | The 28 canonical MSA consonants (including the emphatics `ص ض ط ظ`) + 5 vowel/diacritic tokens + `[PAD]`/`[UNK]`. Dropping the emphatics — as an earlier 31-class version did — silently maps them to `[UNK]` and corrupts ~10–15% of training labels. |

### Trainable parameter count

| Component | Params |
|---|---|
| Wav2Vec2-BERT encoder | ~605 M, **frozen** |
| Other CTC heads (tajweed, sifat, …) | ~50 K, **frozen** |
| **MSA phonemes head** (`Linear(1024, 35)`) | **~36 K, trainable** |

The training run touches only ~0.005% of the total parameter count.

---

## 6. Files in This Layer

| File | Purpose |
|---|---|
| [src/quran_muaalem/modeling/modeling_multi_level_ctc.py](src/quran_muaalem/modeling/modeling_multi_level_ctc.py) | Original multi-level CTC model class. Untouched. |
| [src/quran_muaalem/modeling/configuration_multi_level_ctc.py](src/quran_muaalem/modeling/configuration_multi_level_ctc.py) | Config object holding `level_to_vocab_size` and friends. |
| [src/quran_muaalem/modeling/msa_vocab.py](src/quran_muaalem/modeling/msa_vocab.py) | The 31-token MSA phoneme inventory. |
| [src/quran_muaalem/modeling/msa_tokenizer.py](src/quran_muaalem/modeling/msa_tokenizer.py) | Encode / decode between phoneme strings and IDs. |
| [src/quran_muaalem/modeling/adapt_model_for_msa.py](src/quran_muaalem/modeling/adapt_model_for_msa.py) | One-shot script that produces `checkpoints/msa_model_adapted/`. |

See [TRAINING.md](TRAINING.md) for how the adapted model is trained, and [RUNNING.md](RUNNING.md) for how it is served.
