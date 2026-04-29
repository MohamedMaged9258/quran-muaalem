# Fine-Tuning Guide: Adapting Quran Muaalem for Normal Arabic

This document explains the key components of the Quran Muaalem project and how they relate to your goal of fine-tuning the model for normal (non-Quranic) Arabic recitation.

## 1. Model Architecture: Multi-Level CTC

### Overview
The model uses a **Multi-Level Connectionist Temporal Classification (CTC)** architecture built on **Wav2Vec2-BERT**.

**File:** `src/quran_muaalem/modeling/modeling_multi_level_ctc.py`

### How It Works
```
Audio Input (16kHz)
    ↓
Wav2Vec2-BERT Encoder (speech feature extraction)
    ↓
Multiple CTC Heads (one per linguistic level)
    ├─ Phonemes Level (e.g., 'ء', 'ن', 'ل')
    ├─ Tajweed Rules Level (e.g., 'ghonna', 'tafkheem')
    ├─ Sifat (Properties) Level (e.g., 'hams', 'jahr')
    └─ Other linguistic levels
    ↓
Output: Predicted phonemes + linguistic properties
```

### Why Multi-Level?
- **Phonemes**: Basic sound units
- **Tajweed Rules**: Quranic recitation rules (madd, ghonna, etc.)
- **Sifat**: Detailed phonetic properties (voicing, emphasis, etc.)

**For Normal Arabic:** You'll need to adapt this to capture normal Arabic phonetic properties instead of Quranic tajweed rules.

### Key Classes
- `Wav2Vec2BertForMultilevelCTC`: Main model class
- `MultiLevelTokenizer`: Maps phonemes to token IDs
- `Sifa` dataclass: Stores linguistic properties

---

## 2. Phonetic System: Quranic vs. Normal Arabic

### Current System: Quranic Phonetic Script

**File:** `src/quran_muaalem/modeling/vocab.py`

The project uses a custom phonetic script that represents:
- **Diacritics** (tashkeel): fatha, damma, kasra, sukun
- **Prolongation marks**: For madd (elongation) rules
- **Special symbols**: For Quranic-specific phenomena

**Example:**
```
Standard text: "الله"
Quranic phonetic script: "ءَللَاهُ"
(includes diacritics representing exact pronunciation)
```

### Issues for Normal Arabic Adaptation

1. **Phoneme Inventory May Be Different**
   - Quranic pronunciation might have different phoneme rules
   - Normal Egyptian/Modern Standard Arabic has different phonetic inventory
   - Emphatic consonants might be realized differently

2. **Diacritics Not Needed**
   - Normal Arabic text often comes without diacritics
   - Will need to handle undiacritized text

3. **Tajweed Rules Don't Apply**
   - Madd rules are Quranic-specific
   - Normal speech doesn't follow these rules
   - Need to replace with general phonetic rules

### What You Need to Do
1. **Analyze your target Arabic dialect's phoneme inventory**
2. **Create a new vocabulary mapping** for normal Arabic phonemes
3. **Adapt the tokenizer** to handle your phoneme set
4. **Retrain the output layer** (CTC heads) with new vocabulary

---

## 3. Training Data Pipeline

### Current Pipeline

**Files:** 
- `src/quran_muaalem/inference.py` - Model inference
- `src/quran_muaalem/modeling/multi_level_tokenizer.py` - Tokenization

**Data Flow:**
```
Raw Audio (MP3, WAV, etc.)
    ↓
Load with librosa @ 16kHz sampling rate
    ↓
Audio Features (Wav2Vec2 processor)
    ↓
Pad/Truncate to max length
    ↓
Model Forward Pass
    ↓
CTC Decoding (greedy or beam search)
    ↓
Output Phonemes + Properties
```

### For Fine-Tuning on Normal Arabic

You need:
1. **Annotated dataset** of normal Arabic audio with:
   - Audio files (WAV/MP3 at 16kHz)
   - Ground truth phonetic transcriptions
   - Linguistic property labels (if you want multi-level output)

2. **Data format** should match:
   - Duration: Any length (handles dynamic padding)
   - Sampling rate: 16kHz (critical - built into model)
   - Format: WAV or MP3 (handled by librosa)

3. **Phonetic annotations** need to be:
   - In your custom phonetic script (similar to current one)
   - Aligned with audio (start/end times per phoneme)
   - Labeled with linguistic properties

### Size Estimates
- **Minimum**: 10 hours (very risky)
- **Recommended**: 100+ hours of speech
- **Ideal**: 500+ hours for production quality

---

## 4. Feature Extraction: Wav2Vec2-BERT

### What It Does

**File:** `src/quran_muaalem/engine/serve.py` (uses transformers AutoFeatureExtractor)

```python
processor = AutoFeatureExtractor.from_pretrained(model_name)
features = processor(audio, sampling_rate=16000, return_tensors="pt")
```

### How It Works
1. **Input**: Raw audio waveform
2. **Processing**: 
   - Normalization
   - STFT (Short-Time Fourier Transform)
   - Mel-scale filterbank
   - Log compression
3. **Output**: 768-dimensional feature vectors

### Important Notes
- **Already fine-tuned on speech**: Wav2Vec2-BERT is pre-trained on 53k hours of multilingual speech
- **Handles various accents**: Likely includes some Arabic
- **No modification needed** for normal Arabic (unless your Arabic is extremely different from what it's seen)

### Should You Retrain This?
- **No**, unless your audio is from a very specific domain (e.g., singing, whispering)
- The feature extractor is **language-agnostic** for similar languages
- Fine-tuning just the CTC heads should be sufficient

---

## 5. Model Inference Pipeline

### How It Works

**File:** `src/quran_muaalem/inference.py`

```python
from quran_muaalem import Muaalem

# Initialize
muaalem = Muaalem(device="cpu")  # or "cuda"

# Run inference
output = muaalem(
    [audio_array],                    # List of audio arrays
    [phonetic_reference],             # List of reference phonetic scripts
    sampling_rate=16000
)

# Results
output.phonemes.text                  # Predicted phonemes
output.sifat                          # Linguistic properties per phoneme
```

### For Normal Arabic
- Same inference pipeline works
- Just change the reference phonetic script to your dialect's script
- Output will be in your phoneme vocabulary (after retraining)

---

## 6. Tokenization: Phoneme to Token IDs

### Current System

**File:** `src/quran_muaalem/modeling/multi_level_tokenizer.py`

```python
tokenizer = MultiLevelTokenizer(model_name)

# Maps:
# "ء" → 42 (token ID)
# "ن" → 15
# etc.

# For each phoneme, also stores:
# - Phonetic properties (hams, jahr, etc.)
# - IDs for each property (0-100 per property)
```

### Structure
```python
vocab = {
    "phonemes": {
        0: "<pad>",
        1: "<unk>",
        2: "ء",
        3: "ب",
        ...
    },
    "hams_jahr": {
        0: "no_feature",
        1: "hams",
        2: "jahr"
    },
    # ... other properties
}
```

### For Normal Arabic Adaptation
1. **Create new vocab** from your phoneme inventory
2. **Define new linguistic properties** (e.g., voicing, place, manner)
3. **Build new tokenizer** with those definitions
4. **Update CTC output layer size** to match new vocab size

---

## 7. CTC Decoding

### How It Works

**File:** `src/quran_muaalem/engine/serve.py` (simple_ctc_decode function)

```python
def simple_ctc_decode(batch_arr):
    # Remove blanks (CTC special token)
    # Collapse consecutive repeats
    # Return phoneme IDs
```

**Example:**
```
CTC output: [0, 42, 42, 0, 15, 15, 0, ...]  (0 = blank)
After decoding: [42, 15]  (phoneme IDs)
After lookup: ["ء", "ن"]  (phonemes)
```

### For Fine-Tuning
- **No changes needed** - the decoding is generic
- Works with any phoneme vocabulary
- You can upgrade to beam search for better accuracy (currently uses greedy)

---

## 8. Key Files for Fine-Tuning

### Files You MUST Understand

| File | Purpose | Modification Needed? |
|------|---------|----------------------|
| `src/quran_muaalem/modeling/modeling_multi_level_ctc.py` | Neural network architecture | No (reuse as-is) |
| `src/quran_muaalem/modeling/vocab.py` | Phoneme vocabulary | **YES** - Create your own |
| `src/quran_muaalem/modeling/multi_level_tokenizer.py` | Phoneme ↔ ID mapping | **YES** - Adapt for your vocab |
| `src/quran_muaalem/inference.py` | Model inference wrapper | Minor changes (config only) |
| `src/quran_muaalem/engine/serve.py` | Inference server | Minor changes (device handling) |
| `src/quran_muaalem/app/main.py` | REST API endpoints | Optional changes |

### Files You Should Study

1. **For understanding training:**
   - Look at how Quranic data was processed
   - Check README.md for data preparation steps
   - Study `quran-transcript` library (external dependency)

2. **For understanding features:**
   - `src/quran_muaalem/explain.py` - How to interpret outputs
   - `src/quran_muaalem/explain_gradio.py` - Visualization

3. **For configuration:**
   - `pyproject.toml` - Dependencies and versions
   - `.env` - Runtime settings
   - `src/quran_muaalem/engine/settings.py` - Engine configuration

---

## 9. External Dependencies Critical for Fine-Tuning

### Must Understand

1. **`quran-transcript`** (external library)
   - Handles Quranic text to phonetic conversion
   - **For normal Arabic**: You'll need a different library
   - **Replacement options**:
     - Create custom conversion script
     - Use Arabic transliteration library
     - Manual phonetic annotation

2. **`transformers`** (Hugging Face)
   - Provides Wav2Vec2-BERT model
   - Used for feature extraction
   - You'll use this for training too

3. **`torch` / `pytorch`**
   - Deep learning framework
   - Used for training the CTC heads
   - You'll use this for fine-tuning

4. **`librosa`**
   - Audio loading and processing
   - Handles various audio formats
   - No changes needed

---

## 10. Fine-Tuning Strategy (High-Level)

### What You Need to Do

```
1. Data Preparation
   ├─ Collect normal Arabic audio
   ├─ Create phonetic annotations
   └─ Create vocabulary mapping
   
2. Model Adaptation
   ├─ Keep Wav2Vec2-BERT encoder (pre-trained)
   ├─ Freeze early layers (optional)
   └─ Retrain CTC heads with new vocabulary
   
3. Training
   ├─ Set up training loop
   ├─ Load pre-trained checkpoint
   └─ Fine-tune on your data
   
4. Evaluation
   ├─ Measure phoneme accuracy (CER)
   ├─ Evaluate linguistic property prediction
   └─ Test on new Arabic dialects
```

### Recommended Approach

1. **Start small**: Fine-tune on 10-50 hours first
2. **Keep encoder frozen**: Don't change Wav2Vec2-BERT
3. **Train only CTC heads**: Faster convergence
4. **Validate frequently**: Prevent overfitting
5. **Use same sampling rate**: 16kHz (critical!)

---

## 11. Potential Challenges & Solutions

### Challenge 1: Phoneme Inventory Mismatch
**Problem**: Quranic phonemes ≠ Normal Arabic phonemes
**Solution**: 
- Analyze your target dialect phonetically
- Create mapping from Quranic to normal phonemes
- Retrain vocabulary and CTC heads

### Challenge 2: Lack of Annotated Data
**Problem**: Hard to get audio + phonetic annotations
**Solution**:
- Use automatic annotation tools first (whisper-based)
- Manual review and correction
- Data augmentation (pitch shift, speed change)

### Challenge 3: Tajweed Rules Don't Apply
**Problem**: Output includes tajweed rules meant for Quran
**Solution**:
- Replace with linguistic properties relevant to normal speech
- Retrain the "sifat" (properties) output layers
- Or remove multi-level output entirely (phonemes only)

### Challenge 4: Model Overfitting on Limited Data
**Problem**: Pre-trained model overfits on small datasets
**Solution**:
- Use early stopping
- Data augmentation
- Fine-tune upper layers only (keep encoder frozen)
- Use regularization (dropout, weight decay)

---

## 12. Evaluation Metrics

### Primary Metric: Phoneme Error Rate (PER)

```
PER = (S + D + I) / N × 100%

Where:
- S = Substitutions (wrong phoneme)
- D = Deletions (missing phoneme)
- I = Insertions (extra phoneme)
- N = Total reference phonemes
```

### Secondary Metrics

1. **Character Error Rate (CER)**: Error rate per character
2. **Word Error Rate (WER)**: Error rate per word (if you align to words)
3. **Property Accuracy**: Accuracy of linguistic properties (tajweed → your properties)

---

## 13. Resources for Learning

### Understanding CTC
- Original CTC paper: Graves et al. (2006)
- Good tutorial: https://distill.pub/2017/ctc/

### Wav2Vec2-BERT
- Hugging Face docs: https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert
- Paper: "WavLM: Large-Scale Self-Supervised Pre-Training for Speech"

### Arabic NLP/Speech
- Farasa toolkit (text processing): https://farasa.qcri.org/
- Arabic speech recognition: Look at QCRI, AubCv datasets

### Fine-Tuning Transformers
- Hugging Face Course: https://huggingface.co/course
- Transfer learning tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

## 14. Summary: What to Focus On

### Must Do
1. ✅ Understand **multi-level CTC architecture** (src/quran_muaalem/modeling/)
2. ✅ Create **new vocabulary** for normal Arabic phonemes
3. ✅ Prepare **annotated audio dataset** (100+ hours recommended)
4. ✅ Adapt the **tokenizer** to your phonemes
5. ✅ Fine-tune **CTC output heads** (keep encoder frozen)

### Nice to Have
1. 📚 Replace `quran-transcript` with your phonetization library
2. 📊 Create evaluation benchmark for your dialect
3. 🚀 Optimize for inference speed on CPU
4. 📝 Add beam search decoding (better accuracy)

### Skip For Now
1. ❌ Modifying Wav2Vec2-BERT encoder (it's good as-is)
2. ❌ Training from scratch (pre-training is expensive)
3. ❌ Supporting multiple dialects (start with one)

---

## Questions to Ask Yourself

Before starting fine-tuning, answer these:

1. **What is your target dialect?** (Egyptian, Levantine, Gulf, MSA, etc.)
2. **How much annotated data can you get?** (hours of audio + transcriptions)
3. **What phonetic properties matter?** (emphasis, voicing, etc.)
4. **What's your quality target?** (90% PER? 95%?)
5. **Do you need linguistic properties or just phonemes?**
6. **Can you use pre-existing Arabic speech datasets?**

---

Good luck with your fine-tuning project! Start small, validate often, and iterate.
