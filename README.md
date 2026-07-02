# Ghost in the Machine

A multi-tiered AI text detection system designed to distinguish between authentic human authors (Charles Dickens, Jane Austen) and AI language models attempting to mimic their literary style.

## What It Is

The project investigates whether AI text generation leaves a detectable "mathematical fingerprint" regardless of the stylistic guise it adopts. It classifies text paragraphs into three categories:
1. **Human**: Authentic excerpts from classic literature (Project Gutenberg).
2. **AI Generic**: Standard AI-generated text.
3. **AI Impostor**: AI-generated text explicitly prompted to mimic a specific author.

## How It Works

The system evaluates text through three distinct detection tiers:
1. **Tier A (Stylometric RF)**: A Random Forest classifier using 37 handcrafted stylometric features (Type-Token Ratio, syntax depth, readability scores, punctuation frequencies, and function word ratios).
2. **Tier B (Semantic MLP)**: A multi-layer perceptron trained on dense sentence embeddings (`all-MiniLM-L6-v2`) to capture deep semantic structures.
3. **Tier C (Transformer)**: A DistilBERT model fine-tuned using Low-Rank Adaptation (LoRA) for direct sequence classification.

Additionally, the project employs **SHAP** (global feature importance) and **Captum Layer Integrated Gradients** (token-level saliency) for explainability, mapping out exactly which features and words trigger AI detection.

## How To Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Dataset Generation** (Requires `GEMINI_API_KEY`):
   ```bash
   python3 generate_more_data.py
   python3 reassemble_and_run.py
   ```
3. **Run the Pipeline**:
   ```bash
   python3 -m src.task1_stylometry      # Feature extraction & statistical plots
   python3 -m src.task2_tier_a          # Train Tier A
   python3 -m src.task2_tier_b          # Train Tier B
   python3 -m src.task2_tier_c          # Train Tier C
   python3 -m src.task3_explainability  # Run SHAP & Captum
   python3 run_task4_local.py           # Run Genetic Algorithm Attack
   python3 run_experiments.py           # Run Cross-Author & Ablation tests
   ```

## Results

### Detection Accuracy
* **Tier A (Stylometric RF)**: 97.0% Accuracy, 0.999 AUC
* **Tier B (Semantic MLP)**: 95.0% Accuracy, 0.995 AUC
* **Tier C (DistilBERT LoRA)**: 93.0% Accuracy, 0.997 AUC

### Experiment 1: Cross-Author Transfer
Testing if the model detects universal "AI-ness" rather than just memorizing an author's style.
* Train on Dickens -> Test on Austen: **83.8% Accuracy**
* Train on Austen -> Test on Dickens: **95.4% Accuracy**

### Experiment 2: Feature Ablation (Tier A)
Measuring the isolated impact of stylometric feature groups.
* Removing Lexical Richness: No drop (96.9%)
* Removing Function Words: Significant drop to 93.0%
* Training on **ONLY Function Words**: 92.2% Accuracy
* Training on **ONLY Lexical Richness**: 62.5% Accuracy

### The Turing Test (Genetic Algorithm Attack)
A genetic algorithm iteratively mutated AI-generated text to bypass the Tier C detector.
* Generation 0: 17.2% probability of being human.
* Generation 7: 90.3% probability of being human (Detector successfully bypassed).
