# 🧠 NLP Final Project

### Authors
- Abdelhamid Ahmed Mahmoud Abdelmoneim  
- Noureldin Mohamed Abdelsalm Mohamed Hamedo  
- Sergio Rodrigo Fernandez Testa  
- Shehata, Ahmed Mohamed Elghamry  

---

## 🚀 Before You Start

1. **Clone the repository**

```bash
git clone [COMPLETE_URL]
```

2. **Install the requirements**

```bash
pip install -r requirements.txt
```

3. **Download the dataset**

```bash
gdown 1lnoaa6tE2gGDQEEz0DW2hvOnjIMK9oTo -O data/dataset.csv
```

4. **Download the model embeddings**

```bash
gdown 1EWw7GLrt_B0r8zPcubthMxNhJ37LRMjr -O data/embeddings.model
```

5. **Install the indexer locally**  
[Download Indexer](https://drive.google.com/drive/folders/1QZpsn5Y6phyHtXDJNIENOpkoQEZ63ebm?usp=sharing)

6. **Install the classifier locally**  
[Download Classifier](https://drive.google.com/drive/folders/1Ps5Rg36ivyrD2CCrLZiT-iuSPmisWXDQ?usp=sharing)

7. **Add a `.env` file** to the root directory.

---

## 📦 Dataset Overview: RecipeNLG

![cook](https://drive.google.com/uc?export=view&id=1HT2TdXwilP8ovSMFoyr4_n7yqo71Aj2W)

This project uses the **RecipeNLG** dataset — a large-scale cooking recipe dataset designed for natural language generation tasks.

### 📂 Dataset Structure

The dataset contains **2,231,142 recipes**, each with the following attributes:

- **title**: Name of the meal  
- **ingredients**: List of ingredients  
- **directions**: Step-by-step cooking instructions  
- **link**: URL to the original source  
- **source**: Indicates data origin  
  - "Gathered" (0): Web-scraped  
  - "Recipes1M" (1): From Recipe1M+ dataset  
- **NER**: Named food entities such as ingredients, cooking actions, and quantities

---

## 🎯 Dataset Purpose

The dataset is curated for **semi-structured text generation**, primarily recipe generation from structured inputs like ingredients and NER tags. It supports tasks in:

- Procedural text generation  
- Cooking assistants  
- Food-focused NLP applications  
- Language modeling and summarization

RecipeNLG enhances the original Recipe1M+ dataset by adding over 1 million cleaned, deduplicated recipes.

---

## 📊 Dataset Statistics

- **Total Recipes**: 2,231,142  
- **Average Ingredients**: ~8 (typically between 4–12)  
- **Average Directions**: ~5 steps  
- **Average Document Length**: 100–130 tokens  
- **Vocabulary Size (Corpus)**: 86,609 unique tokens  
- **Average Vocabulary per Document**: 33 tokens

### 📈 Distribution Insights

- **Length Distribution**: Follows a long-tail pattern — many short, simple recipes with a few long, complex ones  
- **Tokens**: ~278 million tokens in total  
- **Standard Deviation of Length**: ~64.92  
- **Document Structure**: Mix of structured (lists) and unstructured (text) fields

This makes it suitable for basic to advanced NLP tasks, including generation, classification, and semantic search.

---

## 📌 Use Cases

- Text-to-recipe generation  
- Recipe classification or retrieval  
- Ingredient-based search engines  
- Procedural text summarization  
- Named Entity Recognition in food domain

---

---

## 📌 What we Did

- Recipe classification or retrieval  
- Ingredient-based search engines  
- Named Entity Recognition in food domain

---

## 📎 Citation

If you use the dataset, please cite the original [RecipeNLG paper](https://arxiv.org/abs/2010.02404).

---
