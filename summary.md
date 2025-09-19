# Project Summary: Recipe Dataset NLP Pipeline

This report covers six Jupyter notebooks detailing the steps in processing, analyzing, and building models on the RecipeNLG dataset for various Natural Language Processing (NLP) tasks.

---

## Preliminary analysis

### Overview

Initial exploration of the RecipeNLG dataset.

### Key Points

- **Authors**: Project contributors are listed.
- **Data**: Downloads and extracts `receipeData.zip`, containing `full_dataset.csv`.
- **Dataset Structure**:
  - ~2.2M rows, 6 columns:
    - `title` (str)
    - `ingredients` (list of str)
    - `directions` (list of str)
    - `link` (str)
    - `source` (ClassLabel): "Gathered" or "Recipes1M"
    - `NER` (list of str): Named food entities
- **Purpose**: Supports semi-structured text generation for recipe modeling.
- **Statistics**:
  - 2,231,142 recipes.
  - Avg ~8 ingredients, ~5 directions.
  - Long-tail distribution of lengths.
- **Vocabulary**:
  - ~86,609 unique tokens.
  - Avg ~339 tokens per document.
- **Implementation**: Uses `pandas`, `re`, `ast`, and visualization with histograms.

---

## Word2vec RecipeGen

### Overview

Trains a Word2Vec model on recipe ingredients (NER).

### Key Points

- **Corpus**: Tokenizes `NER` column using `gensim.utils.simple_preprocess`.
- **Model Training**:
  - Parameters:
  - `vector_size=100`: This specifies the dimensionality of the word vectors. A higher value allows the model to capture more complex relationships between words but increases computational cost and memory usage. Typically, values between 100 and 300 are common.
  - `window=5`: This defines the maximum distance between the current word and the words in its context window. A larger window size considers a broader context, while a smaller window focuses on closer relationships.
  - `min_count=2`: This sets the minimum frequency threshold for words to be included in the vocabulary. Words that appear fewer times than this threshold will be ignored, which helps reduce noise from rare words.
  - `sg=1`: This indicates the training algorithm. A value of `1` means the Skip-Gram model is used, which predicts context words given a target word. A value of `0` would use the Continuous Bag of Words (CBOW) model, which predicts a target word based on its context.
  - `workers=4`: This specifies the number of worker threads to use for training. Increasing this value can speed up training on multi-core systems, but it may also increase memory usage.
  - Saved as `./data/embeddings.model`
- **Exploration**:
  - `most_similar("sugar")`
  - `similarity("milk", "cream")`
- **Visualization**:
  - Top 100 frequent words.
  - PCA → 3D plot with Plotly.
- **Semantic Similarity**:
  - `pork` vs `bacon`: 0.4290
  - `pork` vs `cake`: 0.1966

---

## Recipe Generator Indexer

### Overview

This section of the project focuses on creating a text search index for the recipe dataset to enable searching for specific keywords and ingredients. The tool chosen for this task is PyTerrier, which leverages the underlying Terrier information retrieval toolkit. The objective is not semantic search (finding similar meanings) but rather locating documents containing the specific components or keywords queried.

### Key Points

- **Data Setup**:
    - The process begins with downloading the dataset, which is a `.zip` file containing `full_dataset.csv`, using `gdown`.
    - The downloaded archive, `receipeData.zip`, is then unzipped to extract the `full_dataset.csv` file into a `dataset/` directory.
    - Necessary libraries are installed, including updating/reinstalling `numpy` and `gensim`, and installing `python-terrier==0.11.0`.
    - An instance of PyTerrier is initialized.

- **Index Preparation**:
    - The dataset is loaded into a pandas DataFrame from the `/content/dataset/full_dataset.csv` file.
    - A crucial step is adding a `docno` column to the DataFrame, which serves as a unique document identifier for indexing. This is created by converting the DataFrame's index to a string type.
    - The dataset is then converted into a list of Python dictionaries, which is the expected format for PyTerrier's indexer when processing pandas DataFrames.
    - The columns containing `ingredients` and `directions` are processed. Initially stored as string representations of lists (JSON strings), they are decoded using `json.loads` and then re-joined into single strings. This conversion is done to ensure the indexer processes the text content correctly.
    - Before indexing, the recipes are validated to ensure they contain the required fields: `docno`, `title`, `ingredients`, and `directions`, and that these fields are strings.
    - Out of 2,231,142 initial records, 2,231,141 are found to be valid. The validation process identifies one invalid recipe where the `title` field was `nan` (Not a Number) instead of a string. Recipes where the `title` is an empty string after stripping whitespace are also removed.

- **Indexing**:
    - An indexer instance, `pt.IterDictIndexer`, is created to build the index.
    - The index is configured to be written to a directory named `./recipe_index`.
    - The index is built by passing the list of validated recipe dictionaries to the indexer's `index()` method.
    - The fields indexed are `title`, `ingredients`, and `directions`.
    - Internally, PyTerrier uses an inverted index structure (via Terrier) to map terms to documents.

- **Index Statistics**:
    - After indexing, the collection statistics of the built index can be retrieved and printed.
    - These statistics provide insights into the index structure:
        - Number of documents: 2,231,141.
        - Number of terms: 165,126.
        - Number of postings: 129,767,038.
        - Number of fields: 3.
        - Number of tokens: 221,302,602.
        - Field names: `[title, ingredients, directions]`.
        - Positions are noted as `false`.

- **Search**:
    - A reference to the built index (`indexref`) is obtained, which allows for the creation of retrieval models.
    - For searching, a BM25 model is initialized using `pt.BatchRetrieve` with the loaded index. Note that `BatchRetrieve` is deprecated in the used version of PyTerrier.
    - An example search is demonstrated with the query `"eggs"`.
    - The search results show the top-k documents based on their relevance score calculated by the BM25 model for the given query. The output includes columns like `qid`, `docid`, `docno`, `rank`, `score`, and the query itself. The example shows the top 1000 results for the query `"eggs"`.

---

## Encodings

### Overview

Vectorizes ingredients with TF-IDF, clusters recipes, and prepares data for classification.

### Key Points

- **Vectorization**:
  - TF-IDF on `NER`, `max_features=20000`
- **Clustering**:
  - `KMeans` from k=2 to k=10
  - Evaluated with Logistic Regression → Accuracy & Macro F1
- **Best k**: 2
- **Preprocessing for BERT**:
  - Tokenize `NER` column
  - Encode cluster labels
  - Save encodings and labels as CSV + pickle

---

## BERT Categorizer

### Overview

This document details Part 5 of the Final Group Project NLP. The core task described is the training of a sequence classification model, specifically DistilBERT, to perform a binary categorization of recipes. The goal is to classify each recipe into one of two broad categories, identified in a previous step: "Main Dish" or "Dessert". This model is intended to provide a simple categorization based on recipe content.

### Key Points

- **Setup**:
    - The process begins by downloading a `.zip` file named `NLP_data.zip` using `gdown`. This file is expected to contain data from previous steps, as it is approximately 725MB.
    - The downloaded archive is then unzipped, revealing a directory structure including `content/drive/MyDrive/NLP data/recipe_classifier/` and various data files like `Clustered_dataset.csv`, `test_encodings.pkl`, `train_encodings.pkl`, `test_labels.csv`, and `train_labels.csv`.
    - Required libraries are imported, including `pandas`, `re`, `ast`, `transformers`, `torch`, and `joblib`.
    - A variable `data_folder` is set to `/content/content/drive/MyDrive/NLP data` to point to the location of the unzipped data.
    - The tokenizer is loaded from the specified data folder using `AutoTokenizer.from_pretrained`.
    - Training and testing encodings, which were likely generated in a previous tokenization step, are loaded using `joblib.load` from their `.pkl` files within the `data_folder`.
    - Training and testing labels are loaded from `.csv` files (`train_labels.csv`, `test_labels.csv`) using `pandas` and extracted as Python lists.

- **Dataset**:
    - A custom PyTorch `Dataset` class named `RecipeDataset` is defined.
    - This class is designed to wrap the loaded encodings and labels.
    - The `__init__` method stores the encodings and labels.
    - The `__getitem__` method is implemented to retrieve a single sample given an index (`idx`). It extracts the encoding tensors (like `input_ids` and `attention_mask`) and the label for that index.
    - The `__len__` method returns the total number of samples, which is the length of the labels list.
    - Instances of this `RecipeDataset` are created for both the training and test data.

- **Training**:
    - The model used is `DistilBertForSequenceClassification`, initialized with `distilbert-base-uncased` and configured for `num_labels=2`.
    - To reduce training time or resource usage, training is performed on a subset of the full dataset. A training subset of 30,000 samples (`N = 30_000`) is created by randomly sampling indices from the full training dataset without replacement.
    - Similarly, an evaluation subset of 10,000 samples (`N_eval = 10_000`) is created by randomly sampling indices from the test dataset.
    - Hugging Face `TrainingArguments` are configured for the training process. Key arguments include:
        - `output_dir`: `"./results"` (where checkpoints and final model will be saved).
        - `eval_strategy`: `"epoch"` (evaluation is performed at the end of each epoch).
        - `save_strategy`: `"epoch"` (model checkpoints are saved at the end of each epoch).
        - `per_device_train_batch_size`: 16.
        - `per_device_eval_batch_size`: 64.
        - `num_train_epochs`: 3.
        - `weight_decay`: 0.01.
        - `logging_dir`: `"./logs"`.
        - `logging_steps`: 10.
    - A Hugging Face `Trainer` instance is created, combining the model, training arguments, and the defined training and evaluation subsets.
    - The training process is initiated by calling `trainer.train()`.
    - After training, the trained model and tokenizer are saved to the `./results/recipe_classifier` directory.

- **Evaluation**:
    - For evaluation, a subset of 50,000 samples from the full test dataset is used. This is created by taking the first 50,000 indices.
    - Predictions are generated for this evaluation subset using `trainer.predict()`.
    - The `classification_report` from `sklearn.metrics` is used to evaluate the model's performance on this subset. The report compares the true labels (`small_test_labels`) with the predicted labels (`np.argmax(preds_output.predictions, axis=1)`).
    - The two target classes are referred to as 'class1' and 'class2' in the report output.
    - The classification report provides detailed metrics:
        - **Precision**: 1.00 for class1, 0.95 for class2.
        - **Recall**: 0.98 for class1, 0.99 for class2.
        - **F1-score**: 0.99 for class1, 0.97 for class2.
        - **Support**: 36,080 for class1, 13,920 for class2 (total 50,000).
        - **Overall Accuracy**: 0.98 (or 98%). The report also includes macro and weighted averages for precision, recall, and F1-score, all around 0.98-0.99.

## Agent

### Overview

This document describes Part 6 of the Final Group Project NLP, which culminates in the creation of an interactive Langchain agent. The purpose of this agent is to integrate the various components developed in the previous steps – including the PyTerrier recipe index, the Word2Vec ingredient similarity model, and the BERT recipe categorizer – into a unified system capable of responding to natural language queries about recipes. The agent acts as an orchestrator, selecting and using the appropriate tools (which wrap the previously built models and data sources) to fulfill user requests.

### Key Points

- **Setup**:
    - The setup process involves acquiring the necessary data and models and configuring the runtime environment.
        - **Downloading**:
            - The base dataset (`receipeData.zip`) containing the recipe information.
            - The pre-trained Word2Vec model file (`embeddings.model`), which was saved in a previous step.
            - The BERT recipe categorizer model and tokenizer (presumably from the directory saved in step 5), downloaded as a folder.
            - The PyTerrier index directory (`./recipe_index`), which was built in step 3.
        - **Environment Configuration**:
            - Installation of various Python libraries including `langchain`, `langchain_openai`, `langchain_community`, `python-dotenv`, `pymongo`, `sqlalchemy`, `rank-bm25`, `nltk`, `gensim` (specific version 4.3.3 or updated), `numpy` (specific version 1.26.0 or updated), `scipy`, `pyterrier` (version 0.11.0).
            - Configuration for database access: MongoDB (`mongo_db`) and a PostgreSQL database are set up, with connection URIs loaded from environment variables or Google Colab user data. A `MongoClient` is initialized for MongoDB. A SQLAlchemy engine and sessionmaker are configured for the PostgreSQL database.
            - Initialization of PyTerrier. A check is made to ensure PyTerrier is started.
            - Loading the BERT model (`AutoModelForSequenceClassification`) and tokenizer (`AutoTokenizer`) from the downloaded categorizer folder. A `Trainer` instance is also initialized with the model.
            - Loading the PyTerrier index using `pt.IndexFactory.of()` pointing to the index data properties file.
            - Loading the Word2Vec embedding model from its saved path.

- **Functions**:
    - Several Python functions are defined to perform specific tasks that will be used by the agent's tools.
        - `init_pyterrier()`: Ensures PyTerrier is initialized.
        - `create_index()`: Provides a reference to the loaded PyTerrier index, creating the directory if it doesn't exist.
        - `query_index_topk(query, k=5)`: Takes a query and an optional number `k`, initializes/loads the PyTerrier index, uses a BM25 model for retrieval, searches the index with the given query, and returns the `docids` of the top `k` results as a list.
        - `check_ingredient_compatibility(recipe_ingredients, candidate_ingredient, similarity_threshold=0.5)`: Checks if a `candidate_ingredient` is compatible with a list of `recipe_ingredients` based on Word2Vec similarity. It loads the Word2Vec model, calculates the cosine similarity between the average vector of the recipe ingredients (that are in the model's vocabulary) and the candidate ingredient's vector. It returns a dictionary indicating the similarity score, whether it's compatible based on the threshold, and any words not found in the model's vocabulary.
        - `get_receipe_id(query, k=5)`: Calls `query_index_topk` to get recipe IDs based on a query.
        - `get_titles_from_docids(docids)`: Retrieves the titles for a given list of document IDs (`docids`) by querying the MongoDB collection (`mongo_recipes`).
        - `get_recommendation(query, k=5)`: Combines the search and title retrieval steps by calling `get_receipe_id` and then `get_titles_from_docids`.
        - `get_directions(titles)`: Retrieves the directions for a given list of recipe titles from MongoDB.
        - `get_ingred(titles)`: Retrieves the ingredients list for a given list of recipe titles from MongoDB.

- **Tools**:
    - These functions are wrapped into Langchain Tool objects to make them accessible to the agent.
        - `search_recipes_tool`: This tool wraps `get_recipe_ids` (which calls `query_index_topk`) and `get_recipes_from_mongo`. It takes a query string and an optional `k` and returns a list of recipe dictionaries (including `docno`, `title`, `ingredients`, and `directions`) from MongoDB for the top search results.
        - `get_similar_ingredients_tool`: This tool wraps `check_ingredient_compatibility`. It takes a dictionary containing `recipe_ingredients` (list of strings), `candidate_ingredient` (string), and an optional `similarity_threshold`. It uses the loaded Word2Vec model to determine ingredient compatibility.
        - `categorize_recipe_tool`: This tool wraps a function `predict_labels` which uses the loaded BERT model and tokenizer to classify recipe ingredient text. It takes `ingredient_text` as input and returns the predicted category label (e.g., "Main Dish", "Dessert").

- **Agent**:
    - An instance of a Langchain agent is created to use the defined tools.
        - The agent uses a ChatOpenAI language model with `model="gpt-4"` and `temperature=0.3`.
        - The agent type specified is `AgentType.ZERO_SHOT_REACT_DESCRIPTION`, indicating it uses a zero-shot approach based on the tool descriptions and a ReAct (Reasoning and Acting) framework to decide which tool to use and in what sequence.
        - The agent is initialized with the list of available tools and the language model. Verbose mode is enabled (`verbose=True`) to show the agent's thought process and tool usage.

- **Simulation**:
    - An example query is used to demonstrate the agent's capabilities.
        - The user provides the prompt: "Give me 3 recipes with chicken and rice, and tell me if chicken is similar to turkey and categorize each recipe".
        - The agent begins its execution chain.
        - Based on the request for recipes, the agent decides its first action is to use the `search_recipes_tool` with the query "chicken and rice" and `k=3`.
        - The `search_recipes_tool` executes, using the PyTerrier index to find relevant recipe IDs, and then fetches the recipe details (`title`, `ingredients`, `directions`, `docno`) from MongoDB for these IDs.
        - The agent receives the search results as an observation. It notes that the first few results don't seem to match exactly but identifies one recipe ('Chicken Rice Casserole') as containing chicken and rice.
        - Next, the agent determines it needs to check the similarity between "chicken" and "turkey" using the `get_similar_ingredients_tool`.
        - It calls the tool with the appropriate input structure. The tool execution proceeds, and the agent interprets the result.
        - Finally, the agent decides to categorize the recipe that contains chicken and rice using the `categorize_recipe_tool`.
        - It provides the ingredient text to the tool, which tokenizes the input, uses the BERT model to predict the class, and returns "Main Dish".
        - The agent synthesizes the information from the tool outputs and provides the final answer: "The recipe 'Chicken Rice Casserole' is a main dish and contains chicken and rice. However, chicken is not similar to turkey."
        - The agent chain finishes execution.
