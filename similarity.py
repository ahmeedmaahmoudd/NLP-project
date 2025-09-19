from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cosine


def check_ingredient_compatibility(model_path, recipe_ingredients,
                                   candidate_ingredient,
                                   similarity_threshold=0.5):
    """
    Check if a candidate ingredient is compatible with a given recipe based on
        Word2Vec similarity.

    Parameters:
        model_path (str): Path to the trained Word2Vec model.
        recipe_ingredients (list of str): List of ingredient names.
        candidate_ingredient (str): The new ingredient to test.
        similarity_threshold (float): Threshold above which an ingredient is
        considered compatible.

    Returns:
        dict: {
            'similarity': float,
            'compatible': bool,
            'missing_words': list of str
        }
    """
    # Load the Word2Vec model
    model = Word2Vec.load(model_path)

    # Track words not in the model
    missing_words = [word for word in recipe_ingredients +
                     [candidate_ingredient] if word not in model.wv]

    # Filter valid ingredients in the model
    valid_recipe_words = [
        word for word in recipe_ingredients if word in model.wv]
    if not valid_recipe_words or candidate_ingredient not in model.wv:
        return {
            'similarity': None,
            'compatible': False,
            'missing_words': missing_words
        }

    # Compute recipe embedding as average of ingredient vectors
    recipe_vector = np.mean([model.wv[word]
                            for word in valid_recipe_words], axis=0)

    # Get candidate vector
    candidate_vector = model.wv[candidate_ingredient]

    # Compute cosine similarity
    similarity = 1 - cosine(recipe_vector, candidate_vector)

    return similarity >= similarity_threshold
