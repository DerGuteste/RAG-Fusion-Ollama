import faiss
import ollama
import requests
import numpy as np
import math
from typing import List

from tqdm import tqdm


def find_embedding_depth(model: str) -> int:
    """
    Used to find the embedding depth of any model
    
    Parameters:
    model (str): The model you wish to use.
    
    Returns:
    int: the depth of the embedding
    """


    resTest = requests.post('http://localhost:11434/api/embeddings', 
                        json={
                            'model': model,
                            'prompt': "t" 
                        })

    embTest = resTest.json()['embedding']

    return len(embTest)

def embedd_list(texts: List[str], model='llama3') -> np.ndarray:
    """
    This function returns the embedding vectors for the provided Strings
    
    Parameters:
    texts (List[str]): A list of strings for which the embedding will be generated.
    model (str): The model you wish to use.
    
    Returns:
    np.ndarray: The embedding vectors are a numpy ndarray.
    """

    d = 4096

    if model != "llama3":
        d = find_embedding_depth(model)

    X = np.zeros((len(texts), d), dtype='float32')

    progress_bar = tqdm(total=len(texts), desc='Embedding Strings')

    for i, text in enumerate(texts):
        res = requests.post('http://localhost:11434/api/embeddings', 
                        json={
                            'model': model,
                            'prompt': text 
                        })
        
        embedding = res.json()['embedding']
        X[i] = np.array(embedding)

        progress_bar.update(1)

    progress_bar.close()

    return X

def embedd(prompt: str, model="llama3") -> List[float]:
    """
    This funciton embedds a single string.

    Parameters:
    prompt (str): the string you whish to embedd
    model (str): the model you whish to use

    Returns:
    List[float]: the embedding vector for your string
    """


    res = requests.post('http://localhost:11434/api/embeddings', 
                        json={
                            'model': model,
                            'prompt': prompt 
                        })
        
    return res.json()['embedding']




def search_for_querries(texts: List[str], querries: List[str], m: int, text_embeddings: np.ndarray = np.empty, model='llama3') -> List[List[str]]:
    """
    Finds the most similar strings to the provided querries. Employs faiss's similarity search.

    Parameters:
    texts (List[str]): the texts to search through
    querries (List[str]): the querries to search for
    m (int): number of search results to return
    text_embeddings (np.ndarray): provide these if already computed this will skip computing them again
    model (str): the model you whish to use for embedding. This need to be the same as the one used to compute the text_embeddings

    Returns:
    List[List[str]]: a list of lists of string whith the dimensions of (len(querries), m). If you want the second result for the first querry index to res[0][1]
    """

    if len(text_embeddings) < 1:
        embedded_texts = embedd_list(texts, model=model)
    else:
        embedded_texts = text_embeddings

    d = len(embedded_texts[0])
    index = faiss.IndexFlatL2(d)
    index.add(embedded_texts)

    print(index)

    output = []

    for querry in querries:
        embedding = np.array([embedd(querry, model=model)], dtype='float32')
        D, I = index.search(embedding, m)

        arr = np.array(texts)[I.flatten()]  

        output.append(arr)


    return output


def similarity_matrix(texts: List[str], model: str = 'llama3') -> tuple[np.ndarray, np.ndarray]:
    """
    computes the similarities between the provided texts

    Parameters:
    texts (List[str]): 
    querries: (List[str):
    m (int):
    text_embeddings ():
    """


    X = embedd_list(texts)

    d = len(X[0])

    index = faiss.IndexFlatL2(d)
    index.add(X)

    return index.search(X, k=len(texts))


def frequency_sort(search_results: List[List[str]]) -> List[str]:
    results = {}

    for result in search_results:

        for j, r in enumerate(result):

            if results.__contains__(r):
                results[r] += math.log(len(result) - j + 1) / math.log(len(result))
            else:
                results[r] = math.log(len(result) - j + 1) / math.log(len(result))
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])[::-1]
    results_list = [key for key, value in sorted_results]

    return results_list


def individual_results(search_results: List[List[str]]) -> List[str]:

    individuals = []

    for result in search_results:
        for text in result:
            if not individuals.__contains__(text):
                individuals.append(text)

    return individuals


def similarity_sort(individual_results: List[str], similarities: tuple[np.ndarray, np.ndarray]) -> List[str]:
    similarity_scores, similiarity_indices = similarities

    similarity_map = {}

    for i, text in enumerate(individual_results):
        for j, score in enumerate(similarity_scores[i]):

            if(similarity_map.__contains__(individual_results[similiarity_indices[i][j]])):
                similarity_map[individual_results[similiarity_indices[i][j]]] += score
            else:
                similarity_map[individual_results[similiarity_indices[i][j]]] = score


    new_sort = sorted(similarity_map.items(), key=lambda x: x[1])
    similiarity_list = [key for key, value in new_sort]

    return similiarity_list
