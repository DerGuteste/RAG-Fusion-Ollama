# RAG Fusion
This package was highly inspired by https://arxiv.org/abs/2402.03367.
The aim of this package is to provide you with the tools you need to setup a working RAG - Fusion environment.

RAG or Retrival Argumented Generation aims to increase a chatbots accuracy through providing it with relevant information.
RAG - Fusion takes this a step further. Here we do not only look for information that may be relevant to the users prompt but we also generate extra prompts for which we also search.

## Setup
To use search_fusion you need to setup a local Ollama server. **[Download Ollama](https://ollama.com/download)**

After downloading Ollama download a model you whish to use
```
ollama pull (the model)
```
Recommended for prompt generation: `llama3`

Recommended for embedding: `mxbai-embed-large`

You will also need to install **[Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)**
with conda:
```
conda install -c pytorch faiss-cpu=1.9.0
```

Now install `search-fusion`
```
pip install search-fusion
```
Now you are all set to start.

## Usage
For a working demo of how to use search_fusion look at demo.py.

To start you need to embedd all the information you want to search through.
```python
from search_fusion import fusion

embedding_model = 'mxbai-embed-large'
articles = ["Content...", ...]

embeddings = fusion.embedd_list(articles, model=embedding_model)

```

Now you are ready to search for relevant articles.

First generate some extra prompts.
```python
from search_fusion import spice

model = 'llama3'
prompt_count = 5
prompt = "recent archievements"

prompts = spice.spice_up(model=model, prompt=prompt, count=prompt_count)
prompts.append(prompt)

```

Now you can search for the most relevant articles.
```python

search_res = fusion.search_for_querries(texts=titles, querries=prompts, m=results_per_prompt, embeddings=embeddings, model=embedding_model)
#returns a List[List[str]]. A list for each querry

```

Now you want to get the most relevant of these results. You can do this in two different ways. 
Experiment whith them and find the one that suits you the most.

```python
#1.
frequency_sorted = fusion.frequency_sort(search_results=search_res)

#2.
unique_results = fusion.individual_results(search_results=search_res)
similarity_matrix = fusion.similarity_matrix(texts=unique_results, model=embedding_model)

similarity_sorted = fusion.similarity_sort(individual_results=unique_results, similarities=similarity_matrix)

```

### frequency sort
The results are sorted based on the number of occurences and their ranks.
Each result gets a score assigned which is the log(results_per_prompt - position + 1) / log(results_per_prompt)

### similarity sort
All the results are compared against each other. The distances are added onto their scores and the one with the smallest score is chosen as the most relevant.
