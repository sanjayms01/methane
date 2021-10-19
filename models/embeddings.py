# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (TensorFlow 2.3 Python 3.7 CPU Optimized)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/tensorflow-2.3-cpu-py37-ubuntu18.04-v1
# ---

# # SCROLL TO BOTTOM FOR GEOGRAPHIC EMBEDDINGS CODE
# ## REST OF NOTEBOOK COPIED FROM W266

# # Clusters and Distributions
#
# We'll work through how to build and factorize a co-occurrence matrix, and do some simple visualization of the embeddings.
#
# On Assignment 3 and 4, we'll dig a bit deeper into the properties of these embeddings, and experiment with them on a classification task.
#
# **Note:** If viewing on GitHub, please use this NBViewer link for proper rendering: http://nbviewer.jupyter.org/github/datasci-w266/2021-spring-main/blob/master/materials/embeddings/embeddings.ipynb

# +
# # !pip install tensorflow
# # !pip install nltk

# + code_folding=[]
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

# Standard python helper libraries.
import os, sys, time, shutil
import itertools, collections
from IPython.display import display

# NumPy and SciPy for matrix ops
import numpy as np
import scipy.sparse

# NLTK for NLP utils
import nltk

# Helper libraries.
from w266_common import utils, vocabulary, tf_embed_viz

# Bokeh for plotting.
utils.require_package("bokeh")
# import bokeh.plotting as bp
# from bokeh.models import LabelSet, HoverTool, WheelZoomTool
# bp.output_notebook()
# -

# As before, we'll use the Brown corpus as our dataset, and do our usual simple preprocessing. Since we're just going to explore the embeddings, we don't need a train/dev/test split.

# +
assert(nltk.download('brown'))  # make sure we have the data
corpus = nltk.corpus.brown
vocab = vocabulary.Vocabulary(utils.canonicalize_word(w) for w in utils.flatten(corpus.sents()))
print("Vocabulary: {:,} words".format(vocab.size))

tokens = utils.preprocess_sentences(corpus.sents(), vocab, use_eos=False, emit_ids=False)
print("Corpus: {:,} tokens (counting <s>)".format(len(tokens)))
# -

# # The Co-occurrence Matrix
#
# The base for our word embeddings will be a co-occurrence matrix $M$. In the most general form, we'll consider this to be a **word-context matrix**, where the row indices $i$ correspond to words (types) $w_i$ in the vocabulary. Context could be:
#
# - Documents
# - Paragraphs or sentences
# - Syntactic contexts
# - Topics
# - Nearby words
#
# We're really interested in the words, so we're going to jump right to the last one. How do we define "nearby"? The simplest way is to just position: we'll define a *window* and say that two words co-occur if they appear in this window. For example:
# ```
# the quick brown fox jumped over the lazy dog
# ```
# With a window of $\pm 2$ words, we say that `brown`, `fox`, `over`, and `the` are in the context of `jumped`, and so in our co-occurence matrix $C \in M^{|V|\times|V|}$ we have $C_{\mathtt{brown,jumped}} = 1$, $C_{\mathtt{fox,jumped}} = 1$, and so on.
#
#
#
#
#

# _**Note:**_ It turns out that we can transform any word-context matrix $M$ into a word-word matrix:
#
# Let 
# $$ M_{i\ell} = \mathbf{Count}[w_i \in \text{context}\ \ell] $$ 
#
# Then for $i \ne j$:
#
# $$ (MM^T)_{ij} = \sum_{\ell} M_{i\ell} M_{j\ell} = \mathbf{Count}[w_i \text{ in same context as } w_j] = C_{ij} $$
#
# There's a correction we'd need to do for the diagonal, but it won't change the structure of the representations that we get via the SVD. So regardless of the underlying context type, it's common to just deal with a word-word cooccurence matrix $C_{ij}$.

# ## Constructing the Co-occurrence Matrix
#
# In order to put our words in a matrix, we need to assign each one to a row index. Fortunately, our `Vocabulary` class does this automatically:

token_ids = vocab.words_to_ids(tokens)
print("Sample words: " + str(tokens[:5]))
print("Sample ids:   " + str(token_ids[:5]))

# Our co-occurence counts are pairwise between words, so we'll want to have a sparse representation. The total number of matrix elements is:

V = vocab.size
print("Total matrix elements: {:,} x {:,} = {:,}".format(V, V, V**2))


# But as with bigrams, most of these will be zero. So, we'll define $C$ as a `scipy.sparse` matrix. Like the sparse dicts we used in the [language modeling demo](../../materials/simple_lm/lm1.py), the sparse matrix will only store the nonzero elements we need.
#
# _**Mathematical note:**_  
# We can compute each element by sliding a window over each position $\ell$ in the corpus. Suppose our window is size $W = 2K + 1$. Then:
#
# $$ C_{ij} = \sum_\ell^{|\text{tokens}|} \sum_{k \in [-K,K],\ \delta \ne 0 } \mathbf{1}[w_\ell = i \text{ and } w_{\ell+k} = j] $$
#
# We'll hack this a little bit and change the order of the sum, which makes for simpler code:
#
# $$ C_{ij} = \sum_{k \in [-K,K],\ k \ne 0 } \sum_\ell^{|\text{tokens}|} \mathbf{1}[w_\ell = i \text{ and } w_{\ell+k} = j] $$
#
# Conveniently, the above is symmetric, so we'll simplify further to:
#
# $$ C_{ij}^+ = \sum_{k = 1}^K \sum_\ell^{|\text{tokens}|} \mathbf{1}[w_\ell = i \text{ and } w_{\ell+k} = j] = \sum_{k = 1}^K C_{ij}^+(k)$$
#
# $$ C_{ij}^- = \sum_{k = -K}^1 \sum_\ell^{|\text{tokens}|} \mathbf{1}[w_\ell = i \text{ and } w_{\ell+k} = j] = \sum_{k = -K}^1 C_{ij}^-(k)$$
#
# It's easy to see that $C_{ij} = C_{ij}^+ + C_{ij}^-$, and since $C_{ij}^+ = C_{ji}^-$, $C$ is a symmetric matrix.
#
# Now we can write the formula in code, where our outer loop sums over $k$:

def cooccurrence_matrix(token_ids, V, K=2):
    # We'll use this as an "accumulator" matrix
    C = scipy.sparse.csc_matrix((V,V), dtype=np.float32)

    for k in range(1, K+1):
        print(u"Counting pairs (i, i \u00B1 {:d}) ...".format(k))
        i = token_ids[:-k]  # current word
        j = token_ids[k:]   # k words ahead
        data = (np.ones_like(i), (i,j))  # values, indices
        Ck_plus = scipy.sparse.csc_matrix(data, shape=C.shape, dtype=np.float32)
        Ck_minus = Ck_plus.T  # Consider k words behind
        C += Ck_plus + Ck_minus

    print("Co-occurrence matrix: {:,} words x {:,} words".format(*C.shape))
    print("  {:.02g} nonzero elements".format(C.nnz))
    return C


# Let's look at a toy corpus to see how this works. With a window of 1, we should see co-occurrence counts for each pair of neighboring words:  
# `(<s>, nlp)`,  
# `(nlp, class)`,  
# `(class, is)`,  
# and so on - as well as their reversed versions (remember, C is symmetric!)

# + code_folding=[]
# Show co-occurrence on a toy corpus
toy_corpus = [
    "nlp class is awesome",
    "nlp class is fun"
]

toy_tokens = list(utils.flatten(s.split() for s in toy_corpus))
toy_vocab = vocabulary.Vocabulary(toy_tokens)
# sentence_to_ids adds "<s>" and "</s>"
toy_token_ids = list(utils.flatten(toy_vocab.sentence_to_ids(s.split()) 
                     for s in toy_corpus))

# Here's the important part
toy_C = cooccurrence_matrix(toy_token_ids, toy_vocab.size, K=1)

toy_labels = toy_vocab.ordered_words()
utils.pretty_print_matrix(toy_C.toarray(), rows=toy_labels, cols=toy_labels, dtype=int)
# -

toy_C


# ## Computing Word Vectors
#
# In order to go from our co-occurrence matrix to word vectors, we need to do two things:
#
# - First, convert to **PPMI** to reduce the impact of common words.
# - Compute the **SVD**, and extract our vectors.
#
# ### PPMI
#
# PPMI stands for Positive [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information), which you've seen on [Assignment 1](../../assignment/a1/information_theory.ipynb#Pointwise-Mutual-Information). PMI is a generalization of the idea of correlation, but for arbitrary variables. Here, we're interested in the correlation between word $i$ and word $j$, where we take the samples to be all the word-word pairs in our corpus.  
# Positive just means we'll truncate at zero: $\text{PPMI}(i,j) = \max(0, \text{PMI}(i,j))$
#
# We'll apply PPMI as a transformation of our counts matrix. First, compute probabilities:
# $$ P(i,j) = \frac{C(i,j)}{\sum_{k,l} C(k,l)} = \frac{C_{ij}}{Z}$$
# $$ P(i) = \frac{\sum_{k} C(i,k)}{\sum_{k,l} C(k,l)} = \frac{Z_i}{Z}$$
#
# Then compute PMI:
# $$ \text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)} = \log \frac{C_{ij} \cdot Z}{Z_i \cdot Z_j} $$
#
# Then truncate to ignore negatively-correlated pairs:
# $$\text{PPMI}(i,j) = \max(0, \text{PMI}(i,j))$$
#
# #### Note on Sparse Matricies
#
# In order to compute PPMI, we'll need to "unpack" the nonzero elements. Recall when we were constructing it, we constructed a list of `(values, (indices))`:
# ```
# data = (np.ones_like(i), (i,j))  # values, indices
# ```
# We'll do the inverse of this here, then transform all the values in parallel, then pack them back into a sparse matrix.

# + code_folding=[]
def PPMI(C):
    """Tranform a counts matrix to PPMI.
    
    Args:
      C: scipy.sparse.csc_matrix of counts C_ij
    
    Returns:
      (scipy.sparse.csc_matrix) PPMI(C) as defined above
    """
    Z = float(C.sum())  # total counts
    # sum each column (along rows)
    Zc = np.array(C.sum(axis=0), dtype=np.float64).flatten()
    # sum each row (along columns)
    Zr = np.array(C.sum(axis=1), dtype=np.float64).flatten()
    
    # Get indices of relevant elements
    ii, jj = C.nonzero()  # row, column indices
    Cij = np.array(C[ii,jj], dtype=np.float64).flatten()
    
    ##
    # PMI equation
    pmi = np.log(Cij * Z / (Zr[ii] * Zc[jj]))
    ##
    # Truncate to positive only
    ppmi = np.maximum(0, pmi)  # take positive only
    
    # Re-format as sparse matrix
    ret = scipy.sparse.csc_matrix((ppmi, (ii,jj)), shape=C.shape,
                                  dtype=np.float64)
    ret.eliminate_zeros()  # remove zeros
    return ret


# -

# Let's see what this does on our toy corpus:

utils.pretty_print_matrix(PPMI(toy_C).toarray(), rows=toy_labels, 
                          cols=toy_labels, dtype=float)

# ### The SVD
#
# Recall from async that the [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) decomposes an $m \times n$ matrix $X$ as:
#
# $$ X = UDV^T $$ 
#
# where $U$ is $m\times m$, $D$ is $m \times n$, and $V$ is $n \times n$, $U$ and $V$ are orthonormal matricies, and $D$ is diagonal. 
#
# Conventionally, we take the diagonal elements of $D$ to be in order, so $D_{00}$ is the largest singular value, and so on. Then we can take the first $d$ columns of $U$ to be our word vector representations.
#
# This is a very standard algorithm with many implementations. We'll use the one in [`sklearn.decomposition.TruncatedSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html), which will only compute the $d \ll |V|$ components we need.
#
# #### Note: known Anaconda bug
#
# There's a [known bug](https://github.com/BVLC/caffe/issues/3884) with Anaconda's configuration of some linear algebra libraries. If your Python kernel crashes on running the SVD, open a terminal and run:
# ```
# conda install mkl
# ```
# That should re-link the packages. You may need to restart your kernel for it to take effect.

from sklearn.decomposition import TruncatedSVD
def SVD(X, d=100):
    """Returns word vectors from SVD.
    
    Args:
      X: m x n matrix
      d: word vector dimension
      
    Returns:
      Wv : m x d matrix, each row is a word vector.
    """
    transformer = TruncatedSVD(n_components=d, random_state=1)
    Wv = transformer.fit_transform(X)
    # Normalize to unit length
    Wv = Wv / np.linalg.norm(Wv, axis=1).reshape([-1,1])
    return Wv, transformer.explained_variance_


# Again, applied to our toy corpus. Note that "fun" and "awesome" appear in identical contexts, so they get identical vector representations:

d = 3
utils.pretty_print_matrix(SVD(PPMI(toy_C).toarray(), d=d)[0], 
                          rows=toy_labels, cols=range(d), dtype=float)

# Now we can compute our word vectors on our whole corpus:

K = 1
d = 25
t0 = time.time()
C = cooccurrence_matrix(token_ids, vocab.size, K=K)
print("Computed Co-occurrence matrix in {:s}".format(utils.pretty_timedelta(since=t0))); t0 = time.time()
C_ppmi = PPMI(C)
print("Computed PPMI in {:s}".format(utils.pretty_timedelta(since=t0))); t0 = time.time()
Wv, _ = SVD(C_ppmi, d=d)
print("Computed SVD in {:s}".format(utils.pretty_timedelta(since=t0)))

# # Visualization
#
# For a quick visualization, we can plot the first two dimensions directly. Plotly makes this quite easy, and gives us free hovertext:

# +
n = 1000

hover = HoverTool(tooltips=[("word", "@desc")])
wztool = WheelZoomTool()
fig = bp.figure(plot_width=600, plot_height=600, tools=[hover, wztool, 'pan', 'reset'])
fig.toolbar.active_scroll = wztool
df = bp.ColumnDataSource(dict(x=Wv[:n,0], y=Wv[:n,1], desc=vocab.ids_to_words(range(n))))
fig.circle('x', 'y', source=df)
fig.add_layout(LabelSet(x='x', y='y', text='desc', source=df,
                        x_offset=2, y_offset=2))
bp.show(fig)
# -

# Unfortunately, this plot is quite limited. Pick a point and look at the words nearby - do they look related, either syntactically or semantically?
#
# Plotting two dimensions directly like this is equivalent to just doing the truncated SVD with $d=2$, which throws away quite a lot of information.
#
# ## t-SNE
#
# To get a better sense of our embedding structure, we can use [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) instead. This is a *non*-linear way of embedding high-dimensional data (like our embedding vectors) into a low dimensional space. It works by preserving local distances (like nearby neighbors), at the expense of some global distortion.
#
# The result is no longer a projection, but because it preserves locality  t-SNE is a very useful took to look at **clusters**.
#
#
# *Note: there's also a demo at http://projector.tensorflow.org/, pre-loaded with word2vec vectors.*



#

# ## (optional) Running t-SNE in-notebook
#
# We recommend using the TensorFlow projector, but you can also run t-SNE directly in the notebook, then plot the points with Plotly or another plotting library.
#
# Scikit-learn includes a t-SNE implementation in [`sklearn.manifold.TSNE`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), but the implementation is slow and tends to crash by using too much (>4 GB) memory.
#
# Instead, we'll use the excellent [`bhtsne`](https://github.com/dominiek/python-bhtsne) package. Install with:
# ```
# sudo apt-get install gcc g++
# pip install bhtsne
# ```
#
# The cell below will take around 2-3 minutes to run on a 2 CPU Cloud Compute instance.

# +
import bhtsne

n = 5000  # t-SNE is very slow, so restrict vocab size

t0 = time.time()
print("Running Barnes-Hut t-SNE on word vectors; matrix shape = {:s}".format(str(Wv.shape)))
Wv2 = bhtsne.tsne(Wv[:n])
print("Transformed in {:s}".format(utils.pretty_timedelta(since=t0)))

# +
n = 1000

hover = HoverTool(tooltips=[("word", "@desc")])
wztool = WheelZoomTool()
fig = bp.figure(plot_width=600, plot_height=600, tools=[hover, wztool, 'pan', 'reset'])
fig.toolbar.active_scroll = wztool
df = bp.ColumnDataSource(dict(x=Wv2[:n,0], y=Wv2[:n,1], desc=vocab.ids_to_words(range(n))))
fig.circle('x', 'y', source=df)
fig.add_layout(LabelSet(x='x', y='y', text='desc', source=df,
                        x_offset=2, y_offset=2))
bp.show(fig)
# -

# # GEOGRAPHIC EMBEDDINGS CODE STARTS HERE

# +
# create dictionary of each lat/lon pair



min_lat = 32.0
max_lat = 43.0
min_lon = -124.0
max_lon = -114.0

resolution = 0.2
        
        
lats = np.arange(min_lat, max_lat, resolution)
lons = np.arange(min_lon, max_lon, resolution)

pairs = []

for lat in lats:
    for lon in lons:
        pairs.append((round(lat, 1),round(lon,1)))
        
# print(pairs)

# +
# create haversine calculation
# these calcs take too long for 0.2, 0.1 resolutions. Can filter by pairs that are present in the data

def haversine_distance(point_a, point_b, unit = 'km'):

    lat_s, lon_s = point_a[0], point_a[1] #Source
    lat_d, lon_d = point_b[0], point_b[1] #Destination
    radius = 6371 if unit == 'km' else 3956 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.

    dlat = np.radians(lat_d - lat_s)
    dlon = np.radians(lon_d - lon_s)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat_s)) * np.cos(np.radians(lat_d)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c
    return distance

# +
# create matrix of distances (equivalent to co-occurence matrix for word embeddings)
# these calcs take too long for 0.2, 0.1 resolutions. Can filter by pairs that are present in the data


n = len(pairs)

distances = np.zeros((n,n))

for a in range(n):
    for b in range(n):
        distances[a][b] = haversine_distance(pairs[a], pairs[b])
              
distances   

# +
#SVD

from sklearn.decomposition import TruncatedSVD
def SVD(X, d=100):
    """Returns word vectors from SVD.
    
    Args:
      X: m x n matrix
      d: word vector dimension
      
    Returns:
      Wv : m x d matrix, each row is a word vector.
    """
    transformer = TruncatedSVD(n_components=d, random_state=1)
    Wv = transformer.fit_transform(X)
    # Normalize to unit length
    Wv = Wv / np.linalg.norm(Wv, axis=1).reshape([-1,1])
    return Wv, transformer.explained_variance_


# +
# get final embeddings

d = 10 # embedding dimension
embeddings = SVD(distances, d=d)[0]


# create dict w/ key = lat/lon pair, value = embedding vector

emb_dict = {}

for i in range(len(pairs)):
    emb_dict[pairs[i]] = embeddings[i]
    
# print(emb_dict)

# +
# test out how it worked on a few points

from scipy.spatial import distance

point_a = emb_dict[(32.2, -120.4)]
point_b = emb_dict[(32.2, -121.4)] # very close to point_a
point_c = emb_dict[(32.2, -115.4)] # further from point_a
point_d = emb_dict[(42.2, -115.4)] # very far from point_a

print(f'between points a and b: {distance.euclidean(point_a, point_b)}')
print(f'between points a and c: {distance.euclidean(point_a, point_c)}')
print(f'between points a and d: {distance.euclidean(point_a, point_d)}')
# -




