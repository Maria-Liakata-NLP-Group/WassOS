# WassOS

## Introduction

We developed WassOs, an unsupervised opinion summarization model based on VAE and the
Wasserstein barycenter. In order to caputure the main meaning of different kinds of documents, we disentangle the doc-
ument distributions into separate semantic and syntactic spaces. We introduce these distributions into the Wasserstein
space and construct the summary distribution using the Wasserstein barycenter. This strategy can reduce the
mutual interference of semantic and syntactic information, and identify the representative
summary distribution from multiple noisy documents. 

## Installation

Our code is based on the framework of [Copycat](https://arxiv.org/abs/1911.02247), please follow this [link](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer) to build the conda environment.

## Data

We experimented on 3 different datasets with different types
of content (social media posts, reviews) to allow
for a thorough evaluation across different domains. The social media posts are from Twitter and Reddit. The revirews are from [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets.html).

### Input Data Format

The expected format of input is provided in [artifacts](wassos/artifacts/amazon/data/input_example). 

group_id | review_text | category | review_tag
--- | --- | --- | ---
B000WJ3I1M | I have this cupholder mounted ...  | reviews_electronics_5 | PRP VBP DT NN VBD IN DT ...

We parse each document into the tag sequence with [Zpar](https://www.sutd.edu.sg/cmsresource/faculty/
yuezhang/zpar.html)

## Strategy

We developed two strategies 'T_center' and 'O_center' for this model. For the first strategy 'T_center', It uses the two Wasserstein barycenter 
from semantic and syntactic spaces to construct the summary distribution, and it performs better on social media posts. The second strategy 'O_center' only 
uses one Wasserstein barycenter from semantic space and constrcuct the syntactic distribuiton with the method for each document. It performs better on 
reviews dataset. You can a strategy in [model_hp.py](wassos/utils/hparams/model_hp.py) based on your data.



