# WassOS

## Introduction

We developed THVAE, an unsupervised timline summarization model based on hierarchical VAE 

## Installation

Our code is based on the framework of [Copycat](https://arxiv.org/abs/1911.02247), please follow this [link](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer) to build the conda environment.

## Data

We experimented on talk-lfe datasets 

### Input Data Format

The expected format of input is 

group_id | review_text | category | review_tag
--- | --- | --- | ---
136861_255 | im worthless im literally a fucking failure . let me die | post | im worthless literally failure die


## Key phrases

The method of getting key phrases is in [file](https://github.com/Maria-Liakata-NLP-Group/THVAE-summary/blob/main/get_prompt.py), 'read_timeline'.




