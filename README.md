# HatPAM

### Description
---
HatPAM is Python misuse classification tool, whose inputs are commits containing a commit message (nl) and the code change. The model takes LLM (bert or transformer_xl) as natural language encoder to encode commit message. The code change are parsed to AST, combined with data-flow edges, and encoded by GGNN model. The output logits of nl and code are combined to train a multi-layer classification model.






### Dependencies
---
python>=3.8

pytorch==1.8.1

tqdm

gensim

transformers

dgl

nltk

codecs

torch_scatter==1.2.1


---



##### pretrained_models

This directory contains nl encoder LLMs (bert or transformer_xl), and the trained word2vec model.




##### scripts

This directory contrains some useful scripts to crawl api-misuse-related commits via commit id, and do some preprocess.



##### build_vocab.py

To build vocabulary for both nl and code, and train word2vec model to vectorize each token.



##### code_encoder.py

Contains code encoder and NLPL model to get embeddings for commit messages and code changes, then classify to 6 misuse categories. Run this file to train the model.



##### dataset.py

To transform the code graph to dgl.dataset form for later training process.



##### get_dataflow.py  &&  main_dataflow.py

Parse codes with the rules defined in the paper, and combine them with AST. The results are stored as json files.



##### ggnn_utils.py

Contains the main model definition of GGNN. GraphClsGGNN transforms a graph data to a single vector, and GGSNN is to get a sequence of node embeddings(not used).



##### natural_encoder.py

Use pretrained LLMs to encode the commit messages.




##### parse_python.py

Parse a single python file, to get its simplified AST. The results are stored as json files.




##### preprocess.py

Some preprocess for code.