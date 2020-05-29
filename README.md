![AMORE-UPF](logos/logo-amore-text-diagonal.png)    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      ![UPF](logos/upf-logo.png)

# Modeling word interpretation with deep language models: The interaction between expectations and lexical information

This repository contains the code and language model necessary to replicate the framework presented in " Modeling word interpretation with deep language models: The interaction between expectations and lexical information " (to appear in Proc. CogSci 2020). 

##### Citation

```
@inproceedings{aina-gulordava-boleda:2019:CL,
    title     = {Modeling word interpretation with deep language models: The interaction between expectations and lexical information},
    author    = {Aina, Laura and Brochhagen, Thomas and Boleda, Gemma},
    booktitle = {Proceedings of the 42nd Annual Conference of the Cognitive Science Society},
    year      = {2020}
}
```

### Contents of the repository

* Code for extracting representations from LSTM and BERT models, as reported in the paper:
  - lexical information
  - expectations
  - expectation-driven word interpretation, with avg or delta operations, modulated by parameter alpha 

* Bidirectional LSTM language model used in the experiments reported in the paper (`LSTM/model.pt`)

* Jupyter Notebook showing how to extract representations and basic functions to evaluate them (e.g,, cosine similarity, nearest neighbors)

        `src/demo.ipynb`
        
