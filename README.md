# AMLM
A small transformer implementation.

Can be configured from the config file. This is designed to be a small benchmark of the SwiGLU activation in the Transformer vs ReLU and of GQA vs MHA.

In the config.json file, "SwiGLU" utilises SwiGLU, "ReLU" utilises relu, "GQA" utilises GQA and "MHA" utilises MHA. It should be thoes strings exactly or undefined behaviour.

The generator file allows for a promt-response cycle to generate from the transformer.

The tokeniser file can be used to train a BPE tokeniser.
