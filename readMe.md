# DeepDraw <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1588528588264-C0AX88HTUWZIUZDBCMVY/ke17ZwdGBToddI8pDm48kI8xML9w6WVF-A8Fd6Xh7CFZw-zPPgdn4jUwVcJE1ZvWhcwhEtWJXoshNdA9f1qD7RyFOMTxaKexDScLPdXmhUBUGoFwIJWq0ElUUZM0MaiSl578qHjAUVYQjaMp75n45A/deepdraw-01.png?format=300w" width="350" title="DeepDraw" alt="DLC Utils" align="right" vspace = "50">



- We provide a normative approach to derive neural tuning of proprioceptive features from behaviorally-defined objectives.
- We propose a method for creating a scalable muscle spindles dataset based on kinematic data and define an action recognition task as a benchmark.
- Hierarchical neural networks solve the recognition task from muscle spindle inputs.
- Individual neural network units resemble neurons in primate somatosensory cortex, and networks make predictions for other areas along the proprioceptive pathway.

### This repository contains code for the paper "Task-driven hierarchical deep neural network models of the proprioceptive pathway", by Kai J Sandbrink, Pranav Mamidanna, Claudio Michaelis, Mackenzie W Mathis, Matthias Bethge and Alexander Mathis.
Preprint: https://www.biorxiv.org/content/10.1101/2020.05.06.081372v1

The code is organized as follows:
1. Dataset Generation. (Proprioceptive Character Recognition Task = PCRT Dataset) Code can be found in `dataset`
2. Solving the PCRT Dataset using binary SVMs. Code can be found in `svm-analysis`
3. Solving the PCRT Dataset using various network models. Code in `nn-training`
4. Representational Similarity of the models. Code in `repr-analysis`
5. Single Unit tuning curves. Code in (WIP)

`code` contains classes and helper functions used at one/more places in the analyses.

A Docker container with OpenSim binaries is available here: https://hub.docker.com/r/pranavm19/opensim/tags
