# FairNet
_A Genetic Algorithm to Reduce Marginalization in Social Networks_

## What is marginalization?
**_Discrimination in the literature_:** In a network, algorithmic discrimination focuses on the possible discrimination arising from an individual’s network position, whereas the Data Mining field proposes that in a fair dataset similar individuals should be treated similarly.

**_Discrimination as marginalization:_** Marginalization is defined as the act of relegating someone or something to an unimportant position, and can indeed occur between different groups (e.g.,  a non-white person marginalised by white people) or inside of the same group (e.g., a white person marginalised by other white people). In a fair network, all nodes should be surrounded by a group of peers that manifests a similar distribution with respect to an attribute. Additionally, such distribution should be representative of the label distribution in the whole system. A proportionally-different distribution implies some sort of marginalization against the node – either by nodes with different labels or by those with the same label. 

## Our proposal to quantify marginalization

In our work, we introduce _**Individual Marginalization Score**_ (IMS), a measure that takes into account the attribute distribution in the node’s neighbourhood and compares it to the distribution in the whole network. IMS ranges in [-1, 1] and describes:
- marginalization perpetrated by nodes with the same attribute for IMS < 0;
- marginalization perpetrated by nodes with a different attribute for IMS > 0;
- no marginalization for IMS = 0. 

A node can be considered marginalised if its absolute IMS is beyond a fixed threshold. The number of marginalised nodes can quantify marginalization at the macro-scale level scale. We also the introduce the _**System Marginalization Score**_ (SMS), which captures the average marginalization for all nodes, regardless of the sign.

## Our proposal to reduce marginalization

Within the _FairNet_ library, we propose two independent algorithms -- _FairLabel_, and _FairEdges_.

_FairLabel_ is intended to be used with networks where some nodes have missing metadata. It employs a genetic algorithm to find for these nodes the combination of labels that most reduce the number of marginalised nodes (following the above metric).

_FairEdges_ finds a combination of edges to be added that minimises the number of marginalised nodes with relatively limited modifications to the network. Edges are selected following the triadic closure principle (e.g., we assume that a non-existing edge that would close 10 triangles is more likely to appear than one that would close 2 triangles). Plausible edges are then encoded in a binary vector. Starting from such a vector, a genetic algorithm tries to minimise the number of marginalised nodes. If equal solutions are found, the one with fewer interventions is prioritised.

Experimental results show that the _FairNet_ library successfully reduces the number of discriminated nodes.

<p align="center">
<img width="380" alt="image" src="https://github.com/andreafailla/fairnet/assets/80719913/cb20b9a1-bf21-46c5-ba11-d065b985cd4a">
</p>





[![SBD++](https://img.shields.io/badge/Available%20on-SoBigData%2B%2B-green)](https://sobigdata.d4science.org/group/sobigdata-gateway/explore?siteId=20371853)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a simple template to exemplify: how to structure a python project and leverage GitHub Actions to automatically build and publish it as a python package (to PyPI and Conda).

The template project describes a simple Python package containing an object class, an object container ad a few functions to sort them and convert from/to JSON.

The template comes with:
- a few GitHub Actions to automatically build and publish the package to PyPI (and Conda) when a new release is created (or on demand);
- a GitHub Action to check the code quality using CodeQL;
- a GitHub Action to check compliance with Black code style;
- a few tests to exemplify the use of pytest;
- a minimal sphinx documentation to exemplify the use of sphinx and readthedocs;
- a minimal configuration files to exemplify the use of gitignore and coveragerc;
- a minimal setup.py to exemplify the use of setuptools.

## Wiki
To learn more about the project (and for a crash course on the listed topics), please check the dedicated [wiki](https://github.com/GiulioRossetti/Python-Project-Template/wiki).

## License
This project is licensed under the terms of the BSD-2-Clause license.

## Acknowledgements
This repository was developed within the [SoBigData++](https://sobigdata.d4science.org/group/sobigdata-gateway/explore?siteId=20371853) H2020 project training activities (WP4) to support "Social Mining and Big Data resources Integration" (WP8).

## Contact(s)
[Giulio Rossetti](mailto:giulio.rossetti@gmail.com) - CNR-ISTI 

Twitter: [@giuliorossetti](https://twitter.com/GiulioRossetti)

Mastodon: [@giuliorossetti@mastodon.uno](https://mastodon.uno/@giuliorossetti)

