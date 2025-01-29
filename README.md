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

_FairLabel_ is intended to be used when some nodes have missing metadata. It employs a genetic algorithm to fill these nodes with the combination of labels that most reduce the number of marginalised nodes (following the above metric).
_FairEdges_ finds a combination of edges to be added minimising the number of marginalised nodes, with relatively limited modifications to the network. Edges are selected following the _triadic closure principle_ (e.g., we assume that a non-existing edge that would close 10 triangles is more likely to appear than one that would close 2 triangles). Plausible edges are then encoded in a binary vector. Starting from such a vector, a genetic algorithm tries to minimise the number of marginalised nodes.
 
Experimental results show that the _FairNet_ library successfully reduces the number of discriminated nodes.

