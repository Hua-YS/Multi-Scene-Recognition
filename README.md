# Multi-Scene-Recognition \[[webpage]()\]\[[paper](https://arxiv.org/pdf/2104.02846.pdf)\]
Multi-scene recognition is a challenging task due to that

+ images are large-scale and unconstrained
+ all present scenes in an aerial image need to be exhaustively recognized

<img src="./figures/illustration.png" width = "555" height = "360" alt="example" align=center />

In this work, we propose a large-scale dataset, namely MultiScene dataset, and provide extensive benchmarks.


## MultiScene Dataset
MultiScene dataset aims at two tasks: 1) developing algorithms for multi-scene recognition and 2) network learning with noisy labels.

We collect 100k high-resolution aerial images with the size of 512x512 around the world. All of them are assigned with crowdsourced labels provided by [OpenStreetMap](https://www.openstreetmap.org/) (OSM), and 14k of them are mannually inspected yielding clean labels. In total, 36 scene categories are defined in our dataset, and 22 models are tested. 

<img src="./figures/data_distribution.jpg" width = "1000" height = "405" alt="example" align=center />






