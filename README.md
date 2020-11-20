### Detailed Project Execution Plan

*October 21st, 2020*


#### Topic

Evaluate Unified Virtual Memory (UVM) technology's performance for graph algorithms

#### Algorithms

| Algorithm Covered | Computational Complexity | Previous work | To be finished in the project |
| --- | --- | --- | --- |
| BFS | **Polynomial** O(\|V\|) | 1.Open MP version based on traditional algorithm   <br/> 2. GPU version based on traditional algorithm | CUDA GPU UVM version based on HALO-I algorithm2  <br/> CUDA GPU version based on HALO-I algorithm2 |
| k-truss  <br/> decomposition | **Polynomial** | 1.OpenMP version based on peeling algorithm  <br/> 2.OpenMP version based on SND algorithm  <br/> 3.CUDA GPU version based on traditional algorithm5 | CUDA GPU UVM version based on SND algorithm |

- Why to choose these algorithms:

I want to explore the performance of UVM technology in multiple computational complexities for large scale graph data. Hence I get BFS (linear complexity) and K-truss (generally quadratic complexity).

#### Timeline

| Time Period    |  TODOs   |
| --- | --- |
| **2020.10.05 - 2020.10.15** | literature reading, CUDA UVM program tutorial, reevaluate and improve the project topic. |
| **2020.10.16 - 2020.10.26** | GPU HALO-I algorithm2 |
| **2020.10.27 - 2020.11.10** | GPU UVM version based on HALO-I algorithm2 |
| **2020.11.11 - 2020.11.20** | GPU UVM version based on SND algorithm3 |
| **2020.11.20 -** | Summarizing and doc writing. |

#### Results Metrics

| **Algorithm** | **Metrics** | **Trade-off** |
| --- | --- | --- |
| **BFS** | **speedup, efficiency** | **/** |
| **k-truss** | **speedup, accuracy, iteration numbers** | speedup / iteration numbers v.s. accuracy |

> How to deal with:
> Generally the accuracy should be guaranteed first, in [the reference paper](https://arxiv.org/abs/1704.00386), when the optimized algorithm goes >to convergence, at least 98% accuracy should be reached by the algorithm. 


#### How do you define and quantify success in your project

**Basic**

1. Implement planned algorithms under GPU UVM technique, pass the test;
2. Compare the GPU UVM version program with OpenMP (CPU) version, pure GPU version programs to evaluate GPU UVM&#39;s performance in graph algorithms;
3. Reach speedup and accuracy at least faster than OpenMP version, and seem to be reasonable as shown in [https://dl.acm.org/doi/abs/10.14778/3384345.3384358](https://dl.acm.org/doi/abs/10.14778/3384345.3384358) for BFS and [https://arxiv.org/abs/1704.00386](https://arxiv.org/abs/1704.00386) for k-truss.

**Advanced**

1. Give feasible improvement suggestion for algorithms implemented
2. Show by experiment that the suggestion works.



#### Appendix

**Project Proposal**

*October 5th, 2020*

**Topic:** Will Unified Memory Be the Final Piece of the GPU Programming Puzzle?

**Subtitle:** From the view of k-core algorithms

**Motivations:**

When adopting graph mining algorithms with GPU, most prior work on graph applications has been restricted to the size of the graph due to the limited capacity of GPU memory1,2. Recently a novel technology known as unified virtual memory, which supports unified virtual memory between CPUs and GPUs, makes it possible to address the very puzzle. However, such an approach to accessing host memory while computing is probably much slower, and the problem has been seldom explored1. Hence, I want to focus on implementing and testing some specific graph mining algorithms to catch a glimpse of it.

Prasun Gera at al. (2020)1, has done exciting work about evaluating large graphs traversal on GPUs with unified memory and designing a lightweight offline graph reordering algorithm (HALO) as a preprocessing step to boost the algorithm. Their work yields speedups of 1.5x-1.9x over the baseline. Their work will be a great guideline for my project.

While I looked through the papers in the Github repository, I noticed that k-cores and BFS are so popular to be targets of optimization in the field of HPC on GPUs. Hang Liu (2019)6, Laxman Dhulipala (2017)5, and Ahmet Erdem (2018)7 have all done great jobs to design fast parallel computing frameworks to accelerate graph algorithms, and they all select the k-core algorithm as part of their experiments. However, none of them did it on united virtual memory and imported large scale of graphs as input. I think I can take advantage of their optimization ideas to help me implement the k-core algorithm on united memory and their results can be the baseline results for me to reach the conclusion.

**Goals:**

1. Understand how unified virtual memory works under CUDA architecture, enable to write k-core and other graph algorithms (e.g. BFS, DFS) using unified virtual memory works;
2. Import large graphs in Prasun Gera at al. (2020)1&#39;s paper, try to define the problem for k-core algorithm and manage to prove UVM is working while running the program;
3. Compare the UVM program with GPU-only approaches on their performance, and summarize what causes the difference.
4. Take Prasun Gera at al. (2020)1, Hang Liu (2019)6, Laxman Dhulipala (2017)5, and Ahmet Erdem (2018)7 &#39;s work into consideration, as well as optimization tutorials on Nvidia&#39;s official website3,4. Test whether HALO also works for k-core algorithm, and also try to come up with some other feasible optimizing methods to boost the program above.
5. Review the experiment and make a judgement on whether Unified Memory Will Be the Final Piece of the GPU Programming Puzzle, point out its advantages and disadvantages.
6. \* Try different algorithms to make comparison. (like gSpan8,9 etc.)

**Expected Results:**

1. CUDA source codes of baseline and improved k-core algorithm on GPU with unified memory. If time permits, there will be another version of gSpan algorithm.
2. A detailed report on all the experiment data, together with how the improvement works and the analysis on the unified memory.

**Schedule:**

1. 2020.10.05 - 2020.10.15: literature reading, CUDA UVM program tutorial, reevaluate and improve the project topic.
2. 2020.10.15 - 2020.10.26: Implement baseline algorithms (UVM version, GPU-only version, CPU-only version, etc. ).
3. 2020.10.26 - 2020.11.10: Work on the improvement ideas.
4. 2020.11.10 - : Summarizing and report writing.

### Reference

1. [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/), referred on November 15, 2020

2. [Beyond GPU Memory Limits with Unified Memory on Pascal](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/)

3. [Unified Memory in CUDA 6](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/)

4. [Maximizing Unified Memory Performance in CUDA](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)

1. [Truss Decomposition in Massive Networks](https://arxiv.org/pdf/1205.6693.pdf)

2. [Traversing large graphs on GPUs with unified memory](https://dl.acm.org/doi/abs/10.14778/3384345.3384358)

3. [Local Algorithms for Hierarchical Dense Subgraph Discovery](https://arxiv.org/abs/1704.00386)

4. [KarypisLab/K-Truss](https://github.com/KarypisLab/K-Truss)

5. [awadhesh14/Parallel\_Programming\_Project/parallel/old](https://github.com/awadhesh14/Parallel_Programming_Project/tree/master/parallel/old)

6. [orancanoren/GPU-CPU-Parallel-Graph-Algorithms/1\_BFS](https://github.com/orancanoren/GPU-CPU-Parallel-Graph-Algorithms/tree/master/1_BFS)

7. [kaletap/bfs-cuda-gpu](https://github.com/kaletap/bfs-cuda-gpu)
