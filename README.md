# P9
This project is a quantum circuit verification and simulation tool using TDDs[[1]](#1) in conjunction with the contraction planning tool Cotengra[[2]](#2). The project uses parameterised circuits provided by MQT.Bench[[3]](#3). This project can only be run on Linux (Ubuntu 20.04) and using Python 3.9. 

## Installation and Preparation
To ensure that the required python libraries are installed, run the following command from the root of the folder:
```
pip install -r requirements.txt
```

If the external repositories are not properly collected, run the following command:
```
git submodule update --init --recursive
```


## Running experiment
To perform any of the experiments seen in the associated article, the main experiment file must be targeted from the root of the folder.
```
python3 src/experiment [arguments]
```

All arguments for the experiments can be seen below.

An example of an experiment may be running the DJ algorithm with qubits 64, 128, and 256 using Cotengra's RGreedy heuristic

```
python3 src/experiment --exp_name DJ_RGreedy_Benchmark --exp_type formal --setup standard --algorithms dj --level 0,2 --qubits 64 128 256
```

## Possible options

| Name             | Values                                                                                        | Description                                                                                                 |
|------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| --exp_name       | any string                                                                                    | The name of the experiment and folder where data is saved                                                   |
| --exp_type       | formal, simulation                                                                            | Determines the setup for the experiment                                                                     |
| -counter         |                                                                                               | Flag for whether a counter example when inequality is determined should be found                            |
| -ts              |                                                                                               | Flag for whether to use time stamp for folder name                                                          |
| --setup          | standard, pure_qcec                                                                           | Denotes the type of method to test (standard is article method)                                             |
| --algorithms     | ghz, dj, graphstate, twolocalrandom, qftentangled, qpeexact, su2random, wstate, realamprandom | The circuit algorithm used for the setup                                                                    |
| --levels         | any two-tuple of 2 ints in [0,3], e.g. 0,2                                                    | The abstraction levels for each of the two circuit                                                          |
| --qubit_range    | any two-tuple of 2 ints                                                                       | The start and stop of number of qubits                                                                      |
| --qubits         | list of ints                                                                                  | The exact number of qubits to test for                                                                      |
| --gate_deletions | int                                                                                           | The number of random gates to delete. Must be greater than or equal to zero. Cannot be the entire circuit.  |
| --reps           | int                                                                                           | Number of repetitions                                                                                       |
| -split           |                                                                                               | Flag for splitting the multi-qubit gates, e.g. CNOT                                                         |
| -subnets         |                                                                                               | Flag for using subnetworks                                                                                  |
| --max_cont_time  | int                                                                                           | Max time allowed for contraction, -1 for no limit                                                           |
| --method         | cotengra, linear                                                                              | The overall method used for planning                                                                        |
| --path_method    | rgreedy, betweenness, walktrap, linear, proportional                                          | The heuristic used for planning. Depends on the method selected                                             |
| --minimize       | flops, size, combo                                                                            | The minimization goal for cotengra when finding plans                                                       |
| --max_repeats    | int                                                                                           | The number of repeats plans found by cotengra                                                               |
| --max_plan_time  | int                                                                                           | Max allowed time for planning (soft constraint).                                                            |


## References
<a id="1">[1]</a> 
Xin Hong, Xiangzhen Zhou, Sanjiang Li, Yuan Feng,
and Mingsheng Ying (2021). 
A tensor network based decision diagram for representation of quantum circuits.

<a id="2">[2]</a> 
Johnnie Gray and Stefanos Kourtis (2021). 
Hyper-optimized tensor network contraction. 
Quantum, 5:410, mar 2021.

<a id="3">[3]</a> 
Nils Quetschlich, Lukas Burgholzer, and Robert Wille (2023). 
MQT Bench: Benchmarking software and design automation tools for quantum computing.
Quantum, 2023.