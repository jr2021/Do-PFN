# Do-PFN 🔨🔍

Attached is the code supporting "Do-PFN: In-Context Learning for Causal Effect Estimation". Do-PFN is a pre-trained transformer for causal inference, trained to predict conditional interventional distributions (CIDs) and conditional average treatment effects (CATEs) from observational data alone.



CIDs are the posterior distribution of an outcome $y$ given an intervention $do(t)$ and covariates $x$: 

$$CID := p(y | do(t), x)$$

A CID answers a question like "What is the distribution of outcomes given that (i) a patient has features $x$ and (ii) an intervention $do(t)$ is performed?". We assume the outcome $y$ as well as the covariates $x$ to be continuos, numerical values. The treatment $t$ can take the values $1$ (treatment) and $0$ (no treatment). 

Using Do-PFN one can also estimate Conditional Average Treatment effects (CATEs) via the following formula: 

$$CATE:= E[y | do(t=1), x] - E[y | do(t=0), x]$$

### Installation

Create a conda environment with python version 3.10, activate it, and install the requirements in requirements.txt

```
   conda create -n dopfn_env python=3.10
   conda activate dopfn_env
   pip install -r requirements.txt
```

### Running Do-PFN

Run ```inference_example.py``` to play around with Do-PFN on any of our real-world and synthetic benchmark datasets by changing the ```dataset``` variable (runs on "Observed Confounder" by default)

### Sampling data from the prior

Run ```prior_data_example.py``` to sample data from our prior.

# Background 

### What is Do-PFN? 
Do-PFN is a prior-data-fitted network (PFN) for causal effect estimation. Pre-trained on synthetic data drawn from structural causal models (SCMs), Do-PFN learns across millions of causal structures and simulated interventions to predict conditional interventional distributions.
In practice, Do-PFN can answer causal questions like “what will be the effect of a certain medication on a health outcome” from only observational data, and without explicit knowledge of how the variables in the system interact. We believe that Do-PFN can provide causal insights on diverse and understudied scientific problems, where experimental randomized controlled trial (RCT) data is scarce. 

### How does it work?
Do-PFN is pre-trained on millions of synthetic datasets drawn from a wide variety of causal structures to predict interventional outcomes given observational data. In real-world applications, Do-PFN leverages the many simulated interventions it has seen during pre-training to predict CIDs, relying only on observational data and requiring no information about the causal graph.
We evaluate the performance of Do-PFN on six case studies across more than 1,000 synthetic datasets. For both predicting CID and CATE, Do-PFN (1) achieves competitive performance with baselines that have access to the true causal graph (typically not available in practice) and (2) outperforms real-world baselines even when unconfoundedness assumptions are satisfied. Furthermore, Do-PFN works well on small datasets, is robust to varying base rates of the average treatment effect, and performs consistently on large graph structures.

### Why is this different?

Do-PFN is a radical new approach to causal inference, replacing standard assumptions of a ground-truth causal model (Pearl) or assumptions about its structure (Rubin) with a “prior” over SCMs. In other words, our modeling assumptions lie in our simple, yet general and extensible synthetic data-generating process. 
In practice, we relax the assumption of a specific causal graph or structure, to the weaker: “there exists a causal structure behind the data that is represented in our prior over SCMs”
#In practice, we relax the assumption of a specific causal graph or structure, to the much weaker: “there exists a causal explanation for the data”

### Predicting CIDs

```python
dataset = load_dataset(ds_name='sales')
dopfn = DoPFNRegressor()

train_ds, test_ds = dataset.generate_valid_split(n_splits=2)

dopfn.fit(train_ds.x, train_ds.y)
y_pred = dopfn.predict_full(test_ds.x)
```
### Estimating CATEs

```python
from copy import deepcopy

dopfn.fit(train_ds.x, train_ds.y)

x_1, x_0 = deepcopy(test_ds.x), deepcopy(test_ds.x)
x_1[, 0], x_0[, 0] = 1, 0

y_pred_1 = dopfn.predict_full(x_1)
y_pred_0 = dopfn.predict_full(x_0)

cate_pred = y_pred_1 - y_pred_0
```

For questions, comments, and discussion, please contact corresponding authors:
- Jake Robertson: robertsj@cs.uni-freiburg.de
- Arik Reuter: arik.reuter@campus.lmu.de
