# Multi-Task Bayesian ICL
**Abstract:** Bayesian predictive inference provides a principled framework for uncertainty quantification, data efficiency, and robust generalization. However, exact inference is often intractable, and scalable approximations may remain computationally expensive or require restrictive modeling assumptions that degrade predictive performance. Prior-Data Fitted and in-context learning networks have recently emerged as an amortized alternative by learning to map datasets directly to predictive distributions, but existing approaches are tightly coupled to the support of the training prior and lack explicit mechanisms for adapting to new priors at test time, resulting in limited robustness under distribution shift. We introduce a multi-task in-context learning framework for amortized hierarchical Bayesian predictive inference that explicitly represents prior information as a prefix of in-context datasets. A transformer trained on sequences of prior and target tasks learns to adapt its predictions across families of priors. On a suite of evaluations with increasing difficulty, including out-of-meta-distribution heavy-tailed priors and priors with high-dimensional latent structures, our method matches oracle Bayesian predictors while being orders of magnitude faster. We further demonstrate its practical relevance on a real-world spatiotemporal temperature prediction benchmark.
## Setup
Create the conda environment:
```bash
conda env create -f environment.yml
conda activate mticl
```
## Training MT-ICL
```bash
python src/train_multi_task.py --config configs/...
```
## Evaluating MT-ICL
Please see scripts under `src/scripts/`. 

All ERA5 experiment code is located under `era5/`. After downloading the [data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), you can train the model using `train.sh`, which will also report the test performance using the best val checkpoint. For `2020 test`, evaluate using `eval_2020.sh`.