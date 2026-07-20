# Physics-Informed Reservoir Computing (PIRC)

Official implementation of the paper **"Lightweight Physics-Informed Reservoir Computing for Battery Health Prediction"**, accepted at **IFAC MECC 2026**.

PIRC is a lightweight prognostic framework that integrates physical degradation priors directly into a closed-form ridge regression readout. It employs an adaptive switching mechanism to autonomously select the optimal feature representation, mitigating over-parameterization while strictly minimizing trainable parameters.

##  Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{anurag2026pirc,
  title={Lightweight Physics-Informed Reservoir Computing for Battery Health Prediction},
  author={Anurag, Kumar and Xu, Yanwen and Wan, Wenbin},
  year={2026}
}
```

##  Quick Start

### Prerequisites
The framework relies solely on standard scientific Python libraries. No heavy deep learning frameworks are required.
```bash
pip install numpy scipy pandas matplotlib
```

### Dataset
Official dataset link: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

### Running the Benchmarks

**1. Battery SOH Estimation (NASA PCoE Dataset)**
```bash
python run_battery.py
```

**2. Chaotic Systems (Lorenz & Rössler)**
```bash
python run_chaotic.py
```

## Repository Structure

* **`run_battery.py`** – Main execution script for the battery SOH benchmark.
* **`run_chaotic.py`** – Main execution script for the chaotic systems benchmark.
* **`pirc_core.py`** – Core implementations of the proposed PIRC and Hybrid RC-NGRC models.
* **`reservoir.py`** – Reservoir computing utilities, nonlinear transforms, and NVAR feature construction.
* **`baselines.py`** – Implementations of baseline models (Standard RC and PINN).
* **`systems.py`** – Dynamical system definitions (Lorenz, Rössler, Mackey-Glass).
* **`metrics.py`** – Evaluation metrics and plotting utilities.
* **`5_Battery_Data_Set/`** – Directory containing the NASA PCoE dataset.

##  Developer

[Kumar Anurag](https://kan.phd/)