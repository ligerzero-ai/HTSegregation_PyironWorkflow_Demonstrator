# HT Segregation Pyiron Workflow Demonstrator

High-throughput grain boundary segregation demonstrator using pyiron workflow, atomistics, and LAMMPS.

## Description

This repository contains a Jupyter notebook demonstrating high-throughput grain boundary segregation calculations using the pyiron workflow ecosystem. The workflow includes:

- Crystal structure optimization
- Solution energy calculations
- Grain boundary structure generation and analysis
- Segregation energy calculations

## Requirements

- Python 3.12
- pyiron-workflow-atomistics
- pyiron-workflow-lammps
- LAMMPS (from conda-forge)

## Running on Binder

Click the badge below to launch the notebook in an interactive Binder environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ligerzero-ai/HTSegregation_PyironWorkflow_Demonstrator/main)

## Local Installation

### Using Conda (Recommended)

Create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ht_segregation
```

### Using pip

Alternatively, install the required dependencies with pip:

```bash
pip install -e .
```

Or with notebook support:

```bash
pip install -e ".[notebook]"
```

## Usage

Open and run the `grain_boundary_segregation.ipynb` notebook to execute the workflow.

## License

BSD License
