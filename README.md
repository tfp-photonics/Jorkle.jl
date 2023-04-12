#  Jorkle

**J**ulia-based **O**ptimization of **R**elativistic Spheres to Explore **K**erker-**L**ike **E**ffects

This repository contains code for the paper "Identifying regions of minimal back-scattering by a relativistically-moving sphere" ([arXiv](https://arxiv.org/abs/2304.05319)).

## Installation

To install Jorkle, download/clone this repository and install it using Julia's package manager. Navigate to the package directory, open the Julia REPL, type `]` to enter package mode, and install as follows:
```julia
pkg> add .
```

## Usage

We provide two command-line utilities in the folder `scripts/` with which the results from the paper can be reproduced.

The quadrupole sweeps for specific dipole angles were obtained using the file `scripts/sweep_quadrupoles.jl`.

The backscattering minimization was done using the file `scripts/minimize_directivity.jl`.
Optimization results as shown in Table 1 in the paper can be found in the folder `results/`.

For detailed usage, refer to the `cmdline_args()` function of the two scripts.

## Citing

If you find this code useful in your research, please cite:

```
@article{whittam2023identifying,
  title={Identifying regions of minimal back-scattering by a relativistically-moving sphere},
  author={Whittam, Mitchell R. and Lamprianidis, Aristeidis G. and Augenstein, Yannick and Rockstuhl, Carsten},
  journal={arXiv preprint arXiv:2304.05319},
  year={2023}
}
```
