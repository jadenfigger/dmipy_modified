# Dmipy Codebase: Improvement Analysis

## Table of Contents
1. [Bugs and Correctness Issues](#1-bugs-and-correctness-issues)
2. [Outdated Infrastructure](#2-outdated-infrastructure)
3. [Software Engineering Gaps](#3-software-engineering-gaps)
4. [Missing Diffusion MRI Models](#4-missing-diffusion-mri-models)
5. [Missing Fitting Techniques](#5-missing-fitting-techniques)
6. [Missing Distribution Models](#6-missing-distribution-models)
7. [Testing Gaps](#7-testing-gaps)
8. [Prioritized Recommendations](#8-prioritized-recommendations)

---

## 1. Bugs and Correctness Issues

### 1.1 Reversed Requirements in setup.py (line 11)

```python
requirements=f.read().splitlines()[::-1]
```

The `[::-1]` reverses the dependency list. This means install order is inverted, which can cause dependency resolution failures when packages depend on each other during installation.

### 1.2 Whitespace Typo in acquisition_scheme.py (line 87)

```python
if deltas.size ==0:
```

Minor style issue, but more importantly this line has no validation that `deltas` is even an ndarray at this point.

### 1.3 Potential Negative Tau in acquisition_scheme.py (line 72)

```python
self.tau = Delta - delta / 3.
```

No validation that `delta/3 <= Delta`. A negative `tau` (diffusion time) is physically impossible and would silently produce incorrect signal predictions downstream.

### 1.4 Deprecated LinAlgError Path in mix.py (line 168)

```python
except np.linalg.linalg.LinAlgError:
```

Should be `np.linalg.LinAlgError`. The nested `linalg.linalg` path is a deprecated internal access pattern that may break in future numpy versions.

### 1.5 Nested Volume Fraction Conversion Bug Risk in mix.py (line 174)

```python
for i in np.arange(1, len(vf_nested)):
    vf_nested[i] = vf[i] / vf[i - 1]
```

If `vf[i-1]` is zero or near-zero (which is possible after COBYLA with positivity constraints), this produces `inf` or `nan` that propagates into the L-BFGS-B refinement step silently.

### 1.6 Hardcoded Codecov Token in .travis.yml (line 79)

```yaml
- codecov -t a66df7b8-efce-412e-9ea8-2598b09b186e
```

A secrets token is committed directly in source control. This should be an environment variable.

### 1.7 `pkg_resources` is Deprecated

Used in 7+ files. `pkg_resources` from setuptools is deprecated in favor of `importlib.resources` (Python 3.9+). It is also slow at import time, adding measurable startup latency.

### 1.8 `from __future__ import division` is Unnecessary

Found in 5 files. This was needed for Python 2 compatibility. Since Python 2 is EOL and `/` always does true division in Python 3, these imports are dead code.

### 1.9 `pathlib` in requirements.txt

`pathlib` is a built-in module since Python 3.4. Listing it as a dependency is unnecessary and confusing.

### 1.10 `boto` Instead of `boto3`

`boto` (AWS SDK v1) has been deprecated since 2015. The HCP downloader should use `boto3`.

---

## 2. Outdated Infrastructure

### 2.1 Python Version Support

`.travis.yml` tests Python 2.7, 3.5, 3.6, and 3.7 -- **all are end-of-life**. The project should target Python 3.9+ (ideally 3.10-3.12).

### 2.2 CI/CD Platform

Travis CI has significantly reduced its free tier. Migration to **GitHub Actions** would be more sustainable and modern.

### 2.3 Test Runner

`nosetests` (line 62 of `.travis.yml`) has been unmaintained since 2015. Should migrate to **pytest**.

### 2.4 Dependency Pinning

`requirements.txt` has:
- `numpy(>=1.13)` -- allows extremely old versions
- `dipy`, `cvxpy`, `boto` -- completely unpinned
- No upper bounds, risking silent breakage from API changes

Should use a modern approach: `pyproject.toml` with version ranges, plus a lockfile for reproducibility.

---

## 3. Software Engineering Gaps

### 3.1 No Type Hints

Zero type hints across the entire codebase. Adding type annotations to public APIs would improve IDE support, catch bugs earlier via mypy, and serve as documentation.

### 3.2 No Abstract Base Classes

Model interfaces are purely convention-based. A `CompartmentModel` ABC with `@abstractmethod` for `__call__` and `spherical_mean` would enforce compliance and catch missing implementations at class definition time rather than runtime.

### 3.3 Print Statements Instead of Logging

~70 occurrences of `print()` across 12+ files. These should use Python's `logging` module to allow users to control verbosity, redirect output, and integrate with their own logging infrastructure.

### 3.4 Magic Numbers Without Documentation

- `DIFFUSIVITY_SCALING = 1e-9` (cylinder_models.py:15)
- `DIAMETER_SCALING = 1e-6` (sphere_models.py:7)
- `Dstar_value=7e-9` (intra_voxel_incoherent_motion.py:32)
- `_parameter_ranges = {'lambda_iso': (.1, 3)}` (gaussian_models.py) -- units unclear

These scaling constants and range bounds need docstrings explaining their physical meaning and units.

### 3.5 God Object: modeling_framework.py

At 2,248 lines, `modeling_framework.py` is doing too much. It contains `ModelProperties`, `MultiCompartmentModelProperties`, `MultiCompartmentModel`, `MultiCompartmentSphericalMeanModel`, `MultiCompartmentSphericalHarmonicsModel`, and multiple utility functions. This should be split into separate modules.

### 3.6 No Spatial Constraints in Fitting

All fitting is strictly voxel-by-voxel. There is no support for:
- Total variation regularization (smooth parameter maps)
- Graph-based spatial priors
- Neighborhood-aware fitting
- Joint estimation across voxels

This is a significant limitation for real-world neuroimaging data where spatial coherence is expected.

### 3.7 No Uncertainty Quantification

Models return only point estimates. There is no mechanism for:
- Confidence intervals on fitted parameters
- Fisher Information Matrix computation
- Posterior distributions
- Model comparison metrics (AIC, BIC)

---

## 4. Missing Diffusion MRI Models

### 4.1 High-Impact Clinical Models

| Model | Status | Significance |
|-------|--------|-------------|
| **DKI** (Diffusional Kurtosis Imaging) | Not implemented | Widely used clinically; captures non-Gaussian diffusion. Needs kurtosis tensor + invariant metrics (MK, AK, RK) |
| **WMTI** (White Matter Tract Integrity) | Not implemented | DKI-based biophysical metrics for axonal water fraction, intra/extra-axonal diffusivities |
| **FREE Water Elimination** | Not implemented | Removes CSF contamination from WM voxels; critical for aging/neurodegeneration studies |
| **CHARMED** | Not implemented | Gold-standard multi-compartment model combining hindered + restricted diffusion |
| **ActiveAx** | Not implemented | Axon diameter estimation with optimized acquisition |
| **DIAMOND** | Not implemented | Distribution of anisotropic microstructural environments |
| **SMT2** (Spherical Mean Technique v2) | Not implemented | Microscopic diffusion anisotropy without orientation fitting |

### 4.2 Propagator-Based Models

| Model | Status | Significance |
|-------|--------|-------------|
| **MAP-MRI** | Not implemented | Orthonormal basis (Hermite polynomials) for diffusion propagator; yields RTOP, RTAP, RTPP, non-Gaussianity metrics |
| **SHORE** | Not implemented | Simple Harmonic Oscillator basis for propagator reconstruction |
| **QBI / Q-Ball** | Not implemented | ODF reconstruction via Funk-Radon transform |
| **MAPL** (MAP with Laplacian regularization) | Not implemented | Regularized MAP-MRI for noisy clinical data |

### 4.3 Relaxation-Diffusion Models

| Model | Status | Significance |
|-------|--------|-------------|
| **T2-Diffusion correlation** | Not implemented | Joint T2-relaxation and diffusion modeling for multi-TE acquisitions |
| **T1-Diffusion models** | Not implemented | Inversion recovery + diffusion for myelin water imaging |
| **Exchange models (NEXI, FEXI)** | Not implemented | Water exchange between compartments; emerging field (Jelescu et al. 2022) |

### 4.4 Time-Dependent Diffusion

Currently only `G3TemporalZeppelin` exists. Missing:
- **Full PGSE time-dependence formalism** for intra-axonal compartments
- **Oscillating gradient** (OGSE) models
- **Multi-diffusion-time analysis** framework

---

## 5. Missing Fitting Techniques

### 5.1 Bayesian / Probabilistic Methods (Highest Priority Gap)

| Technique | Impact | Notes |
|-----------|--------|-------|
| **MCMC sampling** | High | Posterior distributions on all parameters; gold standard for uncertainty. Could use `emcee` or `PyMC` |
| **Variational Bayes** | High | Faster approximate posterior; used in FSL's BEDPOSTX for ball-and-stick fitting |
| **Approximate Bayesian Computation (ABC)** | Medium | Likelihood-free inference for complex models where likelihood is intractable |
| **Fisher Information Matrix** | Medium | Analytic uncertainty bounds; fast to compute post-fit |
| **Bayesian model comparison** | Medium | AIC/BIC/Bayes factors for selecting number of compartments |

### 5.2 Machine Learning-Based Fitting

| Technique | Impact | Notes |
|-----------|--------|-------|
| **Supervised NN fitting** | High | Train network on simulated data to predict parameters; 1000x faster than conventional optimization. See MUDI challenge, Microlearn |
| **Physics-informed neural networks (PINNs)** | High | Embed biophysical constraints in network architecture; emerging 2022-2025 |
| **Self-supervised fitting** | Medium | No simulation needed; network learns from data directly |
| **Amortized inference** | Medium | Neural posterior estimation; combines speed of NNs with uncertainty from Bayesian methods |

### 5.3 Dictionary-Based Fitting

| Technique | Impact | Notes |
|-----------|--------|-------|
| **AMICO-like dictionary matching** | Partially exists | Current AMICO implementation is limited; could support more model types |
| **Sparse coding / basis pursuit** | Medium | Represent signal as sparse combination of dictionary atoms |
| **Compressed sensing** | Medium | Exploit sparsity in parameter space for undersampled data |

### 5.4 Regularization Extensions

Current optimizers have minimal regularization. Missing:
- **L1/Lasso regularization** for automatic model selection (sparse volume fractions)
- **Elastic net** combining L1 + L2
- **Total variation** for spatially smooth parameter maps
- **Adaptive regularization** tuning lambda per-voxel based on SNR

### 5.5 Automatic Differentiation

The codebase relies on scipy optimizers with finite-difference gradients. Migrating signal model forward passes to **JAX** or **PyTorch** would enable:
- Exact gradient computation via autodiff
- GPU acceleration for voxelwise fitting
- Compatibility with modern ML pipelines
- Significant speedup for gradient-based optimizers

---

## 6. Missing Distribution Models

| Distribution | Status | Use Case |
|-------------|--------|----------|
| **von Mises-Fisher** | Not implemented | Generalizes Watson; better for high-concentration orientations |
| **Power-law diameter distribution** | Not implemented | Heavy-tailed axon diameter distributions observed in histology |
| **Log-normal distribution** | Not implemented | Alternative to Gamma for axon diameters; common in biology |
| **Truncated distributions** | Not implemented | Enforce physical bounds (e.g., positive diameters) |
| **Mixture of Watsons** | Not implemented | Multiple fiber populations within one distribution model |
| **Matrix Fisher (Bingham-Mardia)** | Not implemented | Full 3D orientation distribution on SO(3) |

---

## 7. Testing Gaps

### 7.1 Entirely Untested Modules

| Module | Files | Risk |
|--------|-------|------|
| `optimizers_fod/` | csd_tournier.py, csd_cvxpy.py, csd_plus.py | **Critical** -- FOD estimation is a primary use case |
| `custom_optimizers/` | intra_voxel_incoherent_motion.py, single_shell_three_tissue_csd.py | **High** -- specialized clinical algorithms |
| `hcp_interface/` | downloader_aws.py | Medium -- external dependency |

### 7.2 Models Without Dedicated Tests

- Capped cylinder models (`CC2`, `CC3`)
- Plane models (`P2`, `P3`) -- only tested indirectly
- Sphere models (`S1`, `S2`) -- only tested indirectly

### 7.3 Optimizers Without Tests

- `brute2fine.py` -- the primary optimizer, no dedicated tests
- `mix.py` -- no dedicated tests
- `multi_tissue_convex_optimizer.py` -- no dedicated tests

### 7.4 Missing Edge Case Coverage

- Empty acquisition schemes
- Single-measurement acquisitions
- All-b0 data
- NaN/Inf input data handling
- Parameter boundary conditions (what happens at the edges of parameter ranges?)
- Numerical stability with very high/low b-values

---

## 8. Prioritized Recommendations

### Tier 1: Fix Now (Bugs + Infrastructure)

1. **Fix reversed requirements** in setup.py (line 11: remove `[::-1]`)
2. **Remove hardcoded codecov token** from .travis.yml
3. **Add tau validation** in acquisition_scheme.py (assert `delta/3 <= Delta`)
4. **Guard against division by zero** in mix.py nested fraction conversion
5. **Migrate CI to GitHub Actions** with Python 3.10+, pytest
6. **Replace `pkg_resources`** with `importlib.resources`
7. **Replace `boto` with `boto3`**, remove `pathlib` from requirements
8. **Pin dependency versions** properly in pyproject.toml

### Tier 2: High-Impact Additions

1. **Bayesian/MCMC fitting** -- add an `MCMCOptimizer` class using `emcee` for posterior sampling and uncertainty quantification
2. **DKI model** -- widely requested, relatively straightforward kurtosis tensor extension
3. **Free water elimination** -- simple two-compartment model, large clinical impact
4. **Fisher Information Matrix** -- analytic uncertainty bounds, fast to compute
5. **Add tests** for optimizers_fod, custom_optimizers, and all signal models

### Tier 3: Strategic Enhancements

1. **Neural network fitting** -- train on simulated data for 1000x speedup
2. **JAX/PyTorch backend** -- automatic differentiation + GPU support
3. **MAP-MRI / SHORE** -- propagator-based analysis
4. **CHARMED model** -- multi-compartment gold standard
5. **Spatial regularization** -- total variation, neighborhood priors
6. **Exchange models (NEXI)** -- water exchange between compartments
7. **Type hints + ABCs** -- improve maintainability and IDE support

### Tier 4: Long-Term Vision

1. **Physics-informed neural networks** for fast, constrained fitting
2. **Amortized Bayesian inference** (neural posterior estimation)
3. **pyproject.toml** migration (modern Python packaging)
4. **Comprehensive API documentation** with Sphinx + autodoc
5. **Refactor modeling_framework.py** into smaller, focused modules
