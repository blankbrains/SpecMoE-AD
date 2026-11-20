# SpecMoE-AD

Spectrum-aware mixture-of-experts for multivariate time-series anomaly detection on MSL, PSM, SMD, SWaT, and WADI.

## Files

- specmoe/ contains the model, data loaders, training, scoring, and checkpoint loading.
- configs/ contains one existing configuration per dataset.
- run_specmoe.py provides protocol inspection, training, and detection.
- eval_full.py provides segmented evaluation.
- metrics.py and get_f1_score.py provide evaluation metrics.
- requirements.txt contains runtime dependencies.

## Data protocol and model settings

All five datasets use the same chronological 50/10/40 allocation before window construction. Each independent loader source of N points contributes floor(N/2) training points, floor(N/10) validation points, and the remaining test points, in that order. The original train/test filenames identify input sources; both are repartitioned independently. Sources, entities and excluded gaps are never concatenated. This explicit implementation uses protocol source_chronological_50_10_40_v1.

Training excludes any window containing an anomalous point. Normalization uses only points covered by eligible normal training windows. Consequently the amount of data actually used for training can be smaller than 50%. Validation and test observations never fit the scaler or model parameters. A source must support one full training window and at least four points per evaluation partition; otherwise the run reports an error without padding or borrowing neighboring data. Short evaluation windows must also retain the trained patch-group embedding widths, as specified in the manuscript; four points is a lower bound, not a guarantee for every patch configuration. The saved manifest records every source identity, length, interval and evaluation reference; evaluation reconstructs those exact partitions.

This release implements the manuscript allocation for new runs. It does not establish the exact historical partition boundaries or reproduce the manuscript's numerical results. Legacy protocol readers remain only for explicit, versioned compatibility; new training always uses 50/10/40. Changing metadata cannot convert an older checkpoint into this protocol.

All five configurations use adaptive frequency patches, variable-conditioned routing with triangular band priors, contiguous masking, learnable routing temperature, contrastive diversity, and pointwise maximum aggregation across STFT scales. The MSL configuration follows the appendix reference setting, including expert widths [8, 16, 32], mask ratio 0.30, and batch size 64. The other four configurations provide dataset-specific starting values for new runs. The original per-dataset configurations underlying the reported comparisons are unverified.

## Setup

Use a Python environment with PyTorch appropriate for your hardware, then install the runtime requirements.

~~~sh
python -m pip install -r requirements.txt
~~~

The local CPU checks used Python 3.12.4, PyTorch 2.13.0, and NumPy 2.5.1. The original declared NumPy range is retained below 2. A fresh environment satisfying that constraint and CUDA execution require further verification.

## Data preparation

Run commands from this directory. Place data under datasets/, or edit data_root in the chosen YAML. The evaluation option --data_root takes the parent directory of the dataset folders.

| Dataset folder | Required files and format |
| --- | --- |
| MSL | train/<channel>.npy and test/<channel>.npy with 55 features, plus labeled_anomalies.csv containing exactly the included channel IDs. |
| PSM | Numeric headerless train.csv and test.csv with 25 feature columns, plus a single-column test_label.csv. Convert provider CSV headers and timestamps explicitly. |
| SMD | train/, test/, and test_label/ with the 28 machine-<group>-<id>.txt files. Features have 38 columns. |
| SWAT | SWaT_Dataset_Normal_v1_train.csv and SWaT_Dataset_Attack_v0_test.csv, or swat_train.csv and swat_test.csv. Use 51 features and a numeric binary label column, with zero labels in training. |
| WADI | WADI_14days_new.csv and WADI_attackdataLABLE.csv with 127 sensor columns. Attack labels use 1 for normal and -1 for anomalous. |

Preserve entity boundaries and record data conversions. Sources are [Telemanom](https://github.com/khundman/telemanom), [RANSynCoders](https://github.com/eBay/RANSynCoders), [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly), and [iTrust](https://www.sutd.edu.sg/itrust/request-for-datasets/).

## Train and evaluate

The following example uses MSL. Select the matching YAML and lowercase dataset name for the other datasets.

~~~sh
python -B run_specmoe.py protocol --config configs/msl.yaml
python -B run_specmoe.py train --config configs/msl.yaml --device cpu --seed 42 --save_dir checkpoints/msl_seed42
python -B eval_full.py --checkpoint checkpoints/msl_seed42/best_model.pt --dataset msl --data_root datasets --device cpu --strategy mv_sr_rank --n_rounds 3 --output checkpoints/msl_seed42/metrics.json
~~~

Use cuda when the GPU environment is ready. Use a fresh save_dir for each new run. Prepare configuration changes before training and preserve the complete code/configuration snapshot with its checkpoints. Strict loading verifies source fingerprints.

The evaluator writes the 13 metrics and protocol/checkpoint identities into the requested JSON report. Three scoring rounds belong to one evaluation. Three independent training runs require separate seeds and checkpoint directories.

For the manuscript score, use the segmented evaluator above. The separate detect command below produces binary alarms on a single input using the checkpoint-calibrated reconstruction score. Its threshold belongs to that score and cannot be applied to the five-view rank score.

~~~sh
python -B run_specmoe.py detect --data datasets/MSL/test/example.npy --model checkpoints/msl_seed42/best_model.pt --device cpu --output checkpoints/msl_seed42/predictions.npz
~~~

Replace the input path with an existing file. See each command's --help for options. Model training and evaluation should use the recorded data and configuration for the intended experiment.

## Validation and use

All five configurations were checked on CPU. A synthetic-data run exercised training, checkpoint loading, and evaluation with all 13 metrics. The original benchmark run outputs are unavailable. Each dataset is distributed under its provider’s terms.


## Single-class test segments

All test points, including normal-only source tails, participate in the pooled point metrics. Range AUC and VUS are computed only for segments containing both label classes, then weighted by their eligible segment lengths. Undefined per-segment values are recorded as null with zero aggregation weight and an explicit single_label_class reason. If no segment has both classes, evaluation fails explicitly instead of emitting invented values. Event boundaries remain separate. The report identifies this aggregation policy; it is not evidence for how unavailable historical outputs were computed.
