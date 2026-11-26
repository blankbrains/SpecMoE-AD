# SpecMoE-AD

Spectrum-aware mixture-of-experts for multivariate time-series anomaly detection on MSL, PSM, SMD, SWaT, and WADI.

## Files

- specmoe/ contains the model, data loaders, training, scoring, and checkpoint loading.
- configs/ contains one configuration per dataset.
- run_specmoe.py provides protocol inspection, training, and detection.
- eval_full.py provides segmented evaluation.
- metrics.py and get_f1_score.py provide evaluation metrics.
- requirements.txt contains runtime dependencies.

## Data protocol and model settings

All five datasets use the same chronological 50/10/40 allocation before window construction. Each independent loader source of N points contributes floor(N/2) training points, floor(N/10) validation points, and the remaining test points, in that order. The original train/test filenames identify input sources; both are repartitioned independently. Sources, entities and excluded gaps are never concatenated.

Training excludes any window containing an anomalous point. Normalization uses only points covered by eligible normal training windows. Consequently the amount of data actually used for training can be smaller than 50%. Validation and test observations never fit the scaler or model parameters. A source must support one full training window and at least four points per evaluation partition; otherwise the run reports an error without padding or borrowing neighboring data. Short evaluation windows must also retain the trained patch-group embedding widths, as specified in the manuscript; four points is a lower bound, not a guarantee for every patch configuration. The saved manifest records every source identity, length, interval and evaluation reference; evaluation reconstructs those exact partitions.

The configurations support new training and evaluation runs. The MSL configuration follows the reference setting in the paper’s appendix. The other four configurations provide dataset-specific starting points. The original pointwise scores underlying the benchmark table are unavailable, as stated in the paper.

All five configurations use adaptive frequency patches, variable-conditioned routing with triangular band priors, contiguous masking, learnable routing temperature, contrastive diversity, and pointwise maximum aggregation across STFT scales. For MSL, the expert widths are 8, 16, and 32, the mask ratio is 0.30, and the batch size is 64.

## Setup

Use a Python environment with PyTorch appropriate for your hardware, then install the runtime requirements.

~~~sh
python -m pip install -r requirements.txt
~~~

Use the dependency ranges in requirements.txt and a PyTorch build that matches your CPU or CUDA environment.

## Data preparation

Run commands from this directory. Place data under datasets/, or edit data_root in the chosen YAML. The evaluation option --data_root takes the parent directory of the dataset folders.

| Dataset folder | Required files and format |
| --- | --- |
| MSL | train/<channel>.npy and test/<channel>.npy with 55 features, plus labeled_anomalies.csv containing exactly the included channel IDs. |
| PSM | Numeric headerless train.csv and test.csv with 25 feature columns, plus a single-column test_label.csv. Convert provider CSV headers and timestamps explicitly. |
| SMD | train/, test/, and test_label/ with the 28 machine-<group>-<id>.txt files. Features have 38 columns. |
| SWAT | SWaT_Dataset_Normal_v1_train.csv and SWaT_Dataset_Attack_v0_test.csv, or swat_train.csv and swat_test.csv. Use 51 features and a numeric binary label column, with zero labels in training. |
| WADI | WADI_14days_new.csv and WADI_attackdataLABLE.csv with 127 sensor columns. Attack labels use 1 for normal and -1 for anomalous. |

Preserve entity boundaries during conversion and obtain each dataset under its provider’s terms. Sources are [Telemanom](https://github.com/khundman/telemanom), [RANSynCoders](https://github.com/eBay/RANSynCoders), [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly), and [iTrust](https://www.sutd.edu.sg/itrust/request-for-datasets/).

## Train and evaluate

The commands below train and evaluate the MSL configuration. Select the matching YAML and lowercase dataset name for the other datasets.

~~~sh
python -B run_specmoe.py protocol --config configs/msl.yaml
python -B run_specmoe.py train --config configs/msl.yaml --device cpu --seed 42 --save_dir checkpoints/msl_seed42
python -B eval_full.py --checkpoint checkpoints/msl_seed42/best_model.pt --dataset msl --data_root datasets --device cpu --strategy mv_sr_rank --n_rounds 3 --output checkpoints/msl_seed42/metrics.json
~~~

For GPU execution, set `--device cuda`. Use a separate checkpoint directory for each training run. Keep the training configuration and source files with the checkpoint. The loader checks that they match before evaluation.

The evaluator writes the 13 metrics and protocol/checkpoint identities into the requested JSON report. Three scoring rounds belong to one evaluation. Three independent training runs require separate seeds and checkpoint directories.

For the manuscript score, use the segmented evaluator above. Set the shell variable `MSL_TEST_FILE` to the path of the MSL channel array to score before running detection. The separate detect command below produces binary alarms on a single input using the checkpoint-calibrated reconstruction score. Its threshold belongs to that score and cannot be applied to the five-view rank score.

~~~sh
python -B run_specmoe.py detect --data "$MSL_TEST_FILE" --model checkpoints/msl_seed42/best_model.pt --device cpu --output checkpoints/msl_seed42/predictions.npz
~~~

See each command's --help for options. Model training and evaluation should use the recorded data and configuration for the intended experiment.

## Single-class test segments

All test points, including normal-only source tails, participate in the pooled point metrics. Range AUC and VUS are computed only for segments containing both label classes, then weighted by their eligible segment lengths. Undefined per-segment values are recorded as null with zero aggregation weight and an explicit single_label_class reason. If no segment has both classes, evaluation exits with an error. Event boundaries remain separate. The JSON report records the aggregation policy and the weight assigned to each segment.
