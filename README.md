# Strict Validation Notebooks for UAV Propeller Fault Diagnosis

This repository contains strict validation notebooks and plotting utilities for UAV propeller fault diagnosis using physics-informed time-series deep learning.

The workflow is designed to reduce over-optimistic performance caused by random window-level splitting and overlapping time windows. It uses non-overlapping windows, source-file-aware splitting, group-wise validation, and label-shuffle leakage testing.

---

## Input Files

### Main input dataset

```text
merged_physics.csv
```

### Required helper file

```text
uav_strict_utils.py
```

Keep `merged_physics.csv` and `uav_strict_utils.py` in the same folder as the notebooks.

---

## Dataset Requirement

The input file `merged_physics.csv` should contain:

- measured UAV propulsion features,
- physics-informed derived features,
- a class label column, preferably named `class_label`,
- a source-file or experimental-run column, preferably named `source_file`.

The `source_file` column is important because it is used for leakage-controlled validation.

---

## Notebooks

### Notebook 01: Strict Run-Wise Validation

```text
01_Strict_Runwise_NonOverlap_All_Models.ipynb
```

This notebook performs strict run-wise validation using:

- non-overlapping time windows,
- idle-row removal,
- source-file-level train, validation, and test splitting.

This directly addresses over-optimistic 100% accuracy that may occur when overlapping windows or random window-level splits are used.

---

### Notebook 02: Stratified Group K-Fold Validation

```text
02_Stratified_Group_KFold_All_Models.ipynb
```

This notebook performs Stratified Group K-Fold validation using `source_file` as the group variable.

It reports:

- fold-level metrics,
- mean accuracy,
- standard deviation of accuracy,
- mean and standard deviation of precision, recall, and F1-score.

This validation strategy gives a more reliable estimate of generalization across unseen experimental runs.

---

### Notebook 03: Label-Shuffle Leakage Test

```text
03_Label_Shuffle_Leakage_Test_All_Models.ipynb
```

This notebook performs a label-shuffle leakage test.

In this test:

- the input features are kept unchanged,
- the training labels are randomly shuffled,
- the model is trained using the shuffled labels.

If the model still achieves high accuracy under shuffled labels, possible data leakage or source-specific artifacts may exist. Near-chance performance indicates that the evaluation pipeline does not show strong evidence of label leakage.

---

## Deep Learning Models Included

Each notebook evaluates the following models:

1. 1D-CNN
2. LSTM
3. CNN-LSTM
4. TCN
5. Transformer
6. Multimodal Transformer

---

## Important Settings

The main settings used in the notebooks are:

```python
WINDOW_SIZE = 50
STRIDE = 50
FILTER_ACTIVE_REGION = True
ACTIVE_RPM_MIN = 500
```

### Meaning of settings

- `WINDOW_SIZE = 50`: each time-window contains 50 sequential samples.
- `STRIDE = 50`: windows are non-overlapping.
- `FILTER_ACTIVE_REGION = True`: idle or inactive motor rows are removed.
- `ACTIVE_RPM_MIN = 500`: only rows with RPM greater than 500 are retained.

---

## Output Files

Each notebook generates model-wise output folders containing:

```text
metrics.csv
predictions.csv
classification_report.txt
confusion_matrix.pdf
metrics_bar.pdf
training_curve.pdf
trained_model.keras
summary CSV files
```

The output files are useful for reporting model performance, plotting confusion matrices, comparing models, and preparing manuscript results.

---

## Class-Wise Physical Signal Plotting for UAV Propeller Dataset

This repository also includes code for generating class-wise physical signal plots. These plots help explain how normal, bent, and cracked propellers differ in propulsion behaviour before deep learning classification.

Recommended plots include:

1. RPM vs thrust
2. RPM vs torque
3. Vibration vs RPM
4. Current vs thrust
5. Thrust-per-watt distribution

These plots can be used in the paper to support the physical interpretation of UAV propeller faults.

---

## Suggested Plotting Script

A plotting script may be named:

```text
plot_classwise_physical_signals.py
```

It should read:

```text
merged_physics.csv
```

and generate PDF figures such as:

```text
Fig_RPM_vs_Thrust.pdf
Fig_RPM_vs_Torque.pdf
Fig_Vibration_vs_RPM.pdf
Fig_Current_vs_Thrust.pdf
Fig_Thrust_per_Watt_Distribution.pdf
All_Classwise_Physical_Signal_Plots.pdf
```

---

## Purpose of Physical Signal Plots

The class-wise plots help demonstrate that propeller faults disturb the normal propulsion relationships among RPM, thrust, torque, current, vibration, and efficiency.

Typical physical interpretations are:

- bent propellers disturb thrust generation due to blade deformation,
- cracked propellers may produce higher vibration due to structural discontinuity and imbalance,
- faulty propellers may require different current levels to generate similar thrust,
- thrust-per-watt distribution indicates changes in propulsion efficiency,
- RPM-thrust and RPM-torque relationships provide physically meaningful separability among propeller conditions.

---

## Recommended Repository Structure

```text
UAV-Propeller-Fault-Diagnosis/
│
├── README.md
├── merged_physics.csv
├── uav_strict_utils.py
│
├── 01_Strict_Runwise_NonOverlap_All_Models.ipynb
├── 02_Stratified_Group_KFold_All_Models.ipynb
├── 03_Label_Shuffle_Leakage_Test_All_Models.ipynb
│
├── plot_classwise_physical_signals.py
│
├── Fig/
│   ├── Fig_RPM_vs_Thrust.pdf
│   ├── Fig_RPM_vs_Torque.pdf
│   ├── Fig_Vibration_vs_RPM.pdf
│   ├── Fig_Current_vs_Thrust.pdf
│   └── Fig_Thrust_per_Watt_Distribution.pdf
│
└── Results/
    ├── Strict_Runwise/
    ├── Stratified_Group_KFold/
    └── Label_Shuffle_Test/
```

---

## Notes for Reliable Validation

If the number of test windows is still small, collect more complete experimental runs for each class.

This is especially important for the cracked propeller class, because severe cracks can generate high vibration and may limit safe data acquisition.

For publication-quality reporting, avoid relying only on a single strict split. Report:

- strict run-wise validation,
- Stratified Group K-Fold validation,
- label-shuffle leakage test,
- class-wise precision, recall, and F1-score,
- confusion matrices,
- mean and standard deviation across folds.

---

## Citation or Acknowledgement

If this repository is used for a paper, mention that the validation pipeline uses source-file-aware splitting and non-overlapping time windows to reduce leakage and over-optimistic model evaluation.

---

## License

Add an appropriate license file before public release, such as:

```text
MIT License
```

or use your institution-approved license.

---

## Contact

For questions, issues, or collaboration, please contact the repository maintainer.
