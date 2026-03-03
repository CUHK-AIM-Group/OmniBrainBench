# OmniBrainBench-Extend (Patient-Level Subset)

`OmniBrainBench-Extend` is a **patient-level** extension of **OmniBrainBench**. This subset groups **multiple question types for the same patient** (and potentially multiple visits), covering the **five clinical phases** of the OmniBrainBench workflow:
**(1) Anatomical & Imaging Assessment → (2) Lesion Identification & Localization → (3) Diagnostic Synthesis → (4) Prognostic Judgment & Risk Forecasting → (5) Therapeutic Cycle Management.**

In this folder, we provide tutorials to generate **OmniBrainBench-Extend labels** from **ADNI** and **UK Biobank (UKB)**.

---

## TODO

- [x] ADNI
- [ ] UKB
- [ ] Generate VQA pairs (patient-level multi-task)

---

## 1. Data Preparation (ADNI example)

Organize ADNI data into a single root directory where **each subject has its own folder**, containing at least:

- `T1.nii` (T1-weighted MRI)
- `FLAIR.nii` (FLAIR MRI)

Example structure:

```text
ADNI_ROOT/
  sub-0001/
    T1.nii
    FLAIR.nii
  sub-0002/
    T1.nii
    FLAIR.nii
  ...
```

### 1.1 Convert ADNI DICOM (.dcm) to NIfTI (.nii / .nii.gz)

ADNI imaging is typically provided in **DICOM** format. Convert DICOM to NIfTI using **dcm2niix**:

- Tool: https://github.com/rordenlab/dcm2niix

Example conversion:

```bash
dcm2niix -z y -o /path/to/output_nifti /path/to/input_dicom_folder
```

After conversion, **rename and place** the outputs into each subject folder as `T1.nii` and `FLAIR.nii` (use `.nii.gz` if preferred, but keep naming consistent).

> Note: ADNI download/access requires compliance with ADNI data usage agreements. Do not redistribute raw images.

---

## 2. Label Generation (ADNI)

### 2.1 Anatomical Structure Identification (FreeSurfer-based)

We use ADNI’s official FreeSurfer results:

- Dataset: **UCSF - Cross-Sectional FreeSurfer (7.x) [ADNI1, GO, 2, 3, 4]**
- CSV file: `UCSFFSX7_03Mar2026.csv`

Steps:

1. **Filter high-quality samples**:
   - Keep only rows with `OVERALLQC = 1` (excellent quality).
2. **Match subjects** to your MRI raw images (your `sub-xxxx` folders).
3. **Generate segmentation** with FreeSurfer (run on T1):

```bash
recon-all -wsatlas -wsless -all -s sub-xxx -i T1.nii
```

Convert segmentations to NIfTI for downstream checks:

```bash
mri_convert sub-xxx/mri/aseg.mgz aseg.nii.gz
mri_convert sub-xxx/mri/aparc+aseg.mgz aparc+aseg.nii.gz
```

**Sanity check suggestion**: randomly sample a subject, use the segmentation output to test whether the expected anatomical structures are identifiable; if something looks abnormal, validate against regional volume/thickness values in `UCSFFSX7_03Mar2026.csv`.

> FreeSurfer requires a valid license and installation: https://surfer.nmr.mgh.harvard.edu/

---

### 2.2 Imaging Modality Identification

No extra processing is needed. The modality can be determined from the file naming / sequence type:

- `T1.nii` → T1W
- `FLAIR.nii` → FLAIR
- `PD.nii` → PD
- ...

---

### 2.3 Disease / Abnormality Diagnosis (Cognitive Status)

Use the ADNI table:

- Table: **Diagnostic Summary [ADNI1, GO, 2, 3, 4]**
- CSV file: `DXSUM_03Mar2026.csv`
- Location in ADNI: `ADNI → Study Files → Assessments → Diagnosis`

Label mapping (from ADNI):

- `1 = CN` (Cognitively Unimpaired)
- `2 = MCI` (Mild Cognitive Impairment)
- `3 = Dementia`

For OmniBrainBench-Extend, you can derive patient-level diagnosis labels such as:
- **Any cognitive impairment**: `DIAGNOSIS ∈ {2, 3}`
- **Dementia**: `DIAGNOSIS = 3`

---

### 2.4 Lesion Localization (WMH Segmentation via MARS-WMH nnU-Net)

We generate **white matter hyperintensity (WMH)** segmentation using the **MARS-WMH nnU-Net** (Docker version).

```bash
# 1. Pull the container image into your local registry
docker pull ghcr.io/miac-research/wmh-nnunet:latest
docker tag ghcr.io/miac-research/wmh-nnunet:latest mars-wmh-nnunet:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--gpus all")
docker run --rm --gpus all -v $(pwd):/data mars-wmh-nnunet:latest --flair /data/FLAIR.nii --t1 /data/T1w.nii
```

---

### 2.5 Risk Forecasting & Treatment-Related Labels (Longitudinal)

For **risk assessment / prognosis** (and treatment-related questions), we use **multiple visits per subject** from `DXSUM_03Mar2026.csv`:

1. Group records by **subject ID**.
2. Identify subjects with **multiple time points**.
3. Use changes in `DIAGNOSIS` across visits to label future risk, e.g.:
   - CN → MCI: elevated risk of cognitive decline
   - MCI → Dementia: progression risk
   - Stable CN: lower short-term risk (relative)

---

## Notes

- This repo provides **label-generation logic and tutorials**; it does **not** redistribute ADNI/UKB raw data.
- UKB support and patient-level VQA generation are under active development (see TODO).