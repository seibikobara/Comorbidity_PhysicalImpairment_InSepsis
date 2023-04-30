
## Lack of Association Between Physical Impairment following sepsis at Hospital Discharge and Pre-existing Comorbidities Prior to ICU Admission

This repository contains codes for analysis and key figures for the manuscript "Lack of Association Between Physical Impairment following sepsis at Hospital Discharge and Pre-existing Comorbidities Prior to ICU Admission" by Seibi Kobara et al. 


### Estimation of an association between pre-existing comorbidity and physical impairment at hospital discharge among patients with sepsis

* analysis.R was used for the estimation

### Estimation of propensity score (PS) of attrition factors including death within 72 hours from ICU admission and death in a hospital after 72 hours from ICU admission

* psEstimation.py was used for PS estimation
* lovePlot_violin_ANOVA.R was used to generate love plots and violin plots from the results of PS estimation and model performance using AUROC. ANOVA was performed to compare AUROC values across models.