# Five-index metabolic-signature analysis

## Indices
- MBI_z: mean 14-limbic HC-z minus mean 12-cortical HC-z.
- D_SUVR: RMS regional SUVR deviation across 26 ROIs.
- D_MC: RMS metabolic-connectivity deviation across 168 cross edges.
- D_MwC: RMS deviation of 26 target MwC values extracted after all-144-node dTOR normative-SC computation.
- D_metabolic_signature: equal mean of HC-standardized D_SUVR, D_MC, and D_MwC.

## Statistical conventions
- TRD is TRD1. TRD1 versus nonTRD uses the main-analysis OLS model controlling age and sex. Uncorrected p<0.01 is the reporting threshold.
- Active treatment pools iTBS and rTMS and uses paired post-minus-pre t-tests. Uncorrected p<0.05 is the reporting threshold.
- BH-FDR across the five scalar indices is exported for both analyses.
