# Pseudonymization and data-use note

## Identifier scheme

- All cross-file identifiers use `P001`–`P168`.
- Public keys were assigned only within the analyzed roster.
- PET and MwC records are strict subsets of those 168 keys.
- `inputs/core/analysis_population_roster.csv` locks the exact participants and
  available PET/MwC sessions used by this package.
- The private original-to-public crosswalk is not bundled or referenced.

## Data minimization

- The clinical workbook contains only 11 fields used by the analyses.
- The PET workbook contains only 144 retained atlas regions and analyzed
  participant-sessions.
- The MwC bundle contains only the GCMI/PET arrays and metadata required for
  MwC calculation.
- Subject/session membership is enforced at input-file level. The workflow does
  not start from a broader participant export and remove ineligible records at
  analysis time.
- Original research codes, header initials, source-system identifiers,
  machine-specific paths, nonessential clinical columns, and quality-control
  exclusion records are absent.

## Residual disclosure risk

The package contains participant-level demographic, clinical, PET, and GCMI
data. These data are pseudonymized, not anonymous. The repository should remain
private unless release has been separately approved under the relevant ethics,
consent, institutional, and data-use requirements.
