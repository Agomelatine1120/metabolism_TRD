"""Auto-extracted pipeline step: Main 09 — five global metabolic indexes.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 41
from __future__ import annotations
import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
ROOT = BASE
OUT = OUTPUT_ROOT / '10_metabolic_signature_five_metrics'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BASE / '.matplotlib'))

import matplotlib.pyplot as plt
import seaborn as sns
CLINICAL = CLINICAL_FILE
PET_FILE = INPUT_ROOT / 'core' / 'AAL3_PET_wbSUVR_Table.xlsx'
ROI_FILE = INPUT_ROOT / 'core' / 'corticolimbic_12x14_ROI_definition.csv'
MWC_FILE = MWC_INPUT_FILE
EXPECTED={'HC':23,'nTRD':25,'TRD1':35,'TRD2':104}


def norm_key(value):
    if pd.isna(value):
        return None
    match = re.fullmatch(r'P(\d{3})', str(value).strip())
    if not match:
        raise ValueError(f'Expected a deidentified P### key, received {value!r}')
    return f'P{int(match.group(1)):03d}'

def pet_column_to_key(column):
    match = re.search(r'swsub_(P\d{3})_', str(column))
    return norm_key(match.group(1)) if match else None

def pet_column_to_session(column):
    m=re.search(r'_ses_(\d+)_',str(column)); return f'ses-{m.group(1)}' if m else None

def aal_id(x):
    m=re.match(r'^AAL3_(\d+)_',str(x));return int(m.group(1)) if m else None

def partial_corr_matrix(values,covariates):
    X=np.column_stack([np.ones(len(values)),covariates])
    resid=values-X@(np.linalg.pinv(X)@values)
    return np.clip(np.corrcoef(resid,rowvar=False),-1,1)

def signed_partial_r(t,df):
    return float(np.sign(t)*np.sqrt(t*t/(t*t+df)))

def group_glm(frame,outcome,case,control):
    d=frame.loc[frame.analysis_cohort.isin([case,control]),[outcome,'analysis_cohort','age','sex']].dropna().copy()
    d['group']=d.analysis_cohort.eq(case).astype(float)
    X=sm.add_constant(d[['group','age','sex']],has_constant='add')
    fit=sm.OLS(d[outcome],X).fit()
    ci=fit.conf_int().loc['group']
    return dict(metric=outcome,comparison=f'{case} - {control}',n_case=int((d.group==1).sum()),n_control=int((d.group==0).sum()),case_mean=float(d.loc[d.group==1,outcome].mean()),control_mean=float(d.loc[d.group==0,outcome].mean()),beta=float(fit.params['group']),ci95_low=float(ci.iloc[0]),ci95_high=float(ci.iloc[1]),t=float(fit.tvalues['group']),effect_size_partial_r=signed_partial_r(float(fit.tvalues['group']),fit.df_resid),p_uncorrected=float(fit.pvalues['group']))

def partial_spearman(x,y,cov):
    x=np.asarray(x,float);y=np.asarray(y,float);cov=np.asarray(cov,float)
    ok=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(cov),axis=1)
    x=x[ok];y=y[ok];cov=cov[ok];n=len(x);k=cov.shape[1]
    if n<=k+3:return np.nan,np.nan,n
    rx=stats.rankdata(x);ry=stats.rankdata(y)
    rc=np.column_stack([stats.rankdata(cov[:,j]) for j in range(k)])
    X=np.column_stack([np.ones(n),rc]);M=np.eye(n)-X@np.linalg.pinv(X)
    a=M@rx;b=M@ry
    r=float(np.corrcoef(a,b)[0,1]);r=float(np.clip(r,-.999999,.999999))
    df=n-k-2;t=r*np.sqrt(df/(1-r*r));p=float(2*stats.t.sf(abs(t),df))
    return r,p,n


# Source cell 43
# Clinical
clinical=pd.read_excel(CLINICAL)
clinical['subject_key']=clinical.subject_key.map(norm_key)
for old,new in [('age','age'),('sex','sex'),('education_years','education'),('hdrs17_pre','hdrs_pre'),('hdrs17_post','hdrs_post'),('msm_total','msm_total')]:clinical[new]=pd.to_numeric(clinical[old],errors='coerce')
clinical['hdrs_improvement']=clinical.hdrs_pre-clinical.hdrs_post
clinical['analysis_arm']=clinical.analysis_arm.fillna('').astype(str)
assert clinical.analysis_cohort.value_counts().to_dict()==EXPECTED
meta=clinical[['subject_key','analysis_cohort','analysis_arm','age','sex','education','hdrs_pre','hdrs_post','hdrs_improvement','msm_total']].copy()

# ROI and PET
roi_def=pd.read_csv(ROI_FILE)
cids=roi_def.loc[roi_def.network.eq('cortical'),'AAL3_ID'].astype(int).tolist();lids=roi_def.loc[roi_def.network.eq('limbic'),'AAL3_ID'].astype(int).tolist()
raw=pd.read_excel(PET_FILE)
roi_all=raw.iloc[:,0].astype(str).tolist()
pet_cols=[c for c in raw.columns[1:] if '_ses_01_' in str(c) or '_ses_02_' in str(c)]
pet=raw.set_index(raw.columns[0])[pet_cols].T;pet.columns=roi_all
pet['subject_key']=[pet_column_to_key(c) for c in pet.index];pet['session']=[pet_column_to_session(c) for c in pet.index]
pet=pet.dropna(subset=['subject_key','session']).drop_duplicates(['subject_key','session'])
ROI=[r for r in roi_all if aal_id(r) in set(cids+lids)];CORT=[r for r in ROI if aal_id(r) in cids];LIMB=[r for r in ROI if aal_id(r) in lids]
assert len(CORT)==12 and len(LIMB)==14 and len(ROI)==26
for r in ROI:pet[r]=pd.to_numeric(pet[r],errors='coerce')
pet=meta.merge(pet[['subject_key','session',*ROI]],on='subject_key',how='inner').dropna(subset=ROI+['age','sex']).reset_index(drop=True)

# Cross edge layout
pos={r:i for i,r in enumerate(ROI)};tri=np.triu_indices(len(ROI),1);pair2idx={(int(i),int(j)):k for k,(i,j) in enumerate(zip(*tri))}
cross=[]
for a in CORT:
 for b in LIMB:
  i,j=sorted((pos[a],pos[b]));cross.append((i,j,a,b,pair2idx[(i,j)]))
edge_ids=[x[4] for x in cross];edge_names=[(x[2],x[3]) for x in cross]


# Source cell 45
# Regional age/sex-adjusted HC reference and LOO metrics.
hc=pet[(pet.analysis_cohort=='HC')&(pet.session=='ses-01')].sort_values('subject_key').reset_index(drop=True)
assert len(hc)==23

def _regional_age_sex_normative_z(row, reference):
    ref_design=sm.add_constant(reference[['age','sex']],has_constant='add')
    beta=np.linalg.pinv(ref_design.to_numpy(float))@reference[ROI].to_numpy(float)
    residual=reference[ROI].to_numpy(float)-ref_design.to_numpy(float)@beta
    residual_sd=np.std(residual,axis=0,ddof=1)
    target=np.asarray([1.0,float(row.age),float(row.sex)])
    vals=row[ROI].to_numpy(float)
    return (vals-target@beta)/residual_sd

reg_rows=[]
for _,row in pet.iterrows():
    vals=row[ROI].to_numpy(float)
    ref=hc[hc.subject_key!=row.subject_key] if row.analysis_cohort=='HC' and row.session=='ses-01' else hc
    z=_regional_age_sex_normative_z(row,ref)
    cmean=float(np.mean(vals[:12]));lmean=float(np.mean(vals[12:]))
    reg_rows.append(dict(subject_key=row.subject_key,session=row.session,analysis_cohort=row.analysis_cohort,analysis_arm=row.analysis_arm,MBI_log=float(np.log(lmean/cmean)),MBI_z=float(np.mean(z[12:])-np.mean(z[:12])),D_SUVR=float(np.sqrt(np.mean(z*z)))))
regional_metrics=pd.DataFrame(reg_rows)


# Source cell 47
# MCM all formal subject-sessions with HC LOO reference
ref_vals=hc[ROI].to_numpy(float);ref_cov=hc[['age','sex']].to_numpy(float)
pcc=partial_corr_matrix(ref_vals,ref_cov);den=(1-pcc[tri]**2)/(len(hc)-1);den[np.abs(den)<1e-12]=1e-12
mcm_rows=[]
for _,row in pet.iterrows():
    if row.analysis_cohort=='HC':
        if row.session!='ses-01':continue
        ref=hc[hc.subject_key!=row.subject_key]
        rv=ref[ROI].to_numpy(float);rc=ref[['age','sex']].to_numpy(float)
        pr=partial_corr_matrix(rv,rc);dn=(1-pr[tri]**2)/(len(ref)-1);dn[np.abs(dn)<1e-12]=1e-12
        pert=partial_corr_matrix(np.vstack([rv,row[ROI].to_numpy(float)]),np.vstack([rc,row[['age','sex']].to_numpy(float)]))
        z=((pert-pr)[tri]/dn)[edge_ids]
    else:
        pert=partial_corr_matrix(np.vstack([ref_vals,row[ROI].to_numpy(float)]),np.vstack([ref_cov,row[['age','sex']].to_numpy(float)]))
        z=((pert-pcc)[tri]/den)[edge_ids]
    mcm_rows.append(dict(subject_key=row.subject_key,session=row.session,D_MC=float(np.sqrt(np.mean(z*z)))))
mcm_metrics=pd.DataFrame(mcm_rows)


# Source cell 49
# MwC values extracted for all 26 formal targets after all-144-node dTOR normative-SC computation
mwc=pd.read_csv(MWC_FILE)
mwc['subject_key']=mwc.subject_key.map(norm_key)
mwc=mwc.merge(meta,on=['subject_key'],how='inner',suffixes=('_mwc',''))
mwc['analysis_cohort']=mwc.analysis_cohort
mwc['analysis_arm']=mwc.analysis_arm
mcols=[c for c in mwc.columns if c.startswith('MwC_')]
assert len(mcols)==26
mwc_hc=mwc[(mwc.analysis_cohort=='HC')&(mwc.session=='ses-01')].drop_duplicates('subject_key').reset_index(drop=True)

def _mwc_age_sex_normative_z(row, reference):
    z=np.full(len(mcols),np.nan,dtype=float)
    for idx,column in enumerate(mcols):
        ref=reference[[column,'age','sex']].dropna().astype(float)
        design=sm.add_constant(ref[['age','sex']],has_constant='add')
        fit=sm.OLS(ref[column],design).fit()
        residual_sd=float(fit.resid.std(ddof=1))
        value=float(row[column]) if pd.notna(row[column]) else np.nan
        if np.isfinite(value):
            target=pd.DataFrame({'const':[1.0],'age':[float(row.age)],'sex':[float(row.sex)]})
            z[idx]=(value-float(fit.predict(target[design.columns]).iloc[0]))/residual_sd
    return z

mwc_rows=[]
for _,row in mwc.iterrows():
    ref=mwc_hc[mwc_hc.subject_key!=row.subject_key] if row.analysis_cohort=='HC' and row.session=='ses-01' else mwc_hc
    z=_mwc_age_sex_normative_z(row,ref)
    mwc_rows.append(dict(subject_key=row.subject_key,session=row.session,D_MwC=float(np.sqrt(np.mean(z*z)))))
mwc_metrics=pd.DataFrame(mwc_rows).drop_duplicates(['subject_key','session'])


# Source cell 51
# Merge and HC-normalize burdens by their own LOO HC distributions
metrics=regional_metrics.merge(mcm_metrics,on=['subject_key','session'],how='left').merge(mwc_metrics,on=['subject_key','session'],how='left').merge(meta,on=['subject_key','analysis_cohort','analysis_arm'],how='left',suffixes=('','_meta'))
for domain in ['SUVR','MC','MwC']:
    col=f'D_{domain}'
    h=metrics[(metrics.analysis_cohort=='HC')&(metrics.session=='ses-01')][col].dropna()
    metrics[f'ZD_{domain}']=(metrics[col]-h.mean())/h.std(ddof=1)
metrics['D_Metabolic']=metrics[['ZD_SUVR','ZD_MC','ZD_MwC']].mean(axis=1,skipna=False)
# standardized directional log MBI for distance calculation
hlog=metrics[(metrics.analysis_cohort=='HC')&(metrics.session=='ses-01')].MBI_log.dropna()
metrics['MBI_log_HC_z']=(metrics.MBI_log-hlog.mean())/hlog.std(ddof=1)


# Source cell 53
# Five prespecified scalar indices
metrics["D_metabolic_signature"] = metrics["D_Metabolic"]
metric_specs = [
    ("MBI_z", r"$MBI$", "Cortico-limbic metabolic imbalance"),
    ("D_SUVR", r"$D_{SUVR}$", "Regional metabolic deviation burden"),
    ("D_MC", r"$D_{MC}$", "Metabolic-connectivity deviation burden"),
    ("D_MwC", r"$D_{MwC}$", "Metabolism-weighted centrality deviation burden"),
    ("D_metabolic_signature", r"$D_{metabolic\ signature}$", "Composite metabolic-signature deviation"),
]
test_metrics = [item[0] for item in metric_specs]
metric_display = {item[0]: item[1] for item in metric_specs}

metric_definitions = pd.DataFrame([
    {
        "metric": "MBI_z",
        "display": "MBI",
        "definition": "mean(HC-referenced z in 14 limbic ROIs) - mean(HC-referenced z in 12 cortical ROIs)",
        "formula": "mean(z_limbic) - mean(z_cortical)",
        "direction": "Higher values indicate a stronger limbic-over-cortical metabolic shift.",
    },
    {
        "metric": "D_SUVR",
        "display": "D_SUVR",
        "definition": "Root-mean-square HC-referenced regional SUVR deviation across 26 ROIs.",
        "formula": "sqrt(mean(z_SUVR,i^2)), i=1,...,26",
        "direction": "Higher values indicate a larger regional distance from the HC metabolic profile.",
    },
    {
        "metric": "D_MC",
        "display": "D_MC",
        "definition": "Root-mean-square volcano-Z deviation across 168 predefined cortical-limbic edges.",
        "formula": "sqrt(mean(z_MC,e^2)), e=1,...,168",
        "direction": "Higher values indicate a larger distance from the HC metabolic-connectivity architecture.",
    },
    {
        "metric": "D_MwC",
        "display": "D_MwC",
        "definition": "Root-mean-square HC-referenced deviation of 26 target MwC values extracted after full AAL3-144 dTOR normative-SC computation.",
        "formula": "sqrt(mean(z_MwC,i^2)), i=1,...,26",
        "direction": "Higher values indicate a larger distance from the HC metabolism-weighted centrality profile.",
    },
    {
        "metric": "D_metabolic_signature",
        "display": "D_metabolic signature",
        "definition": "Equal-weight complete-case mean of HC-standardized D_SUVR, D_MC, and D_MwC.",
        "formula": "mean(z(D_SUVR), z(D_MC), z(D_MwC))",
        "direction": "Higher values indicate a larger multimodal metabolic-signature deviation.",
    },
])


def bh_fdr(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        result[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return result


def paired_test(pre, post, metric):
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    valid = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[valid], post[valid]
    delta = post - pre
    test = stats.ttest_rel(post, pre)
    n = len(delta)
    mean_delta = float(delta.mean())
    delta_sd = float(delta.std(ddof=1))
    sem = delta_sd / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean_delta, scale=sem)
    return {
        "metric": metric,
        "n": n,
        "pre_mean": float(pre.mean()),
        "pre_sd": float(pre.std(ddof=1)),
        "post_mean": float(post.mean()),
        "post_sd": float(post.std(ddof=1)),
        "post_minus_pre": mean_delta,
        "post_minus_pre_CI95_low": float(ci_low),
        "post_minus_pre_CI95_high": float(ci_high),
        "paired_t": float(test.statistic),
        "df": n - 1,
        "effect_size_Cohen_dz": float(mean_delta / delta_sd),
        "p_uncorrected": float(test.pvalue),
    }


# Source cell 55
# TRD1 versus nonTRD: covariate-adjusted GLM used in the main discovery analysis.
baseline = metrics.loc[metrics["session"].eq("ses-01")].copy()
group_tests = pd.DataFrame([
    group_glm(baseline, metric, "TRD1", "nTRD") for metric in test_metrics
])
group_tests["q_fdr_BH_5_metrics"] = bh_fdr(group_tests["p_uncorrected"])
group_tests["reported_main_p_lt_0p01"] = group_tests["p_uncorrected"] < 0.01
group_tests = group_tests.sort_values("p_uncorrected").reset_index(drop=True)


# Source cell 57
# Active treatment pooled across the prespecified iTBS and rTMS arms: paired post minus pre.
active = metrics.loc[
    metrics["analysis_cohort"].eq("TRD2")
    & metrics["analysis_arm"].isin(["iTBS", "rTMS"])
].copy()
active_wide = active.pivot(index="subject_key", columns="session", values=test_metrics)
active_prepost_rows = []
for metric in test_metrics:
    active_prepost_rows.append(
        paired_test(
            active_wide[(metric, "ses-01")].to_numpy(float),
            active_wide[(metric, "ses-02")].to_numpy(float),
            metric,
        )
    )
active_prepost = pd.DataFrame(active_prepost_rows)
active_prepost["q_fdr_BH_5_metrics"] = bh_fdr(active_prepost["p_uncorrected"])
active_prepost["reported_main_p_lt_0p05"] = active_prepost["p_uncorrected"] < 0.05
active_prepost = active_prepost.sort_values("p_uncorrected").reset_index(drop=True)

paired_pet_subjects = active_wide.loc[:, [
    ("D_SUVR", "ses-01"), ("D_SUVR", "ses-02")
]].dropna().index
paired_arm_map = meta.drop_duplicates("subject_key").set_index("subject_key")["analysis_arm"]
paired_arm_counts = paired_arm_map.reindex(paired_pet_subjects).value_counts()
sample_counts = pd.DataFrame([
    {"analysis": "TRD1 versus nonTRD", "group": "TRD1", "n": int(baseline.analysis_cohort.eq("TRD1").sum())},
    {"analysis": "TRD1 versus nonTRD", "group": "nonTRD", "n": int(baseline.analysis_cohort.eq("nTRD").sum())},
    {"analysis": "Active baseline PET", "group": "iTBS", "n": int(meta.analysis_arm.eq("iTBS").sum())},
    {"analysis": "Active baseline PET", "group": "rTMS", "n": int(meta.analysis_arm.eq("rTMS").sum())},
    {"analysis": "Active baseline PET", "group": "active pooled", "n": int(meta.analysis_arm.isin(["iTBS", "rTMS"]).sum())},
    {"analysis": "Active paired PET", "group": "iTBS", "n": int(paired_arm_counts.get("iTBS", 0))},
    {"analysis": "Active paired PET", "group": "rTMS", "n": int(paired_arm_counts.get("rTMS", 0))},
    {"analysis": "Active paired PET", "group": "active pooled", "n": int(len(paired_pet_subjects))},
])
assert paired_arm_counts.to_dict() == {"iTBS": EXPECTED_PAIRED_ARMS["iTBS"], "rTMS": EXPECTED_PAIRED_ARMS["rTMS"]}

subject_export_columns = [
    "subject_key", "session", "analysis_cohort", "analysis_arm",
    "age", "sex", "education", "hdrs_pre", "msm_total", *test_metrics,
]
subject_metrics = metrics[subject_export_columns].sort_values(
    ["analysis_cohort", "analysis_arm", "subject_key", "session"], na_position="last"
)

method_notes = pd.DataFrame([
    {"item": "analysis status", "value": "Five predefined scalar indices; main-analysis statistical conventions."},
    {"item": "TRD comparison", "value": "TRD means TRD1 (n=35); comparator is nonTRD/nTRD (n=25)."},
    {"item": "TRD model", "value": "OLS: metric ~ 1 + group(TRD1=1, nonTRD=0) + age + sex; ordinary two-sided p-values."},
    {"item": "TRD reporting", "value": "Uncorrected p<0.01 is the main feature-wise reporting threshold; BH-FDR across five metrics is retained."},
    {"item": "active sample", "value": f"Pooled active treatment: iTBS baseline n={EXPECTED_ARMS['iTBS']} and rTMS baseline n={EXPECTED_ARMS['rTMS']}; each metric uses its available paired sample."},
    {"item": "active model", "value": "Paired two-sided t-test of post minus pre; no covariate is added to the within-subject paired test."},
    {"item": "active reporting", "value": "Uncorrected p<0.05 is the main pre-post reporting threshold; BH-FDR across five metrics is retained for audit."},
    {"item": "effect sizes", "value": "TRD contrast: signed partial r for the group coefficient. Active post-pre: Cohen's dz."},
    {"item": "HC reference", "value": "SUVR and MC use HC n=23 (HC leave-one-out); MwC uses HC n=23 (HC leave-one-out)."},
    {"item": "composite completeness", "value": "D_metabolic_signature requires nonmissing D_SUVR, D_MC, and D_MwC."},
])


# Source cell 59
# Long-format values used by the final raincloud figures and exports.
group_long_rows = []
active_long_rows = []
for metric, title, description in metric_specs:
    frame = baseline.loc[
        baseline.analysis_cohort.isin(["nTRD", "TRD1"]),
        ["subject_key", "analysis_cohort", metric],
    ].dropna().copy()
    frame["group"] = frame.analysis_cohort.map({"nTRD": "nonTRD", "TRD1": "TRD1"})
    group_long_rows.append(frame.assign(metric=metric))

    pair = active_wide.loc[:, [(metric, "ses-01"), (metric, "ses-02")]].dropna().copy()
    pair.columns = ["Pre", "Post"]
    pair["subject_key"] = pair.index
    long = pair.melt(id_vars="subject_key", var_name="time", value_name="value")
    long["metric"] = metric
    active_long_rows.append(long)

group_long = pd.concat(group_long_rows, ignore_index=True)
active_long = pd.concat(active_long_rows, ignore_index=True)


# Source cell 61
xlsx = OUT / "metabolic_signature_five_metrics.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    metric_definitions.to_excel(writer, sheet_name="metric_definitions", index=False)
    sample_counts.to_excel(writer, sheet_name="sample_counts", index=False)
    group_tests.to_excel(writer, sheet_name="TRD1_vs_nonTRD_all", index=False)
    group_tests.loc[group_tests.reported_main_p_lt_0p01].to_excel(writer, sheet_name="TRD1_vs_nonTRD_p001", index=False)
    active_prepost.to_excel(writer, sheet_name="active_postpre_all", index=False)
    active_prepost.loc[active_prepost.reported_main_p_lt_0p05].to_excel(writer, sheet_name="active_postpre_p05", index=False)
    subject_metrics.to_excel(writer, sheet_name="subject_metrics", index=False)
    group_long.to_excel(writer, sheet_name="figure_group_data", index=False)
    active_long.to_excel(writer, sheet_name="figure_active_data", index=False)
    method_notes.to_excel(writer, sheet_name="method_notes", index=False)

for standalone, frame in [
    (OUT / "TRD1_vs_nonTRD_five_metrics.xlsx", group_tests),
    (OUT / "active_postpre_five_metrics.xlsx", active_prepost),
    (OUT / "five_metric_figure_data.xlsx", pd.DataFrame()),
]:
    if standalone.name == "five_metric_figure_data.xlsx":
        with pd.ExcelWriter(standalone, engine="openpyxl") as writer:
            group_long.to_excel(writer, sheet_name="TRD1_vs_nonTRD", index=False)
            active_long.to_excel(writer, sheet_name="active_postpre", index=False)
            group_tests.to_excel(writer, sheet_name="group_statistics", index=False)
            active_prepost.to_excel(writer, sheet_name="paired_statistics", index=False)
            metric_definitions.to_excel(writer, sheet_name="metric_definitions", index=False)
    else:
        with pd.ExcelWriter(standalone, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="all_five_metrics", index=False)
            if "TRD1" in standalone.name:
                frame.loc[frame.reported_main_p_lt_0p01].to_excel(writer, sheet_name="reported_p_lt_0p01", index=False)
            else:
                frame.loc[frame.reported_main_p_lt_0p05].to_excel(writer, sheet_name="reported_p_lt_0p05", index=False)
            metric_definitions.to_excel(writer, sheet_name="metric_definitions", index=False)


# Source cell 63
summary = {
    "metric_count": 5,
    "MwC_provenance": "dTOR-985 group-normative SC; AAL3-144; fixed top-10% 1,030-edge mask; 26 formal ROIs extracted afterward",
    "metrics": test_metrics,
    "TRD_definition": "TRD1",
    "nonTRD_definition": "nTRD",
    "TRD1_n": int(group_tests.n_case.max()),
    "nonTRD_n": int(group_tests.n_control.max()),
    "active_n_available_by_metric": active_prepost.set_index("metric")["n"].astype(int).to_dict(),
    "TRD1_vs_nonTRD_p_lt_0p01": group_tests.loc[group_tests.reported_main_p_lt_0p01, "metric"].tolist(),
    "TRD1_vs_nonTRD_BH_FDR_q_lt_0p05": group_tests.loc[group_tests.q_fdr_BH_5_metrics.lt(0.05), "metric"].tolist(),
    "active_postpre_p_lt_0p05": active_prepost.loc[active_prepost.reported_main_p_lt_0p05, "metric"].tolist(),
    "active_postpre_BH_FDR_q_lt_0p05": active_prepost.loc[active_prepost.q_fdr_BH_5_metrics.lt(0.05), "metric"].tolist(),
}
(OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

readme = """# Five-index metabolic-signature analysis

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
"""
(OUT / "RESULTS_README.md").write_text(readme, encoding="utf-8")


# Source cell 65
expected_files = [
    xlsx,
    OUT / "TRD1_vs_nonTRD_five_metrics.xlsx",
    OUT / "active_postpre_five_metrics.xlsx",
    OUT / "five_metric_figure_data.xlsx",
    OUT / "analysis_summary.json",
    OUT / "RESULTS_README.md",
]
assert len(group_tests) == 5 and len(active_prepost) == 5
assert set(group_tests.metric) == set(test_metrics)
assert set(active_prepost.metric) == set(test_metrics)
# Each model must use every finite complete case; true MwC missingness is not imputed.
observed_group_n = group_tests.set_index("metric")[["n_case", "n_control"]].astype(int)
for metric in test_metrics:
    expected_case_n = int(
        baseline.loc[baseline.analysis_cohort.eq("TRD1"), metric].notna().sum()
    )
    expected_control_n = int(
        baseline.loc[baseline.analysis_cohort.eq("nTRD"), metric].notna().sum()
    )
    assert observed_group_n.loc[metric].to_dict() == {
        "n_case": expected_case_n,
        "n_control": expected_control_n,
    }
assert (observed_group_n >= 20).all().all()
assert active_prepost["n"].gt(0).all()
for path in expected_files:
    assert path.exists() and path.stat().st_size > 0, path

print("Five-index metabolic-signature analysis completed.")
print("Workbook:", xlsx)
print(group_tests.to_string(index=False))
print(active_prepost.to_string(index=False))


# Source cell 67
# PNAS-style five-index raincloud figures
from matplotlib.patches import Patch

RAINCLOUD_SEED = 20260830
RAINCLOUD_GROUP_COLORS = {"nTRD": "#44853E", "TRD1": "#83328D"}
# Exact solid blue/orange sampled from the supplied PNAS reference figure.
RAINCLOUD_TIME_COLORS = {"Pre": "#73BCED", "Post": "#E8A85C"}
# Opaque pastel fills sampled from the supplied PNAS rainclouds.
RAINCLOUD_TIME_FILL_COLORS = {"Pre": "#ACCFE2", "Post": "#E0C5A1"}
RAINCLOUD_GROUP_SPACING = 0.80
RAINCLOUD_POSITIONS = {"left": 0.0, "right": RAINCLOUD_GROUP_SPACING}
RAINCLOUD_P_FONTSIZE = 10.5
RAINCLOUD_YLABEL_FONTSIZE = 11.0
RAINCLOUD_STAR_SIZE = 45

RAINCLOUD_METRICS = [
    {"metric": "MBI_z", "title": r"$MBI$", "ylabel": r"$MBI$"},
    {"metric": "D_SUVR", "title": r"$D_{SUVR}$", "ylabel": r"$D_{SUVR}$"},
    {"metric": "D_MC", "title": r"$D_{MC}$", "ylabel": r"$D_{MC}$"},
    {"metric": "D_MwC", "title": r"$D_{MwC}$", "ylabel": r"$D_{MwC}$"},
    {
        "metric": "D_metabolic_signature",
        "title": r"$D_{metabolic\ signature}$",
        "ylabel": r"$D_{metabolic\ signature}$",
    },
]


def configure_raincloud_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def raincloud_p_text(p_value: float) -> str:
    return r"$P < 0.001$" if p_value < 0.001 else rf"$P = {p_value:.3f}$"


def raincloud_density_y(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread = float(np.ptp(values))
    pad = max(0.12 * spread, np.std(values, ddof=1) * 0.25, 1e-6)
    y_values = np.linspace(values.min() - pad, values.max() + pad, 256)
    density = stats.gaussian_kde(values)(y_values)
    return y_values, density / density.max()


def raincloud_add_bracket(
    ax: plt.Axes, x1: float, x2: float, y: float, height: float, label: str,
) -> None:
    ax.plot(
        [x1, x1, x2, x2], [y, y + height, y + height, y],
        color="#4D4D4D", linewidth=0.75, clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2, y + height * 1.35, label,
        ha="center", va="bottom", fontsize=RAINCLOUD_P_FONTSIZE, color="#333333",
    )


def raincloud_significance_spec(metric: str, comparison: str, p_value: float) -> tuple[str | None, int]:
    # Symbols are determined only by the current analysis p value; no metric has a frozen result.
    if p_value < 0.001:
        return "solid", 3
    if p_value < 0.01:
        return "solid", 2
    if p_value < 0.05:
        return "solid", 1
    if p_value < 0.10:
        return "outline", 1
    return None, 0


def five_metric_raincloud_panel(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    colors: dict[str, str],
    left_key: str,
    right_key: str,
    title: str,
    ylabel: str,
    p_value: float,
    star_style: str | None,
    star_count: int,
    seed: int,
    fill_colors: dict[str, str] | None = None,
    fill_alpha: float = 0.48,
) -> None:
    rng = np.random.default_rng(seed)
    positions = {left_key: RAINCLOUD_POSITIONS["left"], right_key: RAINCLOUD_POSITIONS["right"]}
    for key in [left_key, right_key]:
        position = positions[key]
        values = np.asarray(arrays[key], dtype=float)
        values = values[np.isfinite(values)]
        color = colors[key]
        fill_color = color if fill_colors is None else fill_colors[key]
        cloud_x = position - 0.19 + rng.normal(0, 0.035, len(values))
        ax.scatter(cloud_x, values, s=11, color=color, alpha=0.78, edgecolors="none", zorder=3)

        box = ax.boxplot(
            [values], positions=[position], widths=0.10, patch_artist=True,
            showfliers=False, manage_ticks=False,
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
            boxprops={"edgecolor": "#555555", "linewidth": 0.8}, zorder=4,
        )
        box["boxes"][0].set_facecolor("white")
        box["boxes"][0].set_alpha(0.84)
        ax.scatter(
            [position], [np.mean(values)], marker="D", s=12,
            facecolor=color, edgecolor="white", linewidth=0.45, zorder=5,
        )

        y_values, density = raincloud_density_y(values)
        right = position + 0.06 + 0.31 * density
        ax.fill_betweenx(
            y_values, position + 0.06, right, facecolor=fill_color, alpha=fill_alpha,
            linewidth=0, edgecolor="none", zorder=1,
        )

    all_values = np.concatenate([np.asarray(arrays[left_key]), np.asarray(arrays[right_key])])
    all_values = all_values[np.isfinite(all_values)]
    data_min, data_max = float(all_values.min()), float(all_values.max())
    span = data_max - data_min
    if span == 0:
        span = max(abs(data_max), 1.0)

    bracket_height = 0.025 * span
    bracket_y = data_max + 0.25 * span
    raincloud_add_bracket(
        ax, positions[left_key] - 0.05, positions[right_key] + 0.05,
        bracket_y, bracket_height, raincloud_p_text(p_value),
    )

    if star_style is not None and star_count > 0:
        star_y = data_max + 0.15 * span
        offsets = (np.arange(star_count) - (star_count - 1) / 2) * 0.11
        for key, opposite in ((left_key, right_key), (right_key, left_key)):
            star_color = colors[opposite]
            ax.scatter(
                positions[key] + offsets, np.full(star_count, star_y),
                marker="*", s=RAINCLOUD_STAR_SIZE,
                facecolors=star_color if star_style == "solid" else "none",
                edgecolors=star_color, linewidths=0.9, zorder=8,
            )

    ax.set_xlim(-0.43, positions[right_key] + 0.43)
    ax.set_ylim(data_min - 0.14 * span, data_max + 0.36 * span)
    ax.set_xticks([positions[left_key], positions[right_key]], ["", ""])
    ax.set_ylabel(ylabel, fontsize=RAINCLOUD_YLABEL_FONTSIZE)
    ax.set_title(title, pad=7, fontweight="semibold")
    ax.grid(axis="y", color="#D5D8DC", linewidth=0.55, alpha=0.70)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="x", bottom=False, length=0)
    ax.tick_params(axis="y", length=3)


configure_raincloud_style()

# Figure 1: nTRD versus TRD (TRD uses the TRD1 cohort).
fig, axes = plt.subplots(1, 5, figsize=(12.3448, 3.40), gridspec_kw={"wspace": 0.32})
group_plot_rows = []
group_figure_stats = []
for index, (ax, specification) in enumerate(zip(axes, RAINCLOUD_METRICS)):
    metric = specification["metric"]
    arrays = {
        "nTRD": baseline.loc[baseline.analysis_cohort.eq("nTRD"), metric].dropna().to_numpy(float),
        "TRD1": baseline.loc[baseline.analysis_cohort.eq("TRD1"), metric].dropna().to_numpy(float),
    }
    stat_row = group_tests.loc[group_tests.metric.eq(metric)].iloc[0]
    star_style, star_count = raincloud_significance_spec(
        metric, "nTRD_TRD", float(stat_row.p_uncorrected)
    )
    five_metric_raincloud_panel(
        ax, arrays, RAINCLOUD_GROUP_COLORS, "nTRD", "TRD1",
        specification["title"], specification["ylabel"], float(stat_row.p_uncorrected),
        star_style, star_count, RAINCLOUD_SEED + index,
    )
    group_figure_stats.append({
        "metric": metric, "p_uncorrected_exact": float(stat_row.p_uncorrected),
        "q_fdr_BH_5_metrics": float(stat_row.q_fdr_BH_5_metrics),
        "star_style": star_style, "star_count": star_count,
        "note": "TRD label uses TRD1 cohort",
    })
    for group, values in arrays.items():
        group_plot_rows.extend({"metric": metric, "group": "TRD" if group == "TRD1" else group,
                                "value": value} for value in values)

group_handles = [
    Patch(facecolor=RAINCLOUD_GROUP_COLORS["nTRD"], edgecolor="none", label="nTRD"),
    Patch(facecolor=RAINCLOUD_GROUP_COLORS["TRD1"], edgecolor="none", label="TRD"),
]
fig.legend(
    handles=group_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
    ncol=2, frameon=False, fontsize=8.5, handlelength=1.75, handleheight=1.75,
    columnspacing=2.4, handletextpad=0.55,
)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.86)
group_stem = OUT / "Figure_five_metric_nTRD_TRD_raincloud"
for suffix in [".pdf", ".svg"]:
    fig.savefig(group_stem.with_suffix(suffix), bbox_inches="tight")
fig.savefig(group_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
plt.close(fig)

# Sensitivity figure: the user-prespecified four regional, four edge, and two
# MwC results from the baseline-HDRS-adjusted nTRD-versus-TRD1 model.
SENSITIVITY_RAINCLOUD_FEATURES = [
    {"feature_type": "regional", "roi": "AAL3_078_Putamen_R", "title": "Rt.PUT"},
    {"feature_type": "regional", "roi": "AAL3_077_Putamen_L", "title": "Lt.PUT"},
    {"feature_type": "regional", "roi": "AAL3_080_Pallidum_R", "title": "Rt.PAL"},
    {"feature_type": "regional", "roi": "AAL3_079_Pallidum_L", "title": "Lt.PAL"},
    {
        "feature_type": "edge",
        "feature": ("AAL3_021_Frontal_Med_Orb_L", "AAL3_080_Pallidum_R"),
        "title": "Lt.vmPFC-Rt.PAL",
    },
    {
        "feature_type": "edge",
        "feature": ("AAL3_019_Frontal_Sup_Medial_L", "AAL3_080_Pallidum_R"),
        "title": "Lt.dmPFC-Rt.PAL",
    },
    {
        "feature_type": "edge",
        "feature": ("AAL3_003_Frontal_Sup_2_L", "AAL3_080_Pallidum_R"),
        "title": "Lt.SFG-Rt.PAL",
    },
    {
        "feature_type": "edge",
        "feature": ("AAL3_019_Frontal_Sup_Medial_L", "AAL3_079_Pallidum_L"),
        "title": "Lt.dmPFC-Lt.PAL",
    },
    {"feature_type": "MwC", "roi": "Frontal_Med_Orb_R", "title": "Rt.vmPFC MwC"},
    {"feature_type": "MwC", "roi": "Frontal_Med_Orb_L", "title": "Lt.vmPFC MwC"},
]

fig, axes = plt.subplots(
    1, 10,
    figsize=(12.3448, 3.40),
    gridspec_kw={"wspace": 0.06},
)
sensitivity_raincloud_raw_rows = []
sensitivity_raincloud_stat_rows = []
ntrd_keys = trd1_result["ntrd"]["subject_key"].to_numpy()
trd1_keys = trd1_result["case_rows"]["subject_key"].to_numpy()
sensitivity_mwc_plot_source = pd.read_csv(MWC_DEVIATION_FILE)
sensitivity_mwc_plot_source["subject_key"] = sensitivity_mwc_plot_source["subject_key"].map(norm_key)
sensitivity_mwc_plot_source = sensitivity_mwc_plot_source.loc[
    sensitivity_mwc_plot_source["session"].eq("ses-01")
    & sensitivity_mwc_plot_source["analysis_cohort"].isin(["nTRD", "TRD1"])
].set_index("subject_key")

for panel_position, (ax, specification) in enumerate(
    zip(axes, SENSITIVITY_RAINCLOUD_FEATURES), start=1
):
    if specification["feature_type"] == "regional":
        roi = specification["roi"]
        arrays = {
            "nTRD": trd1_result["ntrd_regional"][roi].to_numpy(float),
            "TRD1": trd1_result["case_regional"][roi].to_numpy(float),
        }
        selected = sensitivity_regional_all.loc[
            sensitivity_regional_all["ROI"].eq(roi)
        ]
        roi_1, roi_2 = roi, ""
    elif specification["feature_type"] == "edge":
        feature = specification["feature"]
        sensitivity_edge_position = CROSS_EDGE_NAMES.index(feature)
        arrays = {
            "nTRD": trd1_result["ntrd_edge"][:, sensitivity_edge_position],
            "TRD1": trd1_result["case_edge"][:, sensitivity_edge_position],
        }
        selected = sensitivity_edge_all.loc[
            sensitivity_edge_all["ROI_1"].eq(feature[0])
            & sensitivity_edge_all["ROI_2"].eq(feature[1])
        ]
        roi_1, roi_2 = feature
    else:
        roi = specification["roi"]
        mwc_column = f"MwC_deviation_{roi}"
        arrays = {
            "nTRD": sensitivity_mwc_plot_source.loc[ntrd_keys, mwc_column].to_numpy(float),
            "TRD1": sensitivity_mwc_plot_source.loc[trd1_keys, mwc_column].to_numpy(float),
        }
        selected = sensitivity_mwc_all.loc[sensitivity_mwc_all["ROI"].eq(roi)]
        roi_1, roi_2 = roi, ""
    if len(selected) != 1:
        raise RuntimeError(f"Sensitivity feature was not uniquely resolved: {specification}")
    stat_row = selected.iloc[0]
    displayed_p = float(stat_row["HDRS_adjusted_p_uncorrected"])
    star_style, star_count = raincloud_significance_spec(
        specification["title"], "nTRD_TRD_sensitivity", displayed_p
    )
    five_metric_raincloud_panel(
        ax,
        arrays,
        RAINCLOUD_GROUP_COLORS,
        "nTRD",
        "TRD1",
        specification["title"],
        "Deviation (HC z)" if panel_position == 1 else "",
        displayed_p,
        star_style,
        star_count,
        RAINCLOUD_SEED + 300 + panel_position,
    )
    if panel_position > 1:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    sensitivity_raincloud_stat_rows.append({
        "panel_position": panel_position,
        "feature_type": specification["feature_type"],
        "feature_label": specification["title"],
        "ROI_1_or_region": roi_1,
        "ROI_2": roi_2,
        "primary_p_uncorrected": float(stat_row["primary_p_uncorrected"]),
        "HDRS_adjusted_p_uncorrected_displayed": displayed_p,
        "HDRS_adjusted_q_fdr_BH_family": float(
            stat_row["HDRS_adjusted_q_fdr_BH_audit_only"]
        ),
        "HDRS_adjusted_n": int(stat_row["HDRS_adjusted_n"]),
        "star_style": star_style,
        "star_count": star_count,
        "model_covariates": "age; sex; baseline HDRS",
    })
    for group_key, group_label, subject_keys in [
        ("nTRD", "nTRD", ntrd_keys),
        ("TRD1", "TRD", trd1_keys),
    ]:
        for subject_key, value in zip(subject_keys, arrays[group_key]):
            sensitivity_raincloud_raw_rows.append({
                "panel_position": panel_position,
                "feature_type": specification["feature_type"],
                "feature_label": specification["title"],
                "ROI_1_or_region": roi_1,
                "ROI_2": roi_2,
                "subject_key": subject_key,
                "group": group_label,
                "deviation": float(value),
            })

sensitivity_group_handles = [
    Patch(facecolor=RAINCLOUD_GROUP_COLORS["nTRD"], edgecolor="none", label="nTRD"),
    Patch(facecolor=RAINCLOUD_GROUP_COLORS["TRD1"], edgecolor="none", label="TRD"),
]
fig.legend(
    handles=sensitivity_group_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=2,
    frameon=False,
    fontsize=8.5,
    handlelength=1.75,
    handleheight=1.75,
    columnspacing=2.4,
    handletextpad=0.55,
)
fig.subplots_adjust(
    left=0.075, right=0.995, bottom=0.16, top=0.86, wspace=0.06
)
sensitivity_raincloud_stem = (
    FIGURE_OUT / "nTRD_TRD_HDRS_sensitivity_top10_raincloud"
)
for suffix in [".pdf", ".svg"]:
    fig.savefig(sensitivity_raincloud_stem.with_suffix(suffix), bbox_inches="tight")
fig.savefig(
    sensitivity_raincloud_stem.with_suffix(".png"),
    dpi=600,
    bbox_inches="tight",
)
plt.close(fig)

sensitivity_raincloud_data_file = (
    FIGURE_OUT / "nTRD_TRD_HDRS_sensitivity_top10_raincloud_data.xlsx"
)
with pd.ExcelWriter(sensitivity_raincloud_data_file, engine="openpyxl") as writer:
    pd.DataFrame(sensitivity_raincloud_raw_rows).to_excel(
        writer, sheet_name="raw_long", index=False
    )
    pd.DataFrame(sensitivity_raincloud_stat_rows).to_excel(
        writer, sheet_name="panel_statistics", index=False
    )
    pd.DataFrame(SENSITIVITY_RAINCLOUD_FEATURES).to_excel(
        writer, sheet_name="panel_order", index=False
    )
    pd.DataFrame([
        {
            "item": "selection",
            "value": "user-specified 4 regional, 4 edge, and 2 MwC results in fixed requested order",
        },
        {
            "item": "displayed model",
            "value": "TRD1 versus nTRD GLM adjusted for age, sex, and baseline HDRS",
        },
        {"item": "figure width", "value": "12.3448 inches; unchanged from nTRD-TRD five-index raincloud"},
        {"item": "panel spacing", "value": "wspace=0.06"},
        {"item": "nTRD color", "value": RAINCLOUD_GROUP_COLORS["nTRD"]},
        {"item": "TRD color", "value": RAINCLOUD_GROUP_COLORS["TRD1"]},
    ]).to_excel(writer, sheet_name="figure_notes", index=False)

for path in [
    sensitivity_raincloud_stem.with_suffix(".png"),
    sensitivity_raincloud_stem.with_suffix(".pdf"),
    sensitivity_raincloud_stem.with_suffix(".svg"),
    sensitivity_raincloud_data_file,
]:
    assert path.exists() and path.stat().st_size > 0, path
assert len(sensitivity_raincloud_stat_rows) == 10

# Figure 2: pooled active treatment (iTBS + rTMS) pre versus post.
fig, axes = plt.subplots(1, 5, figsize=(12.3448, 3.40), gridspec_kw={"wspace": 0.32})
active_plot_rows = []
active_figure_stats = []
for index, (ax, specification) in enumerate(zip(axes, RAINCLOUD_METRICS)):
    metric = specification["metric"]
    pair = active_wide.loc[:, [(metric, "ses-01"), (metric, "ses-02")]].dropna().copy()
    arrays = {
        "Pre": pair[(metric, "ses-01")].to_numpy(float),
        "Post": pair[(metric, "ses-02")].to_numpy(float),
    }
    stat_row = active_prepost.loc[active_prepost.metric.eq(metric)].iloc[0]
    star_style, star_count = raincloud_significance_spec(
        metric, "active_pre_post", float(stat_row.p_uncorrected)
    )
    five_metric_raincloud_panel(
        ax, arrays, RAINCLOUD_TIME_COLORS, "Pre", "Post",
        specification["title"], specification["ylabel"], float(stat_row.p_uncorrected),
        star_style, star_count, RAINCLOUD_SEED + 100 + index,
        fill_colors=RAINCLOUD_TIME_FILL_COLORS, fill_alpha=1.0,
    )
    active_figure_stats.append({
        "metric": metric, "n": int(stat_row.n),
        "p_uncorrected_exact": float(stat_row.p_uncorrected),
        "q_fdr_BH_5_metrics": float(stat_row.q_fdr_BH_5_metrics),
        "star_style": star_style, "star_count": star_count,
        "note": "Symbols are derived from the current exact p value; hollow denotes 0.05 <= p < 0.10.",
    })
    for time, values in arrays.items():
        active_plot_rows.extend({"metric": metric, "time": time, "value": value} for value in values)

active_handles = [
    Patch(facecolor=RAINCLOUD_TIME_COLORS["Pre"], edgecolor="none", label="Pre"),
    Patch(facecolor=RAINCLOUD_TIME_COLORS["Post"], edgecolor="none", label="Post"),
]
fig.legend(
    handles=active_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
    ncol=2, frameon=False, fontsize=8.5, handlelength=1.75, handleheight=1.75,
    columnspacing=2.4, handletextpad=0.55,
)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.86)
active_stem = OUT / "Figure_five_metric_active_pre_post_raincloud"
for suffix in [".pdf", ".svg"]:
    fig.savefig(active_stem.with_suffix(suffix), bbox_inches="tight")
fig.savefig(active_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
plt.close(fig)

raincloud_data_file = OUT / "Figure_five_metric_raincloud_data.xlsx"
with pd.ExcelWriter(raincloud_data_file, engine="openpyxl") as writer:
    pd.DataFrame(group_plot_rows).to_excel(writer, sheet_name="nTRD_TRD_raw", index=False)
    pd.DataFrame(group_figure_stats).to_excel(writer, sheet_name="nTRD_TRD_statistics", index=False)
    pd.DataFrame(active_plot_rows).to_excel(writer, sheet_name="active_pre_post_raw", index=False)
    pd.DataFrame(active_figure_stats).to_excel(writer, sheet_name="active_pre_post_statistics", index=False)
    metric_definitions.to_excel(writer, sheet_name="metric_definitions", index=False)
    pd.DataFrame([
        {"item": "figure width", "value": "12.3448 inches; identical to 03.ipynb six-panel MwC figure"},
        {"item": "nTRD color", "value": RAINCLOUD_GROUP_COLORS["nTRD"]},
        {"item": "TRD color", "value": RAINCLOUD_GROUP_COLORS["TRD1"]},
        {"item": "Pre color", "value": RAINCLOUD_TIME_COLORS["Pre"]},
        {"item": "Post color", "value": RAINCLOUD_TIME_COLORS["Post"]},
        {"item": "Pre violin fill", "value": RAINCLOUD_TIME_FILL_COLORS["Pre"]},
        {"item": "Post violin fill", "value": RAINCLOUD_TIME_FILL_COLORS["Post"]},
        {"item": "TRD naming", "value": "TRD in the figure denotes TRD1"},
        {"item": "borderline symbol", "value": "hollow outlined star for any current 0.05 <= p < 0.10"},
    ]).to_excel(writer, sheet_name="figure_notes", index=False)

raincloud_outputs = [
    group_stem.with_suffix(".png"), group_stem.with_suffix(".pdf"), group_stem.with_suffix(".svg"),
    active_stem.with_suffix(".png"), active_stem.with_suffix(".pdf"), active_stem.with_suffix(".svg"),
    raincloud_data_file,
]
for path in raincloud_outputs:
    assert path.exists() and path.stat().st_size > 0, path
assert len(group_figure_stats) == len(active_figure_stats) == 5
for comparison, rows in [
    ("nTRD_TRD", group_figure_stats),
    ("active_pre_post", active_figure_stats),
]:
    for row in rows:
        expected_style, expected_count = raincloud_significance_spec(
            row["metric"], comparison, row["p_uncorrected_exact"]
        )
        assert row["star_style"] == expected_style
        assert row["star_count"] == expected_count

print("Wrote PNAS-style five-index raincloud figures:")
for path in raincloud_outputs:
    print(" -", path)
