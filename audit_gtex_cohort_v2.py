"""Audit GTEx v8 Whole Blood cohort selection and reproduce the original IFN score.

Place this script in the same folder as:
  GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
  GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt
  GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz

Run: python audit_gtex_cohort_v2.py

This is an audit, not a replacement for the full notebook. It reproduces the
notebook's per-gene z-scoring (z-score log2(TPM+1) across selected samples,
then average across the 17 IFN genes) for both the original site-code selection
and a stricter exact Whole Blood sample-ID selection.
"""
import gzip
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, zscore

SAMPLE_FILE = 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
PHENO_FILE = 'GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt'
GCT_FILE = 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz'
IFN_GENES = [
    'IFI27','IFI44','IFI44L','IFI6','IFIT1','IFIT2','IFIT3','ISG15',
    'MX1','MX2','OAS1','OAS2','OAS3','OASL','RSAD2','SIGLEC1','USP18'
]

def norm_id(x):
    return str(x).replace('.', '-')

def parse_gct_col(col):
    parts = norm_id(col).split('-')
    if len(parts) < 5:
        return None
    return (col, '-'.join(parts[:2]), parts[2], '-'.join(parts[3:5]))

print('Loading GTEx v8 annotation files...')
samples = pd.read_csv(SAMPLE_FILE, sep='\t', dtype=str)
pheno = pd.read_csv(PHENO_FILE, sep='\t', dtype=str)
if not {'SAMPID','SMTSD'}.issubset(samples.columns):
    raise ValueError('Sample annotation is missing SAMPID or SMTSD columns.')
if not {'SUBJID','SEX'}.issubset(pheno.columns):
    raise ValueError('Phenotype annotation is missing SUBJID or SEX columns.')

wb = samples.loc[samples['SMTSD'].eq('Whole Blood')].copy()
wb['SAMPID_NORM'] = wb['SAMPID'].map(norm_id)
wb['SUBJID_PARSED'] = wb['SAMPID_NORM'].str.split('-').str[:2].str.join('-')
wb['SM_ID_PARSED'] = wb['SAMPID_NORM'].str.split('-').str[3:5].str.join('-')
wb = wb.sort_values('SAMPID_NORM').drop_duplicates('SUBJID_PARSED', keep='first')
wb_sample_ids = set(wb['SAMPID_NORM'])
wb_sm_ids = set(wb['SM_ID_PARSED'])
wb_subjects = set(wb['SUBJID_PARSED'])
sex_lookup = pheno.set_index('SUBJID')['SEX'].to_dict()

print('Reading only the GCT header and target gene rows (not loading the whole matrix into RAM)...')
with gzip.open(GCT_FILE, 'rt') as f:
    f.readline()  # GCT version
    f.readline()  # dimensions
    header = f.readline().rstrip('\n').split('\t')
gct_cols = header[2:]
parsed = [x for x in (parse_gct_col(c) for c in gct_cols) if x is not None]
gct = pd.DataFrame(parsed, columns=['col','SUBJID_PARSED','site','SM_ID_PARSED'])
gct['COL_NORM'] = gct['col'].map(norm_id)

# Original notebook logic: identify site codes from matching sample IDs, then
# select all columns whose donor is in the blood manifest and site is one of
# those codes. This can potentially include extra samples from a donor/site.
confirmed = gct[gct['SM_ID_PARSED'].isin(wb_sm_ids)]
site_codes = set(confirmed['site'])
legacy = gct[gct['SUBJID_PARSED'].isin(wb_subjects) & gct['site'].isin(site_codes)].copy()
legacy = legacy.sort_values('col').drop_duplicates('SUBJID_PARSED', keep='first')

# Strict selection: exact sample identifier must appear in Whole Blood manifest.
strict = gct[gct['COL_NORM'].isin(wb_sample_ids)].copy()
strict = strict.sort_values('col').drop_duplicates('SUBJID_PARSED', keep='first')

for name, df in [('ORIGINAL SITE-CODE SELECTION', legacy), ('STRICT EXACT-SAMPLE SELECTION', strict)]:
    df['SEX'] = df['SUBJID_PARSED'].map(sex_lookup)
    df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')
    df.dropna(subset=['SEX'], inplace=True)
    df['SEX'] = df['SEX'].astype(int)
    print(f'{name}: n={len(df)}, male={(df.SEX==1).sum()}, female={(df.SEX==2).sum()}')

legacy_ids = set(legacy['SUBJID_PARSED'])
strict_ids = set(strict['SUBJID_PARSED'])
extra_ids = sorted(legacy_ids - strict_ids)
missing_ids = sorted(strict_ids - legacy_ids)
print(f'\nSubjects in original selection but not strict selection: {len(extra_ids)}')
if extra_ids:
    print('These donor IDs need inspection:', ', '.join(extra_ids))
print(f'Subjects in strict selection but not original selection: {len(missing_ids)}')
if missing_ids:
    print('These donor IDs need inspection:', ', '.join(missing_ids))

# Read only the 17 IFN genes from the compressed expression matrix.
rows = []
with gzip.open(GCT_FILE, 'rt') as f:
    f.readline(); f.readline()
    cols = f.readline().rstrip('\n').split('\t')
    for line in f:
        fields = line.rstrip('\n').split('\t')
        if len(fields) > 1 and fields[1] in set(IFN_GENES):
            rows.append(fields)
expr = pd.DataFrame(rows, columns=cols)
expr = expr.drop_duplicates('Description').set_index('Description')
missing_genes = [g for g in IFN_GENES if g not in expr.index]
if missing_genes:
    print('WARNING: missing IFN genes:', ', '.join(missing_genes))
used_genes = [g for g in IFN_GENES if g in expr.index]

# Reproduce original notebook's exact score construction separately per cohort.
def score_cohort(df, label):
    df = df.copy()
    df['SEX'] = pd.to_numeric(df['SUBJID_PARSED'].map(sex_lookup), errors='coerce')
    df = df.dropna(subset=['SEX']).copy()
    df['SEX'] = df['SEX'].astype(int)
    mat = expr.loc[used_genes, df['col'].tolist()].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    log_mat = np.log2(mat + 1)
    gene_z = zscore(log_mat, axis=1, ddof=0, nan_policy='omit')
    scores = np.nanmean(gene_z, axis=0)
    df['IFN'] = scores
    df = df.dropna(subset=['IFN'])
    male = df.loc[df['SEX'].eq(1), 'IFN'].to_numpy()
    female = df.loc[df['SEX'].eq(2), 'IFN'].to_numpy()
    pooled_sd = np.sqrt(((len(female)-1)*np.var(female, ddof=1) + (len(male)-1)*np.var(male, ddof=1)) / (len(female)+len(male)-2))
    d = (female.mean()-male.mean())/pooled_sd
    p2 = mannwhitneyu(female, male, alternative='two-sided').pvalue
    p1 = mannwhitneyu(female, male, alternative='greater').pvalue
    print(f'\n{label} — ORIGINAL NOTEBOOK SCORING')
    print(f'n={len(df)} (male={len(male)}, female={len(female)})')
    print(f'mean male={male.mean():.6f}; mean female={female.mean():.6f}; difference F-M={female.mean()-male.mean():.6f}')
    print(f"Cohen's d={d:.6f}; Mann–Whitney two-sided p={p2:.8g}; one-sided F>M p={p1:.8g}")
    out = df[['col','SUBJID_PARSED','site','SM_ID_PARSED','SEX','IFN']].copy()
    out.to_csv(f'GTEx_{label.lower().replace(" ", "_").replace("—", "")}_audit.csv', index=False)
    return out

legacy_scores = score_cohort(legacy, 'ORIGINAL SITE-CODE SELECTION')
strict_scores = score_cohort(strict, 'STRICT EXACT-SAMPLE SELECTION')

# Save a concise membership comparison to inspect the extra/missing subjects.
comparison = pd.DataFrame({
    'SUBJID_PARSED': sorted(legacy_ids | strict_ids)
})
comparison['in_original_site_code_selection'] = comparison['SUBJID_PARSED'].isin(legacy_ids)
comparison['in_strict_exact_sample_selection'] = comparison['SUBJID_PARSED'].isin(strict_ids)
comparison['SEX'] = comparison['SUBJID_PARSED'].map(sex_lookup)
comparison.to_csv('GTEx_cohort_membership_comparison.csv', index=False)
print('\nSaved: GTEx_cohort_membership_comparison.csv and two sample-level audit CSVs.')
print('Please share the printed cohort counts and p-values before changing any manuscript text.')
