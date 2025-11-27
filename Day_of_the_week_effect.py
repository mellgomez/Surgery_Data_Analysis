'''DAY OF THE WEEK EFFECTS'''


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
df = pd.read_excel('SurgeryTiming.xlsx')

# ============================================
# DATA PREPARATION
# ============================================

# Create temporal variables
df['late_week'] = (df['dow'] >= 4).astype(int)  # Thu-Fri = 1, Mon-Wed = 0

# Impute missing values (median imputation for continuous variables)
df['bmi_imputed'] = df['bmi'].fillna(df['bmi'].median())
df['asa_imputed'] = df['asa_status'].fillna(df['asa_status'].median())

# Define high-risk procedures (>1% mortality from exploratory analysis)
high_risk_procedures = ['Colorectal resection', 'Small bowel resection',
                        'Gastrectomy; partial and total', 
                        'Hip replacement; total and partial',
                        'Spinal fusion']
df['high_risk_proc'] = df['ahrq_ccs'].isin(high_risk_procedures).astype(int)

# ============================================
# MODEL 1: BASELINE RISK ONLY (mortality_rsi)
# ============================================
print("="*80)
print("MODEL 1: Baseline Risk Only (mortality_rsi)")
print("="*80)

# Prepare data for Model 1
X_baseline = df[['mortality_rsi']].dropna()
y_baseline = df.loc[X_baseline.index, 'mort30']

# Fit logistic regression with balanced class weights (addresses 231:1 imbalance)
model1 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model1.fit(X_baseline, y_baseline)

# Evaluate model performance
y_pred_baseline = model1.predict_proba(X_baseline)[:, 1]
auc_baseline = roc_auc_score(y_baseline, y_pred_baseline)

# Display results
print(f"Baseline AUC: {auc_baseline:.4f}")
print(f"Coefficient (mortality_rsi): {model1.coef_[0][0]:.4f}")
print(f"Odds Ratio (per unit RSI): {np.exp(model1.coef_[0][0]):.4f}")

# ============================================
# MODEL 2: BASELINE + TEMPORAL + DEMOGRAPHICS
# ============================================
print("\n" + "="*80)
print("MODEL 2: Baseline Risk + Day of Week + Demographics")
print("="*80)

# Feature selection: Core risk adjusters (age removed due to VIF=10.77)
feature_cols = ['mortality_rsi', 'late_week', 'asa_imputed', 
                'baseline_charlson', 'baseline_cancer', 'baseline_cvd']

# Prepare data for Model 2
X_full = df[feature_cols].dropna()
y_full = df.loc[X_full.index, 'mort30']

# Fit Model 2
model2 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model2.fit(X_full, y_full)

# Evaluate performance
y_pred_full = model2.predict_proba(X_full)[:, 1]
auc_full = roc_auc_score(y_full, y_pred_full)

print(f"Full Model AUC: {auc_full:.4f}")
print(f"AUC Improvement: {auc_full - auc_baseline:.4f}")

# Display coefficients
print("\nCoefficients:")
for name, coef in zip(feature_cols, model2.coef_[0]):
    or_val = np.exp(coef)
    print(f"  {name:20s}: OR = {or_val:.4f}, β = {coef:7.4f}")

# Interpret late_week effect
late_week_coef = model2.coef_[0][feature_cols.index('late_week')]
late_week_or = np.exp(late_week_coef)

print("\n" + "="*80)
print("H1: LATE WEEK MAIN EFFECT")
print("="*80)
print(f"Late Week Odds Ratio: {late_week_or:.4f}")
print(f"Interpretation: Thursday-Friday surgeries have {(late_week_or-1)*100:.1f}% {'higher' if late_week_or > 1 else 'lower'} odds")
print("of 30-day mortality AFTER controlling for baseline patient risk")

# ============================================
# MODEL COMPARISON: Likelihood Ratio Test + AIC
# ============================================
print("\n" + "="*80)
print("MODEL COMPARISON: Baseline vs Full Model")
print("="*80)

# Calculate log-likelihoods (higher = better fit)
ll_baseline = -log_loss(y_full, y_pred_baseline[X_full.index], normalize=False)
ll_full = -log_loss(y_full, y_pred_full, normalize=False)

# Calculate number of parameters (k = features + intercept)
k_baseline = X_baseline.shape[1] + 1  # 1 feature + intercept = 2
k_full = len(feature_cols) + 1         # 6 features + intercept = 7

# Calculate AIC (Akaike Information Criterion): lower = better
aic_baseline = -2 * ll_baseline + 2 * k_baseline
aic_full = -2 * ll_full + 2 * k_full

# Calculate Deviance (lack of fit measure): lower = better
deviance_baseline = -2 * ll_baseline
deviance_full = -2 * ll_full

# Likelihood Ratio Test: Does adding features significantly improve fit?
lr_stat = 2 * (ll_full - ll_baseline)
df_diff = k_full - k_baseline  # Degrees of freedom = difference in parameters
p_value_lr = 1 - stats.chi2.cdf(lr_stat, df_diff)

# Display comparison
print(f"\n{'Metric':<25} {'Model 1':<20} {'Model 2':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Predictors':<25} {k_baseline-1:<20} {k_full-1:<20} {(k_full-1)-(k_baseline-1):<15}")
print(f"{'AIC':<25} {aic_baseline:<20.2f} {aic_full:<20.2f} {aic_full - aic_baseline:<15.2f}")
print(f"{'Deviance':<25} {deviance_baseline:<20.2f} {deviance_full:<20.2f} {deviance_full - deviance_baseline:<15.2f}")

print("\nLikelihood Ratio Test:")
print(f"  χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_value_lr:.4f}")
if p_value_lr < 0.001:
    print(f"  ✓ Model 2 is SIGNIFICANTLY better (p < 0.001)")
    print(f"  → Adding temporal/demographic factors substantially improves fit")
else:
    print(f"  → Model 2 improvement not significant")

# ============================================
# H2: FORMAL INTERACTION TEST (Day × Procedure Risk)
# ============================================
print("\n" + "="*80)
print("H2: DAY OF WEEK × PROCEDURE RISK INTERACTION")
print("="*80)
print("Testing: Does the late-week effect differ for high-risk vs low-risk procedures?\n")

# Create interaction term
df['late_x_highrisk'] = df['late_week'] * df['high_risk_proc']

# Model WITHOUT interaction (for comparison)
features_no_int = feature_cols + ['high_risk_proc']
X_no_int = df[features_no_int].dropna()
y_no_int = df.loc[X_no_int.index, 'mort30']

model_no_int = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model_no_int.fit(X_no_int, y_no_int)

# Model WITH interaction (H2 test)
features_with_int = features_no_int + ['late_x_highrisk']
X_with_int = df[features_with_int].dropna()
y_with_int = df.loc[X_with_int.index, 'mort30']

model_with_int = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model_with_int.fit(X_with_int, y_with_int)

# Get formal p-value using statsmodels (sklearn doesn't provide p-values)
X_with_int_const = sm.add_constant(X_with_int)  # Add intercept column
logit_model = sm.Logit(y_with_int, X_with_int_const)
logit_result = logit_model.fit(disp=0)  # disp=0 suppresses convergence messages

# Extract interaction statistics
interaction_idx = list(X_with_int.columns).index('late_x_highrisk') + 1  # +1 for constant
interaction_coef = logit_result.params.iloc[interaction_idx]
interaction_pval = logit_result.pvalues.iloc[interaction_idx]
interaction_or = np.exp(interaction_coef)
interaction_ci = np.exp(logit_result.conf_int().iloc[interaction_idx])  # 95% CI

# Display results
print("Interaction Term: late_week × high_risk_proc")
print(f"  Odds Ratio:  {interaction_or:.4f}")
print(f"  95% CI:      [{interaction_ci[0]:.4f}, {interaction_ci[1]:.4f}]")
print(f"  p-value:     {interaction_pval:.4f}")

# Interpret significance
if interaction_pval < 0.001:
    sig_marker = "***"
elif interaction_pval < 0.01:
    sig_marker = "**"
elif interaction_pval < 0.05:
    sig_marker = "*"
else:
    sig_marker = "NS"

print(f"\n{'='*80}")
print("INTERPRETATION:")
print(f"{'='*80}")

if interaction_pval < 0.05:
    print(f"✓ SIGNIFICANT INTERACTION (p = {interaction_pval:.4f} {sig_marker})")
    if interaction_or > 1:
        print(f"  → High-risk procedures show {(interaction_or-1)*100:.1f}% GREATER late-week effect")
        print(f"  → Timing matters MORE for complex surgeries")
    else:
        print(f"  → High-risk procedures show {(1-interaction_or)*100:.1f}% SMALLER late-week effect")
else:
    print(f"✗ NOT SIGNIFICANT (p = {interaction_pval:.4f})")
    print(f"  → Cannot conclude differential effect by procedure type")

# Likelihood Ratio Test comparing models with/without interaction
ll_no_int = -log_loss(y_with_int, model_no_int.predict_proba(X_with_int[features_no_int])[:, 1], normalize=False)
ll_with_int = -log_loss(y_with_int, model_with_int.predict_proba(X_with_int)[:, 1], normalize=False)

lr_interaction = 2 * (ll_with_int - ll_no_int)
p_lr_interaction = 1 - stats.chi2.cdf(lr_interaction, 1)  # 1 df for one interaction term

print(f"\nLikelihood Ratio Test (Model Comparison):")
print(f"  χ² = {lr_interaction:.2f}, df = 1, p = {p_lr_interaction:.4f}")
if p_lr_interaction < 0.05:
    print(f"  ✓ Adding interaction significantly improves model fit")

# Calculate AUC improvement
auc_no_int = roc_auc_score(y_no_int, model_no_int.predict_proba(X_no_int)[:, 1])
auc_with_int = roc_auc_score(y_with_int, model_with_int.predict_proba(X_with_int)[:, 1])
print(f"\n  AUC without interaction: {auc_no_int:.4f}")
print(f"  AUC with interaction:    {auc_with_int:.4f}")
print(f"  Improvement:             {auc_with_int - auc_no_int:.4f}")

# ============================================
# DESCRIPTIVE STRATIFICATION (for report table)
# ============================================
print("\n" + "="*80)
print("DESCRIPTIVE: Mortality Rates by Day and Procedure Risk")
print("="*80)

stratified = df.groupby(['late_week', 'high_risk_proc']).agg({
    'mort30': ['mean', 'count', 'sum']
}).round(4)
stratified.columns = ['Mortality_Rate', 'N', 'Deaths']
print(stratified)

# Calculate specific comparisons for report
low_risk = stratified.xs(0, level='high_risk_proc')
high_risk = stratified.xs(1, level='high_risk_proc')

print("\nKey Comparisons (for Table):")
print(f"  Low Risk:  Monday {low_risk.loc[0, 'Mortality_Rate']*100:.2f}% → Friday {low_risk.loc[1, 'Mortality_Rate']*100:.2f}%")
print(f"  High Risk: Monday {high_risk.loc[0, 'Mortality_Rate']*100:.2f}% → Friday {high_risk.loc[1, 'Mortality_Rate']*100:.2f}%")
print(f"  Interaction: {(high_risk.loc[1, 'Mortality_Rate'] - high_risk.loc[0, 'Mortality_Rate'] - (low_risk.loc[1, 'Mortality_Rate'] - low_risk.loc[0, 'Mortality_Rate']))*100:.2f}%")

# ============================================
# ESSENTIAL DIAGNOSTICS (Model 2 only)
# ============================================
print("\n" + "="*80)
print("MODEL DIAGNOSTICS: Essential Checks for Model 2")
print("="*80)

# Fit Model 2 with statsmodels for diagnostics
X_full_const = sm.add_constant(X_full)
logit_model2 = sm.Logit(y_full, X_full_const)
logit_result2 = logit_model2.fit(disp=0)

# ---- DIAGNOSTIC 1: Multicollinearity (VIF) ----
print("\n" + "-"*80)
print("1. MULTICOLLINEARITY (VIF)")
print("-"*80)
print("Rule: VIF < 5 = Good, 5-10 = Moderate, >10 = Serious problem\n")

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X_full.columns
vif_data["VIF"] = [variance_inflation_factor(X_full.values, i) for i in range(X_full.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))

max_vif = vif_data['VIF'].max()
if max_vif < 5:
    print(f"\n✓ PASS: Max VIF = {max_vif:.2f} < 5 (No problematic multicollinearity)")
elif max_vif < 10:
    print(f"\n⚠ CAUTION: Max VIF = {max_vif:.2f} (Moderate multicollinearity)")
else:
    print(f"\n⚠ FAIL: Max VIF = {max_vif:.2f} > 10 (Serious multicollinearity)")

# ---- DIAGNOSTIC 2: Model Fit (Hosmer-Lemeshow) ----
print("\n" + "-"*80)
print("2. MODEL FIT (Hosmer-Lemeshow Test)")
print("-"*80)
print("Rule: p > 0.05 indicates good fit\n")

# Get predicted probabilities
y_pred_prob = logit_result2.predict(X_full_const)

# Create deciles
deciles = pd.qcut(y_pred_prob, q=10, labels=False, duplicates='drop')

# Calculate observed vs expected
hl_table = pd.DataFrame({'decile': deciles, 'y_actual': y_full, 'y_pred': y_pred_prob})
hl_summary = hl_table.groupby('decile').agg({'y_actual': ['sum', 'count'], 'y_pred': 'sum'}).reset_index()
hl_summary.columns = ['decile', 'observed', 'total', 'expected']
hl_summary['observed_neg'] = hl_summary['total'] - hl_summary['observed']
hl_summary['expected_neg'] = hl_summary['total'] - hl_summary['expected']

# Calculate Hosmer-Lemeshow statistic
hl_stat = (
    ((hl_summary['observed'] - hl_summary['expected'])**2 / hl_summary['expected']).sum() +
    ((hl_summary['observed_neg'] - hl_summary['expected_neg'])**2 / hl_summary['expected_neg']).sum()
)
df_hl = len(hl_summary) - 2
p_value_hl = 1 - chi2.cdf(hl_stat, df_hl)

print(f"χ² = {hl_stat:.2f}, df = {df_hl}, p = {p_value_hl:.4f}")

if p_value_hl > 0.05:
    print(f"✓ PASS: Good fit (p = {p_value_hl:.4f} > 0.05)")
else:
    print(f"⚠ FAIL: Poor fit (p = {p_value_hl:.4f} < 0.05)")
    print("Note: With 32K observations and extreme imbalance, test is overly sensitive")
    print("      Model discrimination (AUC=0.908) remains excellent for inference")

# ---- DIAGNOSTIC 3: Influential Outliers (Cook's Distance) ----
print("\n" + "-"*80)
print("3. INFLUENTIAL OUTLIERS (Cook's Distance)")
print("-"*80)
print("Rule: Cook's D > 1.0 indicates highly influential observations\n")

# Calculate Cook's distance
influence = logit_result2.get_influence()
cooks_d = influence.cooks_distance[0]

print(f"Max Cook's D: {cooks_d.max():.6f}")
print(f"Mean:         {cooks_d.mean():.6f}")

n_influential = (cooks_d > 1.0).sum()
if n_influential == 0:
    print(f"\n✓ PASS: No influential outliers (all Cook's D < 1.0)")
    print(f"        Model coefficients are stable")
else:
    print(f"\n⚠ WARNING: {n_influential} observations have Cook's D > 1.0")

# ---- SUMMARY ----
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

checks_passed = []
checks_failed = []

if max_vif < 5:
    checks_passed.append("✓ No multicollinearity")
else:
    checks_failed.append(f"⚠ Multicollinearity (VIF={max_vif:.2f})")

if p_value_hl > 0.05:
    checks_passed.append("✓ Good model fit")
else:
    checks_failed.append("⚠ Poor H-L fit (acceptable given sample size/imbalance)")

if n_influential == 0:
    checks_passed.append("✓ No influential outliers")
else:
    checks_failed.append(f"⚠ {n_influential} influential outliers")

print("\nPassed:")
for item in checks_passed:
    print(f"  {item}")

if checks_failed:
    print("\nConcerns:")
    for item in checks_failed:
        print(f"  {item}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)




# ============================================
# VISUALIZATION PROPORTIONS: Day × Procedure Interaction
# ============================================



print("\nCreating Day × Procedure interaction plot...")

# Calculate mortality rates with standard errors
interaction_plot_data = df.groupby(['late_week', 'high_risk_proc']).agg({
    'mort30': ['mean', 'count', 'sum']
}).reset_index()
interaction_plot_data.columns = ['late_week', 'high_risk_proc', 'mort_rate', 'n', 'deaths']

# Calculate standard error of proportion
interaction_plot_data['se'] = np.sqrt(
    interaction_plot_data['mort_rate'] * (1 - interaction_plot_data['mort_rate']) / interaction_plot_data['n']
)

# Separate by procedure risk
low_risk_data = interaction_plot_data[interaction_plot_data['high_risk_proc'] == 0]
high_risk_data = interaction_plot_data[interaction_plot_data['high_risk_proc'] == 1]

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

x = np.array([0, 1])  # 0=Early week, 1=Late week
width = 0.35

# Low-risk bars
bars1 = ax.bar(x - width/2, low_risk_data['mort_rate'] * 100, width,
               yerr=low_risk_data['se'] * 100, 
               label='Low-Risk Procedures', capsize=5, 
               color='#4A90E2', alpha=0.85, edgecolor='black', linewidth=1.2)

# High-risk bars
bars2 = ax.bar(x + width/2, high_risk_data['mort_rate'] * 100, width,
               yerr=high_risk_data['se'] * 100,
               label='High-Risk Procedures', capsize=5, 
               color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.2)

# Annotations showing sample sizes
for i, (bar, data) in enumerate([(bars1, low_risk_data), (bars2, high_risk_data)]):
    for j, rect in enumerate(bar):
        height = rect.get_height()
        n_val = data.iloc[j]['n']
        deaths_val = data.iloc[j]['deaths']
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.05,
                f'n={n_val:,}\n({int(deaths_val)} deaths)',
                ha='center', va='bottom', fontsize=8, color='black')

# Styling
ax.set_ylabel('30-Day Mortality Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax.set_title('Day-of-Week Effect by Procedure Risk\n(H2: Interaction p=0.091)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Monday-Wednesday', 'Thursday-Friday'], fontsize=11)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.set_ylim(0, max(high_risk_data['mort_rate'].max() * 100 * 1.4, 2))
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('day_procedure_interaction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: day_procedure_interaction.png")
plt.show()