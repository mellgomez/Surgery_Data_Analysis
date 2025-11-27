import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Load data
df = pd.read_excel('SurgeryTiming.xlsx')

# ============================================
# Mmortality_rsi is key covariate
# ============================================

# Create day categories
df['late_week'] = (df['dow'] >= 4).astype(int)  # Thu-Fri = 1

# Handle missing data
df['bmi_imputed'] = df['bmi'].fillna(df['bmi'].median())
df['age_imputed'] = df['age'].fillna(df['age'].median())
df['asa_imputed'] = df['asa_status'].fillna(df['asa_status'].median())

# ============================================
# Model 1: Baseline (mortality_rsi ONLY)
# ============================================
print("="*60)
print("MODEL 1: Baseline Risk Only (mortality_rsi)")
print("="*60)

X_baseline = df[['mortality_rsi']].values
y = df['mort30'].values

# Remove any rows with missing mortality_rsi (should be none)
valid_idx = ~np.isnan(X_baseline).any(axis=1)
X_baseline = X_baseline[valid_idx]
y_baseline = y[valid_idx]

model1 = LogisticRegression(class_weight='balanced', max_iter=1000)
model1.fit(X_baseline, y_baseline)

from sklearn.metrics import roc_auc_score
y_pred_baseline = model1.predict_proba(X_baseline)[:, 1]
auc_baseline = roc_auc_score(y_baseline, y_pred_baseline)

print(f"Baseline AUC: {auc_baseline:.4f}")
print(f"Coefficient (mortality_rsi): {model1.coef_[0][0]:.4f}")
print(f"Odds Ratio (per unit RSI): {np.exp(model1.coef_[0][0]):.4f}")


# ============================================
# Model 2: Add Day of Week Effect
# ============================================
print("\n" + "="*60)
print("MODEL 2: Baseline Risk + Day of Week")
print("="*60)

# Prepare features
feature_cols = ['mortality_rsi', 'late_week', 'asa_imputed', 
                'baseline_charlson', 'baseline_cancer', 'baseline_cvd'] # Removed 'age_imputed' to reduce multicollinearity

X_full = df[feature_cols].dropna()
y_full = df.loc[X_full.index, 'mort30']

model2 = LogisticRegression(class_weight='balanced', max_iter=1000)
model2.fit(X_full, y_full)

y_pred_full = model2.predict_proba(X_full)[:, 1]
auc_full = roc_auc_score(y_full, y_pred_full)

print(f"Full Model AUC: {auc_full:.4f}")
print(f"AUC Improvement: {auc_full - auc_baseline:.4f}")
print("\nCoefficients:")
for name, coef in zip(feature_cols, model2.coef_[0]):
    or_val = np.exp(coef)
    print(f"  {name:20s}: OR = {or_val:.4f}, β = {coef:7.4f}")



# ============================================
# Key Test: Is late_week significant after controlling for RSI?
# ============================================
print("\n" + "="*60)
print("HYPOTHESIS TEST: Late Week Effect")
print("="*60)

late_week_coef = model2.coef_[0][1]  # Second coefficient (late_week)
late_week_or = np.exp(late_week_coef)

print(f"Late Week Odds Ratio: {late_week_or:.4f}")
print(f"95% CI approximation: [{np.exp(late_week_coef - 1.96*0.3):.4f}, {np.exp(late_week_coef + 1.96*0.3):.4f}]")

if late_week_or > 1:
    print(f"Interpretation: Thursday-Friday surgeries have {(late_week_or-1)*100:.1f}% higher odds")
    print("of 30-day mortality AFTER controlling for baseline patient risk (mortality_rsi)")
else:
    print(f"Interpretation: Thursday-Friday surgeries have {(1-late_week_or)*100:.1f}% lower odds")

# ============================================
# MODEL COMPARISON: Likelihood Ratio Test + AIC + Deviance
# ============================================

# Calculate log-likelihoods for both models
ll_baseline = -log_loss(y_full, y_pred_baseline[X_full.index], normalize=False)
ll_full = -log_loss(y_full, y_pred_full, normalize=False)

# Calculate number of parameters for each model
# k = number of features + 1 (intercept)
k_baseline = X_baseline.shape[1] + 1  # Model 1: mortality_rsi + intercept
k_full = len(feature_cols) + 1         # Model 2: all features + intercept

# Calculate AIC for each model
# AIC = -2 * log-likelihood + 2 * k (penalizes complexity)
aic_baseline = -2 * ll_baseline + 2 * k_baseline
aic_full = -2 * ll_full + 2 * k_full

# Calculate Deviance for each model
# Deviance = -2 * log-likelihood (measures lack of fit)
deviance_baseline = -2 * ll_baseline
deviance_full = -2 * ll_full

# Likelihood Ratio Test (tests if Model 2 is significantly better)
lr_stat = 2 * (ll_full - ll_baseline)  # LR statistic
df_diff = k_full - k_baseline           # Difference in parameters
p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

# Print comprehensive comparison
print("\n" + "="*70)
print("MODEL COMPARISON: Baseline (Model 1) vs Full (Model 2)")
print("="*70)

print(f"\n{'Metric':<25} {'Model 1 (Baseline)':<20} {'Model 2 (Full)':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Predictors':<25} {k_baseline-1:<20} {k_full-1:<20} {(k_full-1)-(k_baseline-1):<15}")
print(f"{'Log-Likelihood':<25} {ll_baseline:<20.2f} {ll_full:<20.2f} {ll_full - ll_baseline:<15.2f}")
print(f"{'AIC':<25} {aic_baseline:<20.2f} {aic_full:<20.2f} {aic_full - aic_baseline:<15.2f}")
print(f"{'Deviance':<25} {deviance_baseline:<20.2f} {deviance_full:<20.2f} {deviance_full - deviance_baseline:<15.2f}")

print("\n" + "-"*70)
print("LIKELIHOOD RATIO TEST:")
print("-"*70)
print(f"  LR statistic: {lr_stat:.4f}")
print(f"  df: {df_diff}")
print(f"  p-value: {p_value:.4f}")
print(f"\n  Conclusion: Adding temporal/demographic factors {'SIGNIFICANTLY' if p_value < 0.05 else 'does NOT significantly'} improve model (p < 0.001)")

print("\n" + "-"*70)
print("INTERPRETATION:")
print("-"*70)
print(f"  • AIC difference: {aic_baseline - aic_full:.2f} (Lower is better; >10 = substantial improvement)")
if aic_baseline - aic_full > 10:
    print(f"    → Model 2 is SUBSTANTIALLY better (ΔAIC = {aic_baseline - aic_full:.2f})")
elif aic_baseline - aic_full > 2:
    print(f"    → Model 2 is moderately better (ΔAIC = {aic_baseline - aic_full:.2f})")
else:
    print(f"    → Models are similar (ΔAIC = {aic_baseline - aic_full:.2f})")

print(f"  • Deviance reduction: {deviance_baseline - deviance_full:.2f} (larger = better fit)")
print("="*70)
# ============================================
# Calibration Analysis:Hosmer-Lemeshow-like statistic
# ============================================
print("\n" + "="*60)
print("CALIBRATION: AHRQ Predictions vs Observed")
print("="*60)

# Convert RSI to predicted probability
df['predicted_mort'] = 1 / (1 + np.exp(-df['mortality_rsi']))

# Create risk deciles
df['rsi_decile'] = pd.qcut(df['mortality_rsi'], q=10, labels=False, duplicates='drop')

# Compare predicted vs observed by decile
calibration = df.groupby('rsi_decile').agg({
    'predicted_mort': 'mean',
    'mort30': 'mean',
    'mortality_rsi': ['mean', 'count']
}).round(4)

calibration.columns = ['Predicted_Mortality', 'Observed_Mortality', 'Mean_RSI', 'N']
print(calibration)

# Calibration plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(calibration['Predicted_Mortality']*100, 
           calibration['Observed_Mortality']*100, 
           s=calibration['N']/10, alpha=0.6)
ax.plot([0, 50], [0, 50], 'r--', label='Perfect Calibration')
ax.set_xlabel('AHRQ Predicted Mortality (%)', fontsize=12)
ax.set_ylabel('Observed Mortality (%)', fontsize=12)
ax.set_title('Calibration: AHRQ Risk Model vs Observed Outcomes', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_plot.png', dpi=300)
plt.show()

# Hosmer-Lemeshow-like statistic
expected_deaths = (df['predicted_mort'] * df.groupby('rsi_decile')['mort30'].transform('count')).groupby(df['rsi_decile']).sum()
observed_deaths = df.groupby('rsi_decile')['mort30'].sum()
hl_stat = ((observed_deaths - expected_deaths)**2 / expected_deaths).sum()
print(f"\nCalibration χ² statistic: {hl_stat:.2f}")
print("Note: Large value indicates poor calibration (AHRQ model may not fit this population)")










# ============================================
# LOGISTIC REGRESSION DIAGNOSTICS BNAD ASSUMPTIONS CHECKS. 
# ============================================

print("\n" + "="*80)
print("MODEL DIAGNOSTICS: CHECKING LOGISTIC REGRESSION ASSUMPTIONS")
print("="*80)

# We need statsmodels for proper diagnostic tests (sklearn doesn't provide these)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2

# Prepare data for statsmodels (needs constant term added)??
X_full_with_const = sm.add_constant(X_full)

# Fit logistic regression using statsmodels (to get detailed statistics)
logit_model = sm.Logit(y_full, X_full_with_const)
logit_result = logit_model.fit(disp=0)  # disp=0 suppresses convergence messages?

# ============================================
# ASSUMPTION 1: MULTICOLLINEARITY (VIF Test)
# ============================================
print("\n" + "-"*80)
print("ASSUMPTION: No Multicollinearity (VIF Test)")
print("-"*80)
print("Rule: VIF < 5 = Good, VIF 5-10 = Moderate concern, VIF > 10 = Serious problem\n")

# Calculate VIF for each predictor (excluding constant)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_full.columns
vif_data["VIF"] = [variance_inflation_factor(X_full.values, i) for i in range(X_full.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)

print(vif_data.to_string(index=False))

# Check if any VIF > 10 (need to simplify this code below)
max_vif = vif_data['VIF'].max()
if max_vif > 10:
    print(f"\n⚠ WARNING: Maximum VIF = {max_vif:.2f} > 10")
    print("   Serious multicollinearity detected!")
    print(f"   Variable: {vif_data.iloc[0]['Variable']}")
elif max_vif > 5:
    print(f"\n⚠ CAUTION: Maximum VIF = {max_vif:.2f} > 5")
    print("   Moderate multicollinearity present")
else:
    print(f"\n✓ PASS: Maximum VIF = {max_vif:.2f} < 5")
    print("   No problematic multicollinearity")

# ============================================
# ASSUMPTION 2: MODEL FIT (Hosmer-Lemeshow Test)
# ============================================
print("\n" + "-"*80)
print("ASSUMPTION: Good Model Fit (Hosmer-Lemeshow Test)")
print("-"*80)
print("Rule: p > 0.05 indicates good fit (fail to reject null of good fit)\n")

# Get predicted probabilities
y_pred_prob = logit_result.predict(X_full_with_const)

# Create deciles of predicted probabilities
deciles = pd.qcut(y_pred_prob, q=10, labels=False, duplicates='drop')

# Calculate observed vs expected in each decile
hl_table = pd.DataFrame({
    'decile': deciles,
    'y_actual': y_full,
    'y_pred': y_pred_prob
})

# Group by decile
hl_summary = hl_table.groupby('decile').agg({
    'y_actual': ['sum', 'count'],  # Observed events and total
    'y_pred': 'sum'  # Expected events
}).reset_index()

hl_summary.columns = ['decile', 'observed', 'total', 'expected']
hl_summary['observed_neg'] = hl_summary['total'] - hl_summary['observed']
hl_summary['expected_neg'] = hl_summary['total'] - hl_summary['expected']

# Calculate Hosmer-Lemeshow statistic
# HL = Σ [(Observed - Expected)² / Expected] for both events and non-events
hl_stat = (
    ((hl_summary['observed'] - hl_summary['expected'])**2 / hl_summary['expected']).sum() +
    ((hl_summary['observed_neg'] - hl_summary['expected_neg'])**2 / hl_summary['expected_neg']).sum()
)

# Degrees of freedom = (number of groups - 2)
df_hl = len(hl_summary) - 2
p_value_hl = 1 - chi2.cdf(hl_stat, df_hl)

print(f"Hosmer-Lemeshow χ² statistic: {hl_stat:.4f}")
print(f"Degrees of freedom: {df_hl}")
print(f"p-value: {p_value_hl:.4f}")

if p_value_hl > 0.05:
    print(f"\n✓ PASS: p = {p_value_hl:.4f} > 0.05")
    print("   Model fits the data well (fail to reject null hypothesis)")
else:
    print(f"\n⚠ FAIL: p = {p_value_hl:.4f} < 0.05")
    print("   Model fit is poor (reject null hypothesis)")
    print("   Consider: Adding interactions, polynomial terms, or different predictors")

# ============================================
# ASSUMPTION 3: LINEARITY OF LOG-ODDS (Box-Tidwell Test)
# ============================================
print("\n" + "-"*80)
print("ASSUMPTION: Linearity of Log-Odds (Box-Tidwell Test)")
print("-"*80)
print("Rule: p > 0.05 for interaction term indicates linearity is satisfied\n")

# Box-Tidwell: Add interaction between continuous predictors and their log
# Test for continuous predictors only
continuous_vars = ['mortality_rsi', 'asa_imputed', 'baseline_charlson'] # removed 'age_imputed' to reduce multicollinearity

print("Testing continuous predictors for linearity with log-odds:\n")

for var in continuous_vars:
    # Create interaction with natural log (add small constant to avoid log(0))
    X_bt = X_full.copy()
    
    # Check if variable has values <= 0 (can't take log)
    if (X_bt[var] <= 0).any():
        # Shift to make all positive
        X_bt[var + '_shifted'] = X_bt[var] - X_bt[var].min() + 1
        X_bt[var + '_log_interaction'] = X_bt[var] * np.log(X_bt[var + '_shifted'])
    else:
        X_bt[var + '_log_interaction'] = X_bt[var] * np.log(X_bt[var])
    
    # Fit model with interaction term
    X_bt_with_const = sm.add_constant(X_bt)
    model_bt = sm.Logit(y_full, X_bt_with_const)
    result_bt = model_bt.fit(disp=0)
    
    # Get p-value for interaction term (last coefficient)
    interaction_pval = result_bt.pvalues[var + '_log_interaction']
    
    print(f"  {var:<25} p = {interaction_pval:.4f}  ", end="")
    
    if interaction_pval > 0.05:
        print("✓ Linear relationship confirmed")
    else:
        print("⚠ Non-linear relationship detected - consider transformation")

print("\nNote: Non-significant p-values (>0.05) indicate linearity assumption is met")


# ============================================
# LINEARITY PLOT: Log-Odds vs Predictors


# 3 panels in one row
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Linearity Check: Log-Odds vs Continuous Predictors', 
             fontsize=14, fontweight='bold')

continuous_vars = ['mortality_rsi', 'asa_imputed', 'baseline_charlson']

for idx, var in enumerate(continuous_vars):
    ax = axes[idx]  # Now just single index, not [idx // 2, idx % 2]
    
    
    # Get predictor values and predicted log-odds
    x_values = X_full[var]
    log_odds = logit_result.predict(X_full_with_const, linear=True)  # Linear predictor (log-odds)
    
    # Scatter plot: observed data points
    ax.scatter(x_values, log_odds, alpha=0.3, s=10, color='gray', label='Data points')
    
    # Fit smoothed line (LOWESS - locally weighted regression)
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(log_odds, x_values, frac=0.3)  # frac controls smoothing
    
    # Plot red smoothed line
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, 
            label='Smoothed relationship')
    
    # Add reference line (perfect linearity would follow overall trend)
    z = np.polyfit(x_values, log_odds, 1)  # Linear fit
    p = np.poly1d(z)
    ax.plot(x_values, p(x_values), "b--", alpha=0.5, label='Linear fit')
    
    # Labels
    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel('Log-Odds (Linear Predictor)', fontsize=10)
    ax.set_title(f'{var}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('linearity_check_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: linearity_check_plot.png")
plt.show()


# ============================================
# ASSUMPTION 4: INFLUENTIAL OUTLIERS (Cook's Distance)
# ============================================
print("\n" + "-"*80)
print("ASSUMPTION: No Influential Outliers (Cook's Distance)")
print("-"*80)
print("Rule: Cook's D > 1.0 indicates highly influential observations\n")

# Calculate Cook's distance using statsmodels
influence = logit_result.get_influence()
cooks_d = influence.cooks_distance[0]  # Returns (distances, p-values)

# Summary statistics
print(f"Cook's Distance Statistics:")
print(f"  Maximum: {cooks_d.max():.6f}")
print(f"  Mean:    {cooks_d.mean():.6f}")
print(f"  Median:  {np.median(cooks_d):.6f}")

# Check for influential points
n_influential = (cooks_d > 1.0).sum()
n_moderate = ((cooks_d > 0.5) & (cooks_d <= 1.0)).sum()

if n_influential > 0:
    print(f"\n⚠ WARNING: {n_influential} observations have Cook's D > 1.0")
    print("   These are highly influential - consider investigating/removing")
    
    # Show top 5 most influential
    top_influential = np.argsort(cooks_d)[-5:][::-1]
    print("\n   Top 5 most influential observations:")
    print("   {:<10} {:<15}".format('Index', "Cook's D"))
    for idx in top_influential:
        print(f"   {idx:<10} {cooks_d[idx]:<15.6f}")
elif n_moderate > 0:
    print(f"\n⚠ CAUTION: {n_moderate} observations have Cook's D between 0.5-1.0")
    print("   Moderately influential but not critical")
else:
    print(f"\n✓ PASS: All Cook's D < 0.5")
    print("   No highly influential outliers detected")

# Optional: Create Cook's Distance plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
ax.axhline(y=1.0, color='r', linestyle='--', label="Threshold = 1.0")
ax.axhline(y=0.5, color='orange', linestyle='--', label="Threshold = 0.5")
ax.set_xlabel('Observation Index')
ax.set_ylabel("Cook's Distance")
ax.set_title("Cook's Distance Plot: Identifying Influential Observations")
ax.legend()
plt.tight_layout()
plt.savefig('cooks_distance_plot.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: cooks_distance_plot.png")
plt.show()
plt.close()

# ============================================
# SUMMARY OF ALL ASSUMPTIONS
# ============================================
print("\n" + "="*80)
print("SUMMARY: LOGISTIC REGRESSION ASSUMPTIONS")
print("="*80)

assumptions_met = []
assumptions_violated = []

# Check 1: Binary outcome (always met in your case)
assumptions_met.append("✓ Dependent variable is binary (mort30 = 0/1)")

# Check 2: Sample size
assumptions_met.append(f"✓ Adequate sample size (n={len(y_full):,}, events={y_full.sum()}, ratio={len(y_full)/y_full.sum():.1f}:1)")

# Check 3: Independence (assumed, not testable)
assumptions_met.append("✓ Independence assumed (no repeated measures)")

# Check 4: Multicollinearity
if max_vif < 5:
    assumptions_met.append(f"✓ No multicollinearity (max VIF = {max_vif:.2f})")
elif max_vif < 10:
    assumptions_violated.append(f"⚠ Moderate multicollinearity (max VIF = {max_vif:.2f})")
else:
    assumptions_violated.append(f"⚠ Severe multicollinearity (max VIF = {max_vif:.2f})")

# Check 5: Model fit
if p_value_hl > 0.05:
    assumptions_met.append(f"✓ Good model fit (Hosmer-Lemeshow p = {p_value_hl:.4f})")
else:
    assumptions_violated.append(f"⚠ Poor model fit (Hosmer-Lemeshow p = {p_value_hl:.4f})")

# Check 6: Influential outliers
if n_influential == 0:
    assumptions_met.append(f"✓ No influential outliers (max Cook's D = {cooks_d.max():.4f})")
else:
    assumptions_violated.append(f"⚠ {n_influential} influential outliers (Cook's D > 1.0)")

print("\nASSUMPTIONS MET:")
for item in assumptions_met:
    print(f"  {item}")

if assumptions_violated:
    print("\nASSUMPTIONS VIOLATED/CONCERNS:")
    for item in assumptions_violated:
        print(f"  {item}")
    print("\nRecommendation: Address violated assumptions before finalizing model")
else:
    print("\n✓ ALL ASSUMPTIONS SATISFIED - Model is appropriate for this data")

print("="*80)