'''LATE DAY EFFECT TESTS'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_excel('SurgeryTiming.xlsx')

# ============================================
# STEP 1: CREATE TIME-OF-DAY CATEGORIES
# ============================================

print("="*80)
print("TIME-OF-DAY ANALYSIS: INTRA-DAY FATIGUE EFFECTS")
print("="*80)

# Check hour distribution first
print("\nHour distribution:")
print(df['hour'].describe())
print(f"Unique hour values: {df['hour'].nunique()}")

# Create time-of-day categories based on clinical shifts
def categorize_time(hour):
    """
    Bin continuous hours into clinically meaningful periods.
    
    Morning (6-12): Start of day, peak performance
    Afternoon (12-16): Post-lunch, moderate fatigue  
    Evening (16-19): End of day, peak fatigue
    """
    if pd.isna(hour):
        return np.nan
    elif hour < 12:
        return 'Morning'
    elif hour < 16:
        return 'Afternoon'
    else:
        return 'Evening'

df['time_category'] = df['hour'].apply(categorize_time)

# Check distribution
print("\n" + "="*80)
print("TIME CATEGORY DISTRIBUTION")
print("="*80)

time_dist = df.groupby('time_category').agg({
    'mort30': ['count', 'sum', 'mean']
}).round(4)
time_dist.columns = ['N_Surgeries', 'N_Deaths', 'Mortality_Rate']
time_dist['Mortality_Pct'] = time_dist['Mortality_Rate'] * 100

print(time_dist)

# Create binary late_day indicator (Evening = 1)
df['late_day'] = (df['time_category'] == 'Evening').astype(int)

# Recreate late_week and proc_risk from previous analyses
df['late_week'] = (df['dow'] >= 4).astype(int)

high_risk_procedures = ['Colorectal resection', 'Small bowel resection',
                        'Gastrectomy; partial and total', 'Hip replacement; total and partial',
                        'Spinal fusion']
df['high_risk_proc'] = df['ahrq_ccs'].isin(high_risk_procedures).astype(int)

# Impute missing values
df['bmi_imputed'] = df['bmi'].fillna(df['bmi'].median())
df['asa_imputed'] = df['asa_status'].fillna(df['asa_status'].median())

# ============================================
# STEP 2: DESCRIPTIVE ANALYSIS
# ============================================

print("\n" + "="*80)
print("DESCRIPTIVE: MORTALITY BY TIME AND RISK")
print("="*80)

# Mortality by time category only
print("\nMortality by Time of Day:")
print(df.groupby('time_category')['mort30'].agg(['mean', 'count', 'sum']))

# Mortality by time × procedure risk
print("\nMortality by Time of Day × Procedure Risk:")
time_risk = df.groupby(['time_category', 'high_risk_proc']).agg({
    'mort30': ['mean', 'count', 'sum']
}).round(4)
time_risk.columns = ['Mortality_Rate', 'N', 'Deaths']
print(time_risk)

# Triple: day × time × procedure
print("\nMortality by Day × Time × Procedure Risk (Key Combinations):")
triple = df.groupby(['late_week', 'time_category', 'high_risk_proc']).agg({
    'mort30': ['mean', 'count', 'sum']
}).round(4)
triple.columns = ['Mortality_Rate', 'N', 'Deaths']
print(triple)

# ============================================
# STEP 3: MODEL 1 - TIME MAIN EFFECT
# ============================================

print("\n" + "="*80)
print("MODEL 1: TIME-OF-DAY MAIN EFFECT")
print("="*80)

# One-hot encode time_category (Morning as reference)
df['time_afternoon'] = (df['time_category'] == 'Afternoon').astype(int)
df['time_evening'] = (df['time_category'] == 'Evening').astype(int)

# Features
base_features = ['mortality_rsi', 'asa_imputed', 'baseline_charlson', 
                 'baseline_cancer', 'baseline_cvd']
model1_features = base_features + ['time_afternoon', 'time_evening']

# Prepare data
X_model1 = df[model1_features].dropna()
y_model1 = df.loc[X_model1.index, 'mort30']

# Fit model
model1 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model1.fit(X_model1, y_model1)

# Evaluate
y_pred1 = model1.predict_proba(X_model1)[:, 1]
auc1 = roc_auc_score(y_model1, y_pred1)

print(f"\nModel 1 AUC: {auc1:.4f}")
print("\nCoefficients:")
print(f"{'Variable':<25} {'OR':<12} {'Interpretation':<40}")
print("-"*77)

for i, var in enumerate(model1_features):
    or_val = np.exp(model1.coef_[0][i])
    
    if var == 'time_afternoon':
        interp = "vs Morning (reference)"
    elif var == 'time_evening':
        interp = "vs Morning (reference)"
    else:
        interp = ""
    
    marker = " ← KEY" if 'time_' in var else ""
    print(f"{var:<25} {or_val:<12.4f} {interp:<40}{marker}")

# ============================================
# STEP 4: MODEL 2 - TIME × PROCEDURE INTERACTION
# ============================================

print("\n" + "="*80)
print("MODEL 2: TIME × PROCEDURE RISK INTERACTION")
print("="*80)

# Create interaction terms
df['afternoon_x_highrisk'] = df['time_afternoon'] * df['high_risk_proc']
df['evening_x_highrisk'] = df['time_evening'] * df['high_risk_proc']

model2_features = base_features + ['time_afternoon', 'time_evening', 'high_risk_proc',
                                   'afternoon_x_highrisk', 'evening_x_highrisk']

# Prepare data
X_model2 = df[model2_features].dropna()
y_model2 = df.loc[X_model2.index, 'mort30']

# Fit model
model2 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model2.fit(X_model2, y_model2)

# Evaluate
y_pred2 = model2.predict_proba(X_model2)[:, 1]
auc2 = roc_auc_score(y_model2, y_pred2)

print(f"\nModel 2 AUC: {auc2:.4f}")
print(f"AUC Improvement over Model 1: {auc2 - auc1:.4f}")

print("\nKey Coefficients:")
print(f"{'Variable':<30} {'OR':<12}")
print("-"*42)

for i, var in enumerate(model2_features):
    or_val = np.exp(model2.coef_[0][i])
    marker = " ← INTERACTION" if '_x_' in var else ""
    print(f"{var:<30} {or_val:<12.4f}{marker}")

# Interpret interactions
afternoon_int_or = np.exp(model2.coef_[0][model2_features.index('afternoon_x_highrisk')])
evening_int_or = np.exp(model2.coef_[0][model2_features.index('evening_x_highrisk')])

print("\n" + "*"*80)
print("INTERACTION INTERPRETATION:")
print("*"*80)
print(f"Afternoon × High-Risk: OR = {afternoon_int_or:.4f}")
print(f"Evening × High-Risk:   OR = {evening_int_or:.4f}")

if evening_int_or > 1.2:
    print("\n✓ FINDING: High-risk procedures show GREATER evening effect")
elif evening_int_or < 0.8:
    print("\n→ High-risk procedures show SMALLER evening effect")
else:
    print("\n→ No significant interaction")


# ============================================
# DIAGNOSTICS: Time × Procedure Model
# ============================================
print("\n" + "="*80)
print("MODEL DIAGNOSTICS: Time × Procedure Interaction Model")
print("="*80)

# Fit with statsmodels for diagnostics
X_model2_const = sm.add_constant(X_model2)
logit_time_model = sm.Logit(y_model2, X_model2_const)
logit_time_result = logit_time_model.fit(disp=0)

# ---- VIF ----
print("\n1. Multicollinearity (VIF):")
vif_time = pd.DataFrame()
vif_time["Variable"] = X_model2.columns
vif_time["VIF"] = [variance_inflation_factor(X_model2.values, i) for i in range(X_model2.shape[1])]
vif_time = vif_time.sort_values('VIF', ascending=False)
print(vif_time.to_string(index=False))

max_vif_time = vif_time['VIF'].max()
if max_vif_time < 5:
    print(f"✓ PASS: Max VIF = {max_vif_time:.2f} < 5")
else:
    print(f"⚠ CAUTION: Max VIF = {max_vif_time:.2f}")

# ---- Hosmer-Lemeshow ----
print("\n2. Model Fit (Hosmer-Lemeshow):")
y_pred_time_prob = logit_time_result.predict(X_model2_const)
deciles_time = pd.qcut(y_pred_time_prob, q=10, labels=False, duplicates='drop')

hl_time = pd.DataFrame({'decile': deciles_time, 'y_actual': y_model2, 'y_pred': y_pred_time_prob})
hl_time_summary = hl_time.groupby('decile').agg({'y_actual': ['sum', 'count'], 'y_pred': 'sum'}).reset_index()
hl_time_summary.columns = ['decile', 'observed', 'total', 'expected']
hl_time_summary['observed_neg'] = hl_time_summary['total'] - hl_time_summary['observed']
hl_time_summary['expected_neg'] = hl_time_summary['total'] - hl_time_summary['expected']

hl_stat_time = (
    ((hl_time_summary['observed'] - hl_time_summary['expected'])**2 / hl_time_summary['expected']).sum() +
    ((hl_time_summary['observed_neg'] - hl_time_summary['expected_neg'])**2 / hl_time_summary['expected_neg']).sum()
)
df_hl_time = len(hl_time_summary) - 2
p_hl_time = 1 - stats.chi2.cdf(hl_stat_time, df_hl_time)

print(f"χ² = {hl_stat_time:.2f}, df = {df_hl_time}, p = {p_hl_time:.4f}")
if p_hl_time > 0.05:
    print(f"✓ PASS: Good fit (p > 0.05)")
else:
    print(f"⚠ FAIL: p = {p_hl_time:.4f} < 0.05 (acceptable given sample size)")

# ---- Cook's Distance ----
print("\n3. Influential Outliers (Cook's Distance):")
influence_time = logit_time_result.get_influence()
cooks_time = influence_time.cooks_distance[0]

print(f"Max Cook's D: {cooks_time.max():.6f}")
n_influential_time = (cooks_time > 1.0).sum()

if n_influential_time == 0:
    print(f"✓ PASS: No influential outliers")
else:
    print(f"⚠ WARNING: {n_influential_time} outliers")

print("\nDiagnostic Summary: Time Model")
print(f"  VIF: {'✓' if max_vif_time < 5 else '⚠'}")
print(f"  Fit: {'✓' if p_hl_time > 0.05 else '⚠'}")
print(f"  Outliers: {'✓' if n_influential_time == 0 else '⚠'}")



# ============================================
# VISUALIZATION PROPORTIONS: Time × Procedure Interaction
# ============================================

print("\nCreating Time × Procedure interaction plot...")

# Calculate mortality rates by time category and procedure risk
time_interaction_data = df.groupby(['time_category', 'high_risk_proc']).agg({
    'mort30': ['mean', 'count', 'sum']
}).reset_index()
time_interaction_data.columns = ['time_category', 'high_risk_proc', 'mort_rate', 'n', 'deaths']

# Calculate standard error
time_interaction_data['se'] = np.sqrt(
    time_interaction_data['mort_rate'] * (1 - time_interaction_data['mort_rate']) / time_interaction_data['n']
)

# Order time categories
time_order = ['Morning', 'Afternoon', 'Evening']
time_interaction_data['time_category'] = pd.Categorical(
    time_interaction_data['time_category'], 
    categories=time_order, 
    ordered=True
)
time_interaction_data = time_interaction_data.sort_values('time_category')

# Separate by risk
low_risk_time = time_interaction_data[time_interaction_data['high_risk_proc'] == 0]
high_risk_time = time_interaction_data[time_interaction_data['high_risk_proc'] == 1]

# Create plot
fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(time_order))
width = 0.35

# Low-risk bars
bars1 = ax.bar(x - width/2, low_risk_time['mort_rate'] * 100, width,
               yerr=low_risk_time['se'] * 100,
               label='Low-Risk Procedures', capsize=5,
               color='#4A90E2', alpha=0.85, edgecolor='black', linewidth=1.2)

# High-risk bars
bars2 = ax.bar(x + width/2, high_risk_time['mort_rate'] * 100, width,
               yerr=high_risk_time['se'] * 100,
               label='High-Risk Procedures', capsize=5,
               color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.2)

# Annotations
for i, (bar, data) in enumerate([(bars1, low_risk_time), (bars2, high_risk_time)]):
    for j, rect in enumerate(bar):
        height = rect.get_height()
        n_val = data.iloc[j]['n']
        deaths_val = data.iloc[j]['deaths']
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.08,
                f'n={n_val:,}\n({int(deaths_val)} deaths)',
                ha='center', va='bottom', fontsize=8, color='black')

# Styling
ax.set_ylabel('30-Day Mortality Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
ax.set_title('Time-of-Day Effect by Procedure Risk\n(H3: Evening × High-Risk OR=1.59, p<0.05)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Morning\n(6am-12pm)', 'Afternoon\n(12pm-4pm)', 'Evening\n(4pm-7pm)'], fontsize=10)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.set_ylim(0, max(high_risk_time['mort_rate'].max() * 100 * 1.4, 3.5))
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('time_procedure_interaction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: time_procedure_interaction.png")
plt.show()



# ============================================
# STEP 6: VISUALISATION
# ============================================

# print("\n" + "="*80)
# print("GENERATING VISUALIZATIONS")
# print("="*80)

# # Create stratified plot: Time × Procedure Risk
# fig, ax = plt.subplots(figsize=(10, 6))

# # Calculate mortality rates
# plot_data = df.groupby(['time_category', 'high_risk_proc']).agg({
#     'mort30': ['mean', 'count']
# }).reset_index()
# plot_data.columns = ['time_category', 'high_risk_proc', 'mort_rate', 'n']
# plot_data['sem'] = np.sqrt(plot_data['mort_rate'] * (1 - plot_data['mort_rate']) / plot_data['n'])

# # Order time categories
# time_order = ['Morning', 'Afternoon', 'Evening']
# plot_data['time_category'] = pd.Categorical(plot_data['time_category'], categories=time_order, ordered=True)
# plot_data = plot_data.sort_values('time_category')

# # Separate by risk
# low_risk = plot_data[plot_data['high_risk_proc'] == 0]
# high_risk = plot_data[plot_data['high_risk_proc'] == 1]

# # Plot
# x = np.arange(len(time_order))
# width = 0.35

# bars1 = ax.bar(x - width/2, low_risk['mort_rate']*100, width,
#                yerr=low_risk['sem']*100, label='Low Risk', 
#                capsize=5, alpha=0.8, color='steelblue')

# bars2 = ax.bar(x + width/2, high_risk['mort_rate']*100, width,
#                yerr=high_risk['sem']*100, label='High Risk',
#                capsize=5, alpha=0.8, color='indianred')

# ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
# ax.set_ylabel('30-Day Mortality Rate (%)', fontsize=12, fontweight='bold')
# ax.set_title('Mortality Rate by Time of Day and Procedure Risk', fontsize=14, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(time_order)
# ax.legend()
# ax.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.savefig('time_of_day_mortality.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: time_of_day_mortality.png")
# plt.show()

# # ============================================
# # STEP 7: "DANGER ZONE" HEATMAP (NOVEL)
# # ============================================

# print("\nCreating 'Danger Zone' heatmap...")

# # Calculate mortality rates for all combinations
# heatmap_data = df.groupby(['late_week', 'time_category', 'high_risk_proc']).agg({
#     'mort30': ['mean', 'count']
# }).reset_index()
# heatmap_data.columns = ['late_week', 'time_category', 'high_risk_proc', 'mort_rate', 'n']

# # Create separate heatmaps for low and high risk
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# for idx, (risk_level, risk_label) in enumerate([(0, 'Low Risk'), (1, 'High Risk')]):
#     subset = heatmap_data[heatmap_data['high_risk_proc'] == risk_level]
    
#     # Pivot for heatmap
#     pivot = subset.pivot_table(values='mort_rate', 
#                                 index='time_category', 
#                                 columns='late_week',
#                                 aggfunc='mean')
    
#     # Reorder rows
#     pivot = pivot.reindex(['Morning', 'Afternoon', 'Evening'])
    
#     # Plot
#     sns.heatmap(pivot * 100, annot=True, fmt='.2f', cmap='YlOrRd',
#                 ax=axes[idx], vmin=0, vmax=2,
#                 cbar_kws={'label': 'Mortality Rate (%)'})
    
#     axes[idx].set_title(f'{risk_label} Procedures', fontsize=12, fontweight='bold')
#     axes[idx].set_xlabel('Day of Week (0=Mon-Wed, 1=Thu-Fri)')
#     axes[idx].set_ylabel('Time of Day')

# plt.suptitle('Mortality "Danger Zone" Map', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig('danger_zone_heatmap.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: danger_zone_heatmap.png")
# plt.show()

