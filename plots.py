

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_excel('SurgeryTiming.xlsx')

# ============================================
# PLOT 1: Mortality by Day × Procedure Risk
# ============================================

# Define high-risk procedures (mortality > 1%)
high_risk_procedures = ['Colorectal resection', 
                        'Small bowel resection',
                        'Gastrectomy; partial and total',
                        'Hip replacement; total and partial',
                        'Spinal fusion',
                        'Endoscopy and endoscopic biopsy of the urinary...']

df['proc_risk'] = df['ahrq_ccs'].apply(
    lambda x: 'High Risk' if x in high_risk_procedures else 'Low Risk'
)

# Calculate mortality by day and procedure risk
mort_by_day_risk = df.groupby(['dow', 'proc_risk'])['mort30'].agg(['mean', 'count']).reset_index()
mort_by_day_risk['sem'] = np.sqrt(mort_by_day_risk['mean'] * (1 - mort_by_day_risk['mean']) / mort_by_day_risk['count'])

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(5)  # Days 1-5
width = 0.35

low_risk = mort_by_day_risk[mort_by_day_risk['proc_risk'] == 'Low Risk']
high_risk = mort_by_day_risk[mort_by_day_risk['proc_risk'] == 'High Risk']

bars1 = ax.bar(x - width/2, low_risk['mean']*100, width, 
               yerr=low_risk['sem']*100, label='Low Risk Procedures',
               capsize=5, alpha=0.8, color='steelblue')

bars2 = ax.bar(x + width/2, high_risk['mean']*100, width,
               yerr=high_risk['sem']*100, label='High Risk Procedures',
               capsize=5, alpha=0.8, color='indianred')

ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax.set_ylabel('30-Day Mortality Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Mortality Rate by Day of Week and Procedure Risk', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mortality_by_day_procedure_risk.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistical test
print("="*60)
print("Interaction Test: Day × Procedure Risk")
print("="*60)
low_fri = low_risk[low_risk['dow']==5]['mean'].values[0]
low_mon = low_risk[low_risk['dow']==1]['mean'].values[0]
high_fri = high_risk[high_risk['dow']==5]['mean'].values[0]
high_mon = high_risk[high_risk['dow']==1]['mean'].values[0]

print(f"Low Risk: Friday {low_fri:.4f} vs Monday {low_mon:.4f} (Δ={low_fri-low_mon:.4f})")
print(f"High Risk: Friday {high_fri:.4f} vs Monday {high_mon:.4f} (Δ={high_fri-high_mon:.4f})")
print(f"Interaction effect: {(high_fri-high_mon)-(low_fri-low_mon):.4f}")

# ============================================
# PLOT 2: Patient Characteristics Distribution
# ============================================

# Create day categories for cleaner visualization
df['day_period'] = df['dow'].map({1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri'})

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Patient Characteristics Distribution by Day of Week\n(Checking for Confounding)', 
             fontsize=16, fontweight='bold')

# Variables to plot
variables = [
    ('age', 'Age (years)'),
    ('asa_status', 'ASA Status'),
    ('bmi', 'BMI'),
    ('baseline_charlson', 'Charlson Score'),
    ('mortality_rsi', 'Mortality RSI'),
    ('baseline_cvd', 'Cardiovascular Disease (%)')
]

for idx, (var, label) in enumerate(variables):
    ax = axes[idx//3, idx%3]
    
    if var in ['baseline_cvd', 'baseline_diabetes']:
        # For binary variables, show proportions
        prop_data = df.groupby('dow')[var].mean() * 100
        prop_data.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_ylabel('Proportion (%)', fontsize=10)
    else:
        # For continuous variables, show boxplots
        df.boxplot(column=var, by='dow', ax=ax, patch_artist=True)
        ax.set_ylabel(label, fontsize=10)
    
    ax.set_xlabel('Day of Week', fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], rotation=0)
    ax.get_figure().suptitle('')  # Remove automatic title from boxplot
    
fig.suptitle('Patient Characteristics Distribution by Day of Week\n(Checking for Confounding)', 
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('patient_characteristics_by_dow.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Statistical Tests for Confounding
# ============================================

print("\n" + "="*60)
print("CONFOUNDING ANALYSIS: Patient Characteristics by Day")
print("="*60)

# ANOVA tests for continuous variables
continuous_vars = ['age', 'asa_status', 'bmi', 'baseline_charlson', 'mortality_rsi']

for var in continuous_vars:
    groups = [df[df['dow']==d][var].dropna() for d in df['dow'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Calculate effect size (eta-squared)
    grand_mean = df[var].mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
    ss_total = sum([(x - grand_mean)**2 for g in groups for x in g])
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    print(f"\n{var}:")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_val:.4f} {significance}")
    print(f"  Effect size (η²): {eta_squared:.4f}")
    print(f"  Interpretation: {'Significant difference' if p_val < 0.05 else 'No significant difference'} across days")

# Chi-square tests for binary variables
binary_vars = ['baseline_cvd', 'baseline_diabetes', 'baseline_cancer']

for var in binary_vars:
    contingency = pd.crosstab(df['dow'], df[var])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    print(f"\n{var}:")
    print(f"  χ² statistic: {chi2:.3f}")
    print(f"  p-value: {p_val:.4f} {significance}")
    print(f"  Interpretation: {'Significant association' if p_val < 0.05 else 'No significant association'} with day")

print("\n" + "="*60)
print("CONCLUSION: Confounding Assessment")
print("="*60)
print("If p-values > 0.05 for all variables:")
print("  → Patient mix is BALANCED across days")
print("  → Temporal effects are NOT confounded by patient characteristics")
print("  → Day-of-week can be interpreted causally (with caution)")