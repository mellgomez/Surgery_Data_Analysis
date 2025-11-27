#To activate environment: .\.venv\Scripts\Activate.ps1



'''Exploratory Data Analysis for Surgery Timing Dataset'''

import pandas as pd
import numpy as np

# Load and inspect
df = pd.read_excel('SurgeryTiming.xlsx')
print(df.shape)
print(df.info())
print(df.describe())
print(df.head(20))


# Check ages counts
df['age_bin'] = pd.cut(df['age'], bins=[0,18,40,60,80,100,120])
df['age_bin'].value_counts().sort_index()
print(df['age_bin'].value_counts().sort_index())


print('-' * 40)
# Missing data patterns
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
print(pd.DataFrame({'Missing': missing, 'Percent': missing_pct}))

# Visualise missingness
import missingno as msno
msno.matrix(df)
plt.show() 
msno.heatmap(df)  # Are certain variables missing together?
plt.show() 


# Check if missingness relates to outcome ???
for col in df.columns:
    if df[col].isnull().sum() > 0:
        mort_rate_missing = df[df[col].isnull()]['mort30'].mean()
        mort_rate_present = df[df[col].notnull()]['mort30'].mean()
        print(f"{col}: Missing={mort_rate_missing:.3f}, Present={mort_rate_present:.3f}")


print('-' * 40)

# Mortality distribution
print(df['mort30'].value_counts())
print(f"Mortality rate: {df['mort30'].mean():.2%}")

# Complication distribution
print(df['complication'].value_counts())
print(f"Complication rate: {df['complication'].mean():.2%}")

# Cross-tabulation
print(pd.crosstab(df['mort30'], df['complication'], normalize='index'))

print('-' * 40)

#### Suspicious variables:

# Examine mortality_rsi, ccsMort30Rate, complication_rsi
print("\n=== Mortality RSI ===")
print(df['mortality_rsi'].describe())
print(f"Correlation with mort30: {df['mortality_rsi'].corr(df['mort30']):.3f}")

print("\n=== ccsMort30Rate ===")
print(df['ccsMort30Rate'].describe())
print(f"Correlation with mort30: {df['ccsMort30Rate'].corr(df['mort30']):.3f}")

# CRITICAL: Do these perfectly predict outcomes?
print(f"Max ccsMort30Rate: {df['ccsMort30Rate'].max()}")
print(f"Min ccsMort30Rate: {df['ccsMort30Rate'].min()}")

# Group by procedure and check
procedure_stats = df.groupby('ahrq_ccs').agg({
    'ccsMort30Rate': 'first',  # Should be same within procedure
    'mort30': 'mean'
}).reset_index()
print(procedure_stats.head(10))

# Test: Are these CONSTANT within procedure categories?
df.groupby('ahrq_ccs')['ccsMort30Rate'].nunique()



### TEMPORAL VARIABLES EXPLORATION:

print('-' * 40)

# Day of week distribution
print(df['dow'].value_counts().sort_index())
print("\nMortality by day:")
print(df.groupby('dow')['mort30'].agg(['mean', 'count', 'sum']))

# Hour distribution
print("\nHour distribution:")
print(df['hour'].describe())
print(df.groupby('hour')['mort30'].mean())

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mortality by dow
dow_mort = df.groupby('dow')['mort30'].mean()
axes[0,0].bar(dow_mort.index, dow_mort.values)
axes[0,0].set_title('Mortality Rate by Day of Week')
axes[0,0].set_xlabel('Day (0=Mon, 6=Sun)')

# Mortality by hour
hour_mort = df.groupby('hour')['mort30'].mean()
axes[0,1].plot(hour_mort.index, hour_mort.values, marker='o')
axes[0,1].set_title('Mortality Rate by Hour')

# Heatmap: dow x hour
pivot = df.pivot_table(values='mort30', index='dow', columns='hour', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[1,0], cmap='YlOrRd')
axes[1,0].set_title('Mortality Rate: Day × Hour')

# Sample sizes
sample_sizes = df.groupby(['dow', 'hour']).size().unstack(fill_value=0)
sns.heatmap(sample_sizes, annot=True, fmt='d', ax=axes[1,1], cmap='Blues')
axes[1,1].set_title('Sample Sizes: Day × Hour')

plt.tight_layout()
plt.savefig('temporal_exploration.png', dpi=300)


### PROCEDURE CATEGORIES (ahrq_ccs) EXPLORATION:

print('-' * 40)

# How many categories?
print(f"Unique procedures: {df['ahrq_ccs'].nunique()}")
print("\nTop 10 procedures by volume:")
print(df['ahrq_ccs'].value_counts().head(10))

# Mortality by procedure
proc_mort = df.groupby('ahrq_ccs').agg({
    'mort30': ['mean', 'count', 'sum']
}).round(4)
proc_mort.columns = ['mort_rate', 'n_cases', 'n_deaths']
proc_mort = proc_mort.sort_values('n_cases', ascending=False)
print(proc_mort.head(15))

# Identify rare procedures
rare_procedures = proc_mort[proc_mort['n_cases'] < 30]
print(f"\nProcedures with <30 cases: {len(rare_procedures)}")


#### PATIENT CHARACTERISTICS DISTRIBUTION:
print('-' * 40)
# Check if patient mix differs by day
print("\n=== Patient Characteristics by Day of Week ===")
for var in ['age', 'asa_status', 'bmi', 'baseline_charlson']:
    print(f"\n{var}:")
    print(df.groupby('dow')[var].mean())
    
####### Statistical test: Are sicker patients scheduled on specific days?
from scipy import stats

# # ASA status by dow (higher = sicker)
asa_by_dow = [df[df['dow']==d]['asa_status'].dropna() for d in df['dow'].unique()]
f_stat, p_val = stats.f_oneway(*asa_by_dow)
print(f"\nASA status differs by dow? F={f_stat:.3f}, p={p_val:.4f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
vars_to_plot = ['age', 'asa_status', 'bmi', 'baseline_charlson', 'mortality_rsi', 'baseline_cvd']

for idx, var in enumerate(vars_to_plot):
    ax = axes[idx//3, idx%3]
    df.boxplot(column=var, by='dow', ax=ax)
    ax.set_title(f'{var} by Day of Week')
    ax.set_xlabel('Day (0=Mon)')
    
plt.tight_layout()
plt.savefig('patient_mix_by_dow.png', dpi=300)


#### MOONPHASE INVESTIGATION:
print('-' * 40)
# What is this variable?
print("Moonphase unique values:", df['moonphase'].unique())
print("Moonphase distribution:", df['moonphase'].value_counts())
print("Moonphase vs mortality:", df.groupby('moonphase')['mort30'].mean())

# Is this a serious variable or a test?
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df['moonphase'], df['mort30'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi-square test: p={p:.4f}")



for column in df.columns:
    unique_values = df[column].unique()
    print(f"--- Column: '{column}' ---")
    print(unique_values)
    print(f"Total unique count: {len(unique_values)}\n")
