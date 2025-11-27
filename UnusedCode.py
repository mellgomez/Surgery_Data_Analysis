'''UNUSED CODE'''


# ============================================
#    TIME OF THE DAY × LATE WEEK INTERACTION
# STEP 5: MODEL 3 - TRIPLE INTERACTION (NOVEL ina way but DON'T INCLUDE THIS: because Late_week is weak (shouldn't be included). Only 101 Friday evening high-risk cases. Over-parameterized (too many interaction terms for sparse data)
# ============================================

print("\n" + "="*80)
print("MODEL 3: TRIPLE INTERACTION (Late Week × Time × Procedure)")
print("="*80)
print("Testing: Is Friday evening especially dangerous for high-risk procedures?")

# Create triple interaction term
df['triple_interaction'] = df['late_week'] * df['time_evening'] * df['high_risk_proc']

# Also include all lower-order interactions
df['late_x_highrisk'] = df['late_week'] * df['high_risk_proc']
df['late_x_evening'] = df['late_week'] * df['time_evening']

model3_features = base_features + ['late_week', 'time_afternoon', 'time_evening', 
                                   'high_risk_proc',
                                   'late_x_highrisk', 'late_x_evening', 
                                   'evening_x_highrisk', 
                                   'triple_interaction']

# Prepare data
X_model3 = df[model3_features].dropna()
y_model3 = df.loc[X_model3.index, 'mort30']

# Fit model
model3 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model3.fit(X_model3, y_model3)

# Evaluate
y_pred3 = model3.predict_proba(X_model3)[:, 1]
auc3 = roc_auc_score(y_model3, y_pred3)

print(f"\nModel 3 AUC: {auc3:.4f}")
print(f"AUC Improvement over Model 2: {auc3 - auc2:.4f}")

print("\nKey Coefficients:")
print(f"{'Variable':<35} {'OR':<12}")
print("-"*47)

for i, var in enumerate(model3_features):
    or_val = np.exp(model3.coef_[0][i])
    
    if var == 'triple_interaction':
        marker = " ← TRIPLE INTERACTION"
    elif '_x_' in var:
        marker = " ← 2-way interaction"
    else:
        marker = ""
    
    print(f"{var:<35} {or_val:<12.4f}{marker}")

# Interpret triple interaction
triple_or = np.exp(model3.coef_[0][model3_features.index('triple_interaction')])

print("\n" + "*"*80)
print("TRIPLE INTERACTION INTERPRETATION:")
print("*"*80)
print(f"Late Week × Evening × High-Risk: OR = {triple_or:.4f}")

if triple_or > 1.5:
    print("\n✓✓ MAJOR FINDING: Friday evening is ESPECIALLY dangerous for high-risk procedures")
    print("   → Strong synergistic effect of combined fatigue factors")
elif triple_or > 1.2:
    print("\n✓ FINDING: Friday evening shows moderately higher risk for high-risk procedures")
elif triple_or < 0.8:
    print("\n→ Protective effect (counterintuitive)")
else:
    print("\n→ No evidence of synergistic effect")


# ============================================
# Summary findingds from the day of the week analysis
# STEP 8: SUMMARY
# ============================================

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print("\n1. Time-of-Day Main Effect:")
afternoon_or_main = np.exp(model1.coef_[0][model1_features.index('time_afternoon')])
evening_or_main = np.exp(model1.coef_[0][model1_features.index('time_evening')])
print(f"   Afternoon vs Morning: OR = {afternoon_or_main:.4f}")
print(f"   Evening vs Morning:   OR = {evening_or_main:.4f}")

print("\n2. Time × Procedure Interaction:")
print(f"   Evening effect for High-Risk: OR = {evening_int_or:.4f}")

print("\n3. Triple Interaction (Friday × Evening × High-Risk):")
print(f"   Synergistic effect: OR = {triple_or:.4f}")

print("\n4. Model Performance:")
print(f"   Model 1 (Time only):       AUC = {auc1:.4f}")
print(f"   Model 2 (+ Proc interact): AUC = {auc2:.4f}")
print(f"   Model 3 (+ Triple):        AUC = {auc3:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)