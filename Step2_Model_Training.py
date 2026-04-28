# =============================================================================
# VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION
# Step 2: Machine Learning Model Training and Evaluation
# Author: Sanusi Moshood Olanrewaju | FUOYE
# =============================================================================
import pandas as pd, numpy as np, pickle, os, json, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("="*60)
print("  VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION")
print("  Step 2: Model Training and Evaluation")
print("="*60)
print("\n✅ All libraries imported successfully.\n")

# =============================================================================
# STAGE 1: LOAD CLEAN DATASET
# =============================================================================
print("─"*60+"\nSTAGE 1: LOADING CLEAN DATASET\n"+"─"*60)

clean_path = 'data/clean_traffic_data.csv'
if not os.path.exists(clean_path):
    print("❌ clean_traffic_data.csv not found. Run Step1 first."); exit()

df = pd.read_csv(clean_path)
print(f"\n✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

all_possible = ['vehicle_count_scaled','hour_scaled','density_index_scaled',
                'rolling_avg_count_scaled','is_peak_hour','is_weekend',
                'day_encoded','usage_encoded','traffic_speed_scaled']
feature_cols = [c for c in all_possible if c in df.columns]
class_names  = ['Low','Medium','High']

X = df[feature_cols].values
y = df['density_encoded'].values

print(f"\n📋 Features ({len(feature_cols)}): {', '.join(feature_cols)}")
print(f"\n📊 Class distribution:")
for i,n in enumerate(class_names):
    cnt=(y==i).sum(); print(f"   {n:8s}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

# =============================================================================
# STAGE 2: TRAIN / TEST SPLIT
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 2: TRAIN / TEST SPLIT (80:20 Stratified)\n"+"─"*60)

X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n✅ Training set : {X_train.shape[0]:,} samples")
print(f"✅ Testing set  : {X_test.shape[0]:,} samples")

# =============================================================================
# STAGE 3: DECISION TREE — TRAINING
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 3: DECISION TREE — TRAINING\n"+"─"*60)

# Well-tuned parameters (validated through cross-validation analysis)
dt_params = {'max_depth':10, 'min_samples_split':5,
             'min_samples_leaf':2, 'criterion':'gini', 'random_state':42}

print(f"\n📌 Parameters: {dt_params}")
dt_model = DecisionTreeClassifier(**dt_params)
dt_model.fit(X_train, y_train)
print(f"✅ Decision Tree trained")
print(f"   Tree depth : {dt_model.get_depth()}")
print(f"   Leaf nodes : {dt_model.get_n_leaves()}")

# =============================================================================
# STAGE 4: RANDOM FOREST — TRAINING
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 4: RANDOM FOREST — TRAINING\n"+"─"*60)

rf_params = {'n_estimators':200, 'max_depth':20, 'max_features':'sqrt',
             'min_samples_leaf':1, 'random_state':42, 'n_jobs':-1}

print(f"\n📌 Parameters: {rf_params}")
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train)
print(f"✅ Random Forest trained")
print(f"   Estimators : {rf_model.n_estimators}")

# =============================================================================
# STAGE 5: EVALUATION
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 5: MODEL EVALUATION ON TEST SET\n"+"─"*60)

def evaluate(model, name, Xt, yt):
    yp = model.predict(Xt)
    return {
        'name': name, 'y_pred': yp,
        'accuracy' : accuracy_score(yt, yp),
        'precision': precision_score(yt, yp, average='macro', zero_division=0),
        'recall'   : recall_score(yt, yp, average='macro', zero_division=0),
        'f1'       : f1_score(yt, yp, average='macro', zero_division=0),
        'cm'       : confusion_matrix(yt, yp),
    }

dt_res = evaluate(dt_model, 'Decision Tree', X_test, y_test)
rf_res = evaluate(rf_model, 'Random Forest', X_test, y_test)

for res in [dt_res, rf_res]:
    print(f"\n{'─'*42}")
    print(f"  {res['name']}")
    print(f"{'─'*42}")
    print(f"  Accuracy  : {res['accuracy']*100:.2f}%")
    print(f"  Precision : {res['precision']*100:.2f}%")
    print(f"  Recall    : {res['recall']*100:.2f}%")
    print(f"  F1-Score  : {res['f1']*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, res['y_pred'], target_names=class_names))

# =============================================================================
# STAGE 6: CROSS-VALIDATION
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 6: 5-FOLD CROSS-VALIDATION\n"+"─"*60)
print("🔍 Running cross-validation (this may take a minute)...")

dt_cv = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

print(f"\n✅ Decision Tree CV: {dt_cv.mean()*100:.2f}% ± {dt_cv.std()*100:.2f}%")
print(f"   Fold scores: {[round(s*100,2) for s in dt_cv]}")
print(f"\n✅ Random Forest CV: {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")
print(f"   Fold scores: {[round(s*100,2) for s in rf_cv]}")

# =============================================================================
# STAGE 7: SAVE MODELS AND RESULTS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 7: SAVING MODELS AND RESULTS\n"+"─"*60)

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

with open('models/decision_tree_model.pkl','wb') as f: pickle.dump(dt_model,f)
with open('models/random_forest_model.pkl','wb') as f: pickle.dump(rf_model,f)
with open('models/feature_cols.pkl','wb') as f: pickle.dump(feature_cols,f)
print("✅ Models saved to models/")

results_summary = {
    'dt_accuracy' : round(dt_res['accuracy']*100,2),
    'dt_precision': round(dt_res['precision']*100,2),
    'dt_recall'   : round(dt_res['recall']*100,2),
    'dt_f1'       : round(dt_res['f1']*100,2),
    'dt_cv_mean'  : round(dt_cv.mean()*100,2),
    'dt_cv_std'   : round(dt_cv.std()*100,2),
    'rf_accuracy' : round(rf_res['accuracy']*100,2),
    'rf_precision': round(rf_res['precision']*100,2),
    'rf_recall'   : round(rf_res['recall']*100,2),
    'rf_f1'       : round(rf_res['f1']*100,2),
    'rf_cv_mean'  : round(rf_cv.mean()*100,2),
    'rf_cv_std'   : round(rf_cv.std()*100,2),
    'feature_cols': feature_cols,
    'n_train'     : int(X_train.shape[0]),
    'n_test'      : int(X_test.shape[0]),
    'dt_params'   : {k:str(v) for k,v in dt_params.items()},
    'rf_params'   : {k:str(v) for k,v in rf_params.items()},
}
with open('outputs/model_results.json','w') as f: json.dump(results_summary,f,indent=2)
print("✅ Results saved to outputs/model_results.json")

# =============================================================================
# STAGE 8: VISUALISATIONS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 8: GENERATING CHARTS\n"+"─"*60)

# Chart 1: Confusion Matrices
fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Confusion Matrices — Decision Tree vs Random Forest',
             fontsize=15, fontweight='bold')
for ax, res in zip(axes,[dt_res,rf_res]):
    cm_n = res['cm'].astype('float')/res['cm'].sum(axis=1)[:,np.newaxis]
    sns.heatmap(cm_n, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='grey',
                annot_kws={'size':13,'weight':'bold'})
    ax.set_title(f"{res['name']}\nAccuracy: {res['accuracy']*100:.2f}%",
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted Label',fontsize=11)
    ax.set_ylabel('True Label',fontsize=11)
plt.tight_layout()
plt.savefig('outputs/Step2_Confusion_Matrices.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step2_Confusion_Matrices.png")

# Chart 2: Performance Comparison
metrics   = ['Accuracy','Precision','Recall','F1-Score']
dt_scores = [dt_res['accuracy'],dt_res['precision'],dt_res['recall'],dt_res['f1']]
rf_scores = [rf_res['accuracy'],rf_res['precision'],rf_res['recall'],rf_res['f1']]
x=np.arange(len(metrics)); w=0.35
fig,ax=plt.subplots(figsize=(11,6))
b1=ax.bar(x-w/2,[s*100 for s in dt_scores],w,label='Decision Tree',
          color='#3498DB',edgecolor='black',lw=0.8)
b2=ax.bar(x+w/2,[s*100 for s in rf_scores],w,label='Random Forest',
          color='#E74C3C',edgecolor='black',lw=0.8)
ax.set_ylabel('Score (%)',fontsize=12)
ax.set_title('Model Performance Comparison: Decision Tree vs Random Forest',
             fontsize=14,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=12)
ax.set_ylim(0,115); ax.legend(fontsize=11)
ax.yaxis.grid(True,linestyle='--',alpha=0.7); ax.set_axisbelow(True)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
            f'{bar.get_height():.1f}%',ha='center',va='bottom',
            fontsize=10,fontweight='bold',color='#2980B9')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
            f'{bar.get_height():.1f}%',ha='center',va='bottom',
            fontsize=10,fontweight='bold',color='#C0392B')
plt.tight_layout()
plt.savefig('outputs/Step2_Performance_Comparison.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step2_Performance_Comparison.png")

# Chart 3: Feature Importance
imps    = rf_model.feature_importances_
idx     = np.argsort(imps)[::-1]
lmap    = {'vehicle_count_scaled':'Vehicle Count','hour_scaled':'Hour of Day',
           'density_index_scaled':'Density Index','rolling_avg_count_scaled':'Rolling Avg Count',
           'is_peak_hour':'Peak Hour','is_weekend':'Weekend',
           'day_encoded':'Day of Week','usage_encoded':'Road Usage Pattern',
           'traffic_speed_scaled':'Traffic Speed'}
sfeat   = [lmap.get(feature_cols[i],feature_cols[i]) for i in idx]
simp    = imps[idx]
fig,ax  = plt.subplots(figsize=(11,6))
colors  = plt.cm.RdYlGn(np.linspace(0.3,0.9,len(sfeat)))
ax.barh(sfeat[::-1],simp[::-1]*100,color=colors,edgecolor='black',lw=0.8)
ax.set_xlabel('Feature Importance (%)',fontsize=12)
ax.set_title('Random Forest — Feature Importance Rankings',
             fontsize=14,fontweight='bold')
ax.xaxis.grid(True,linestyle='--',alpha=0.7); ax.set_axisbelow(True)
for i,(name,val) in enumerate(zip(sfeat[::-1],simp[::-1])):
    ax.text(val*100+0.2,i,f'{val*100:.1f}%',va='center',fontsize=10,fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Step2_Feature_Importance.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step2_Feature_Importance.png")

# Chart 4: Cross-Validation Scores
folds=['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5']
x=np.arange(5); w=0.35
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(x-w/2,dt_cv*100,w,label='Decision Tree',color='#3498DB',edgecolor='black',lw=0.8)
ax.bar(x+w/2,rf_cv*100,w,label='Random Forest',color='#E74C3C',edgecolor='black',lw=0.8)
ax.axhline(dt_cv.mean()*100,color='#2980B9',linestyle='--',lw=1.5,
           label=f'DT Mean: {dt_cv.mean()*100:.2f}%')
ax.axhline(rf_cv.mean()*100,color='#C0392B',linestyle='--',lw=1.5,
           label=f'RF Mean: {rf_cv.mean()*100:.2f}%')
ax.set_ylabel('Accuracy (%)',fontsize=12)
ax.set_title('5-Fold Cross-Validation Accuracy',fontsize=14,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(folds)
ax.set_ylim(0,115); ax.legend(fontsize=10)
ax.yaxis.grid(True,linestyle='--',alpha=0.7); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('outputs/Step2_CrossValidation.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step2_CrossValidation.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"""
{"="*60}
  STEP 2 COMPLETE — RESULTS SUMMARY
{"="*60}

  ┌─────────────────────┬──────────────┬──────────────┐
  │ Metric              │ Dec. Tree    │ Rand. Forest │
  ├─────────────────────┼──────────────┼──────────────┤
  │ Accuracy            │ {dt_res['accuracy']*100:>8.2f}%   │ {rf_res['accuracy']*100:>8.2f}%   │
  │ Precision (macro)   │ {dt_res['precision']*100:>8.2f}%   │ {rf_res['precision']*100:>8.2f}%   │
  │ Recall (macro)      │ {dt_res['recall']*100:>8.2f}%   │ {rf_res['recall']*100:>8.2f}%   │
  │ F1-Score (macro)    │ {dt_res['f1']*100:>8.2f}%   │ {rf_res['f1']*100:>8.2f}%   │
  │ CV Accuracy (mean)  │ {dt_cv.mean()*100:>8.2f}%   │ {rf_cv.mean()*100:>8.2f}%   │
  │ CV Std Dev          │ {dt_cv.std()*100:>8.2f}%   │ {rf_cv.std()*100:>8.2f}%   │
  └─────────────────────┴──────────────┴──────────────┘

  Models saved  : models/
  Charts saved  : outputs/
  Results JSON  : outputs/model_results.json

  ✅ Ready for Step 3: Route Optimisation
""")
