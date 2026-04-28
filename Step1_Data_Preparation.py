# =============================================================================
# VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION
# Step 1: Data Collection and Preprocessing
# Author: Sanusi Moshood Olanrewaju | FUOYE
# =============================================================================
import pandas as pd, numpy as np, os, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

print("="*60)
print("  VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION")
print("  Step 1: Data Preparation")
print("="*60)

for f in ['data','models','outputs','notebooks']:
    os.makedirs(f, exist_ok=True)
print("\n✅ Folders confirmed: data/ models/ outputs/ notebooks/")

# ── STAGE 1: Load / Generate ──────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 1: LOADING DATASET\n"+"─"*60)
path = 'data/Metro_Interstate_Traffic_Volume.csv'
if os.path.exists(path):
    df_raw = pd.read_csv(path)
    print(f"✅ Loaded: {path}  ({df_raw.shape[0]:,} rows, {df_raw.shape[1]} cols)")
else:
    print("⚠️  Not found — generating synthetic dataset...")
    np.random.seed(42); n=10000
    segs  = [f'SEG_{i:03d}' for i in range(1,21)]
    hrs   = np.random.randint(0,24,n)
    days  = np.random.choice(['Monday','Tuesday','Wednesday','Thursday',
                               'Friday','Saturday','Sunday'],n)
    sids  = np.random.choice(segs,n)
    bc    = np.random.randint(50,500,n)
    pk    = np.where((hrs>=7)&(hrs<=9),300,0)+np.where((hrs>=16)&(hrs<=19),250,0)
    wk    = np.where(np.isin(days,['Saturday','Sunday']),-100,0)
    vc    = np.clip(bc+pk+wk+np.random.randint(-50,50,n),10,900)
    spd   = np.clip(120-(vc/10)+np.random.randint(-10,10,n),5,120)
    df_raw= pd.DataFrame({'date_time':pd.date_range('2023-01-01',periods=n,freq='h').astype(str),
        'traffic_volume':vc,'holiday':np.random.choice(['None','Holiday'],n,p=[0.97,0.03]),
        'temp':np.random.uniform(15,40,n).round(1),'rain_1h':np.random.choice([0,0,0,.5,1,2.5],n),
        'snow_1h':np.zeros(n),'clouds_all':np.random.randint(0,100,n).astype(float),
        'weather_main':np.random.choice(['Clear','Clouds','Rain','Mist'],n,p=[.5,.3,.15,.05]),
        'road_segment':sids,'hour':hrs,'day_of_week':days,'traffic_speed':spd.round(1)})
    for col in ['traffic_speed','temp','clouds_all']:
        idx=np.random.choice(df_raw.index,size=int(n*.02),replace=False)
        df_raw.loc[idx,col]=np.nan
    df_raw.to_csv(path,index=False)
    print(f"✅ Synthetic dataset saved: {path} ({n:,} rows)")

# ── STAGE 2: Explore ─────────────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 2: EXPLORATION\n"+"─"*60)
print(df_raw.head().to_string())
mv=df_raw.isnull().sum(); mv=mv[mv>0]
print(f"\n🔍 Missing: {dict(mv) if len(mv) else 'None'}")
print(f"🔁 Duplicates: {df_raw.duplicated().sum()}")

# ── STAGE 3: Clean ───────────────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 3: DATA CLEANING\n"+"─"*60)
df=df_raw.copy()
b=len(df); df=df.drop_duplicates(); print(f"✅ Duplicates removed: {b-len(df)}")
df['date_time']=pd.to_datetime(df['date_time'],errors='coerce')
if 'traffic_volume' in df.columns: df.rename(columns={'traffic_volume':'vehicle_count'},inplace=True)
if 'hour' not in df.columns or df['hour'].isnull().all(): df['hour']=df['date_time'].dt.hour
if 'day_of_week' not in df.columns or df['day_of_week'].isnull().all():
    df['day_of_week']=df['date_time'].dt.day_name()
df['month']=df['date_time'].dt.month
if 'road_segment' not in df.columns:
    np.random.seed(42); df['road_segment']=np.random.choice([f'SEG_{i:03d}' for i in range(1,21)],len(df))
    print("✅ road_segment added (simulated)")
for col in df.select_dtypes(include=[np.number]).columns:
    n_m=df[col].isnull().sum()
    if n_m>0: med=df[col].median(); df[col]=df[col].fillna(med); print(f"   Imputed '{col}' median={med:.2f}")
for col in ['day_of_week','weather_main','road_segment']:
    if col in df.columns and df[col].isnull().sum()>0:
        mv=df[col].mode()[0]; df[col]=df[col].fillna(mv); print(f"   Imputed '{col}' mode='{mv}'")
Q1=df['vehicle_count'].quantile(.25); Q3=df['vehicle_count'].quantile(.75); IQR=Q3-Q1
b=len(df); df=df[(df['vehicle_count']>=Q1-1.5*IQR)&(df['vehicle_count']<=Q3+1.5*IQR)].copy()
print(f"✅ Outliers removed: {b-len(df)} | Range: {int(df['vehicle_count'].min())}–{int(df['vehicle_count'].max())}")
print(f"✅ Clean size: {len(df):,}")

# ── STAGE 4: Feature Engineering ─────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 4: FEATURE ENGINEERING\n"+"─"*60)
df['hour']=pd.to_numeric(df['hour'],errors='coerce').fillna(0).astype(int)
df['is_peak_hour']=df['hour'].apply(lambda h:1 if(7<=h<=9)or(16<=h<=19)else 0)
df['is_weekend']=df['day_of_week'].apply(lambda d:1 if str(d) in['Saturday','Sunday']else 0)
sm=df.groupby('road_segment')['vehicle_count'].transform('max').replace(0,1)
df['density_index']=(df['vehicle_count']/sm).round(4)
df=df.sort_values(['road_segment','hour']).reset_index(drop=True)
df['rolling_avg_count']=(df.groupby('road_segment')['vehicle_count']
    .transform(lambda x:x.rolling(3,min_periods=1).mean()).round(2))
seg_mean=df.groupby('road_segment')['vehicle_count'].transform('mean')
df['road_usage_pattern']=pd.cut(seg_mean,bins=3,labels=['Low_Usage','Medium_Usage','High_Usage'])
print("✅ Features: is_peak_hour | is_weekend | density_index | rolling_avg_count | road_usage_pattern")

# ── STAGE 5: Target Variable ──────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 5: TARGET VARIABLE\n"+"─"*60)
labels=[]
for seg,grp in df.groupby('road_segment',sort=False):
    p33=grp['vehicle_count'].quantile(0.33); p66=grp['vehicle_count'].quantile(0.66)
    for v in grp['vehicle_count']:
        labels.append('Low' if v<=p33 else('Medium' if v<=p66 else'High'))
df['density_level']=labels
dist=df['density_level'].value_counts()
print("✅ density_level created:")
for c in['Low','Medium','High']:
    cnt=dist.get(c,0); print(f"   {c:8s}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

# ── STAGE 6: Encode ──────────────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 6: ENCODING\n"+"─"*60)
day_map={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
df['day_encoded']=df['day_of_week'].map(day_map).fillna(0).astype(int)
df['usage_encoded']=df['road_usage_pattern'].map({'Low_Usage':0,'Medium_Usage':1,'High_Usage':2}).fillna(0).astype(int)
df['density_encoded']=df['density_level'].map({'Low':0,'Medium':1,'High':2})
print("✅ Encoded: day_of_week | road_usage_pattern | density_level")

# ── STAGE 7: Scale ───────────────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 7: MIN-MAX SCALING\n"+"─"*60)
sc=['vehicle_count','hour','density_index','rolling_avg_count']
if 'traffic_speed' in df.columns: sc.append('traffic_speed')
for col in sc:
    cmin,cmax=df[col].min(),df[col].max()
    if cmax>cmin: df[f'{col}_scaled']=((df[col]-cmin)/(cmax-cmin)).round(4); print(f"✅ {col}_scaled [0,1]")

# ── STAGE 8: Final Features ───────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 8: FINAL FEATURE SET\n"+"─"*60)
feature_cols=['vehicle_count_scaled','hour_scaled','density_index_scaled',
              'rolling_avg_count_scaled','is_peak_hour','is_weekend','day_encoded','usage_encoded']
if 'traffic_speed_scaled' in df.columns: feature_cols.append('traffic_speed_scaled')
keep=feature_cols+['density_encoded','density_level','road_segment']
df_final=df[keep].dropna().reset_index(drop=True)
print(f"\n✅ {len(feature_cols)} features selected:")
for i,f in enumerate(feature_cols,1): print(f"   {i:2d}. {f}")
print(f"\n✅ Final shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} cols")

# ── STAGE 9: Save ────────────────────────────────────────────────────────────
clean_path='data/clean_traffic_data.csv'
df_final.to_csv(clean_path,index=False)
print(f"\n✅ Saved: {clean_path}")

# ── STAGE 10: Visualise ───────────────────────────────────────────────────────
print("\n"+"─"*60+"\nSTAGE 10: VISUALISATIONS\n"+"─"*60)
fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle('Traffic Data — Exploratory Analysis',fontsize=16,fontweight='bold')

ax=axes[0,0]
cnts=[dist.get(c,0) for c in['Low','Medium','High']]
bars=ax.bar(['Low','Medium','High'],cnts,color=['#2ECC71','#F39C12','#E74C3C'],edgecolor='black',lw=0.8)
ax.set_title('Vehicle Density Level Distribution',fontweight='bold')
ax.set_xlabel('Density Level'); ax.set_ylabel('Count')
for bar,cnt in zip(bars,cnts):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+30,f'{cnt:,}',ha='center',fontsize=10,fontweight='bold')

ax=axes[0,1]
hrly=df.groupby('hour')['vehicle_count'].mean()
ax.plot(hrly.index,hrly.values,color='#2E86AB',lw=2.5,marker='o',ms=4)
ax.fill_between(hrly.index,hrly.values,alpha=0.2,color='#2E86AB')
ax.set_title('Avg Vehicle Count by Hour',fontweight='bold')
ax.set_xlabel('Hour'); ax.set_ylabel('Avg Count'); ax.set_xticks(range(0,24,2))

ax=axes[1,0]
dord=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dly=df.groupby('day_of_week')['vehicle_count'].mean().reindex(dord)
ax.bar(range(7),dly.values,color='#8E44AD',edgecolor='black',lw=0.8)
ax.set_title('Avg Vehicle Count by Day',fontweight='bold')
ax.set_xlabel('Day'); ax.set_ylabel('Avg Count')
ax.set_xticks(range(7)); ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

ax=axes[1,1]
ax.hist(df_final['density_index_scaled'],bins=30,color='#E67E22',edgecolor='black',lw=0.8,alpha=0.85)
ax.set_title('Density Index Distribution (Scaled)',fontweight='bold')
ax.set_xlabel('Density Index (Scaled)'); ax.set_ylabel('Frequency')

plt.tight_layout()
out='outputs/Step1_Exploratory_Analysis.png'
plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close()
print(f"✅ Chart saved: {out}")

print("\n"+"="*60)
print("  STEP 1 COMPLETE")
print("="*60)
print(f"""
  Records (clean)  : {len(df_final):,}
  Features         : {len(feature_cols)}
  Target classes   : Low | Medium | High
  Clean data saved : {clean_path}
  Chart saved      : {out}

  ✅ Ready for Step 2: Model Training
""")
