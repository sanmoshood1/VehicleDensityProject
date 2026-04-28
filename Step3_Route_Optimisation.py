# =============================================================================
# VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION
# Step 3: Graph-Based Route Optimisation using Dijkstra's Algorithm
# Author: Sanusi Moshood Olanrewaju | FUOYE
# =============================================================================
import numpy as np, pandas as pd, pickle, os, json, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
warnings.filterwarnings('ignore')

print("="*60)
print("  VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION")
print("  Step 3: Route Optimisation")
print("="*60)
print("\n✅ All libraries imported successfully.\n")

# =============================================================================
# STAGE 1: LOAD TRAINED MODEL AND DATA
# =============================================================================
print("─"*60+"\nSTAGE 1: LOADING TRAINED MODEL\n"+"─"*60)

for r in ['models/random_forest_model.pkl','models/feature_cols.pkl',
          'data/clean_traffic_data.csv']:
    if not os.path.exists(r):
        print(f"❌ Missing: {r}. Run Step1 and Step2 first."); exit()

with open('models/random_forest_model.pkl','rb') as f: rf_model  = pickle.load(f)
with open('models/feature_cols.pkl','rb') as f:        feature_cols = pickle.load(f)
df = pd.read_csv('data/clean_traffic_data.csv')

print(f"✅ Random Forest model loaded")
print(f"✅ Features: {len(feature_cols)} columns")
print(f"✅ Dataset : {len(df):,} records")
print(f"✅ Density distribution in dataset:")
for lbl,cnt in df['density_level'].value_counts().items():
    print(f"   {lbl:8s}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

# =============================================================================
# STAGE 2: BUILD ROAD NETWORK GRAPH
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 2: BUILDING ROAD NETWORK GRAPH\n"+"─"*60)

nodes = {
    'A':(0,4),'B':(2,6),'C':(4,6),'D':(6,6),
    'E':(2,4),'F':(4,4),'G':(6,4),'H':(2,2),
    'I':(4,2),'J':(6,2),'K':(0,2),'L':(8,4),
    'M':(8,6),'N':(8,2),'O':(4,0),
}

edges_raw = [
    ('A','B',2.5),('A','E',1.8),('A','K',2.0),
    ('B','C',2.0),('B','E',2.2),
    ('C','D',2.0),('C','F',1.5),('C','M',3.5),
    ('D','G',1.8),('D','M',2.0),
    ('E','F',2.0),('E','H',1.8),
    ('F','G',2.0),('F','I',1.5),
    ('G','J',1.8),('G','L',2.5),('G','M',2.0),
    ('H','I',2.0),('H','K',1.5),
    ('I','J',2.0),('I','O',2.5),
    ('J','N',1.8),('J','L',2.8),
    ('K','O',3.0),
    ('L','M',2.0),('L','N',2.0),
    ('N','O',3.2),
]

G_base = nx.Graph()
G_base.add_nodes_from(nodes.keys())
for u,v,w in edges_raw:
    G_base.add_edge(u,v,base_weight=w,weight=w)

print(f"\n✅ Graph: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges")
print(f"   Total network distance: {sum(d for _,_,d in edges_raw):.1f} km")

# =============================================================================
# STAGE 3: PREDICT DENSITY PER ROAD SEGMENT
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 3: PREDICTING DENSITY PER ROAD SEGMENT\n"+"─"*60)

label_decode  = {0:'Low',1:'Medium',2:'High'}
penalty_vals  = {'Low':1.0,'Medium':1.5,'High':2.5}
color_map     = {'Low':'#2ECC71','Medium':'#F39C12','High':'#E74C3C'}

road_segs     = df['road_segment'].unique()
edges_list    = list(G_base.edges())

# Assign realistic density by sampling individual records (not averages)
np.random.seed(42)
density_map = {}
density_num = {}
penalty_map = {}

# Separate records by density class for controlled assignment
low_recs  = df[df['density_level']=='Low'][feature_cols].values
med_recs  = df[df['density_level']=='Medium'][feature_cols].values
high_recs = df[df['density_level']=='High'][feature_cols].values

# Assign density classes to edges with realistic distribution
# ~33% Low, ~33% Medium, ~34% High — matching dataset distribution
n_edges   = len(edges_list)
n_low     = n_edges // 3
n_med     = n_edges // 3
n_high    = n_edges - n_low - n_med

# Shuffle and assign
assignments = ['Low']*n_low + ['Medium']*n_med + ['High']*n_high
np.random.shuffle(assignments)

print(f"\n{'Edge':<10} {'Assigned Density':<18} {'ML Prediction':<16} {'Penalty'}")
print("─"*55)

for edge, assigned in zip(edges_list, assignments):
    # Pick a representative record matching the assigned density
    if assigned == 'Low':
        rec = low_recs[np.random.randint(len(low_recs))].reshape(1,-1)
    elif assigned == 'Medium':
        rec = med_recs[np.random.randint(len(med_recs))].reshape(1,-1)
    else:
        rec = high_recs[np.random.randint(len(high_recs))].reshape(1,-1)

    pred  = rf_model.predict(rec)[0]
    label = label_decode[int(pred)]

    density_map[edge] = label
    density_num[edge] = int(pred)
    penalty_map[edge] = penalty_vals[label]

    print(f"  {edge[0]+'→'+edge[1]:<10} {assigned:<18} {label:<16} {penalty_vals[label]:.1f}x")

low_cnt  = sum(1 for v in density_map.values() if v=='Low')
med_cnt  = sum(1 for v in density_map.values() if v=='Medium')
high_cnt = sum(1 for v in density_map.values() if v=='High')

print(f"\n📊 Final Density Distribution:")
print(f"   Low    : {low_cnt} segments ({low_cnt/n_edges*100:.0f}%)")
print(f"   Medium : {med_cnt} segments ({med_cnt/n_edges*100:.0f}%)")
print(f"   High   : {high_cnt} segments ({high_cnt/n_edges*100:.0f}%)")

# =============================================================================
# STAGE 4: APPLY PENALTY WEIGHTS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 4: APPLYING DENSITY PENALTIES\n"+"─"*60)

G_aware = G_base.copy()
for u,v in G_aware.edges():
    edge = (u,v) if (u,v) in penalty_map else (v,u)
    base = G_aware[u][v]['base_weight']
    pen  = penalty_map.get(edge,1.0)
    G_aware[u][v]['weight'] = round(base*pen,3)

print("\n✅ Penalty weights applied:")
print("   Low    → × 1.0 (free flow, no penalty)")
print("   Medium → × 1.5 (moderate congestion)")
print("   High   → × 2.5 (severe congestion)")

# =============================================================================
# STAGE 5: ROUTE OPTIMISATION
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 5: ROUTE OPTIMISATION — DIJKSTRA'S ALGORITHM\n"+"─"*60)

def get_edge_density(path, density_map):
    counts = {'Low':0,'Medium':0,'High':0}
    for i in range(len(path)-1):
        u,v = path[i],path[i+1]
        edge = (u,v) if (u,v) in density_map else (v,u)
        lbl  = density_map.get(edge,'Low')
        counts[lbl] += 1
    return counts

scenarios = [
    ('A','L','Scenario 1: North-West to East'),
    ('K','M','Scenario 2: South-West to North-East'),
    ('A','N','Scenario 3: North-West to South-East'),
]

all_results = []

for origin,dest,label in scenarios:
    print(f"\n{'─'*52}\n  {label}")
    print(f"  Origin: {origin}   →   Destination: {dest}\n{'─'*52}")

    static_path = nx.dijkstra_path(G_base,  origin, dest, weight='base_weight')
    static_cost = nx.dijkstra_path_length(G_base, origin, dest, weight='base_weight')
    aware_path  = nx.dijkstra_path(G_aware, origin, dest, weight='weight')
    aware_cost  = nx.dijkstra_path_length(G_aware, origin, dest, weight='weight')

    # Base distance of aware path
    aware_base = sum(G_base[aware_path[i]][aware_path[i+1]]['base_weight']
                     for i in range(len(aware_path)-1))

    s_dens = get_edge_density(static_path, density_map)
    a_dens = get_edge_density(aware_path,  density_map)
    overhead = (aware_base - static_cost) / static_cost * 100

    print(f"  STATIC ROUTE  : {' → '.join(static_path)}")
    print(f"  Distance      : {static_cost:.2f} km")
    print(f"  Density       : Low={s_dens['Low']} | Medium={s_dens['Medium']} | High={s_dens['High']}")

    print(f"\n  AWARE ROUTE   : {' → '.join(aware_path)}")
    print(f"  Distance      : {aware_base:.2f} km")
    print(f"  Density       : Low={a_dens['Low']} | Medium={a_dens['Medium']} | High={a_dens['High']}")
    print(f"  Dist overhead : {overhead:+.1f}%")

    avoided = s_dens['High'] - a_dens['High']
    print(f"\n  ✅ High-density segments avoided: {avoided}")
    if avoided > 0:
        print(f"  ✅ Density-aware routing successfully reduced congestion exposure!")
    else:
        print(f"  ℹ️  Optimal path same or both routes equally congested.")

    all_results.append({
        'scenario'         : label,
        'origin'           : origin, 'dest': dest,
        'static_path'      : ' → '.join(static_path),
        'static_dist'      : round(static_cost,2),
        'static_low'       : s_dens['Low'],
        'static_med'       : s_dens['Medium'],
        'static_high'      : s_dens['High'],
        'aware_path'       : ' → '.join(aware_path),
        'aware_dist'       : round(aware_base,2),
        'aware_low'        : a_dens['Low'],
        'aware_med'        : a_dens['Medium'],
        'aware_high'       : a_dens['High'],
        'high_avoided'     : avoided,
        'dist_overhead_pct': round(overhead,1),
    })

# =============================================================================
# STAGE 6: SAVE RESULTS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 6: SAVING RESULTS\n"+"─"*60)
os.makedirs('outputs',exist_ok=True)
with open('outputs/routing_results.json','w') as f: json.dump(all_results,f,indent=2)
print("✅ Saved: outputs/routing_results.json")

density_export = {f"{k[0]}-{k[1]}":v for k,v in density_map.items()}
with open('outputs/edge_density_map.json','w') as f: json.dump(density_export,f,indent=2)
print("✅ Saved: outputs/edge_density_map.json")

# =============================================================================
# STAGE 7: VISUALISATIONS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 7: GENERATING CHARTS\n"+"─"*60)

# Chart 1: Full road network coloured by density
fig,ax = plt.subplots(figsize=(13,9))
ax.set_facecolor('#F8F9FA'); fig.patch.set_facecolor('#F8F9FA')

for u,v in G_base.edges():
    edge  = (u,v) if (u,v) in density_map else (v,u)
    label = density_map.get(edge,'Low')
    col   = color_map[label]
    x0,y0 = nodes[u]; x1,y1 = nodes[v]
    ax.plot([x0,x1],[y0,y1],color=col,linewidth=5,alpha=0.85,zorder=1)
    mx,my  = (x0+x1)/2,(y0+y1)/2
    bw     = G_base[u][v]['base_weight']
    ax.text(mx,my,f'{bw:.1f}km',fontsize=7.5,ha='center',va='center',
            bbox=dict(boxstyle='round,pad=0.2',facecolor='white',alpha=0.8,edgecolor='none'))

for node,(x,y) in nodes.items():
    ax.scatter(x,y,s=700,c='#2E4057',zorder=3,edgecolors='white',linewidths=2.5)
    ax.text(x,y,node,fontsize=12,ha='center',va='center',
            color='white',fontweight='bold',zorder=4)

patches = [mpatches.Patch(color=c,label=f'{l} Density ({[low_cnt,med_cnt,high_cnt][i]} segs)')
           for i,(l,c) in enumerate(color_map.items())]
ax.legend(handles=patches,loc='upper right',fontsize=11,framealpha=0.92,edgecolor='grey')
ax.set_title('Road Network — Predicted Vehicle Density by Segment\n'
             '(Green=Low | Orange=Medium | Red=High)',
             fontsize=14,fontweight='bold',pad=15)
ax.set_xlim(-0.8,9.2); ax.set_ylim(-0.8,7.5); ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/Step3_Road_Network_Density.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step3_Road_Network_Density.png")

# Chart 2: Route comparison — Scenario 1
sc = all_results[0]
s_path = sc['static_path'].split(' → ')
a_path = sc['aware_path'].split(' → ')

fig,axes = plt.subplots(1,2,figsize=(16,8))
fig.suptitle(f"Route Comparison — {sc['scenario']}",fontsize=14,fontweight='bold')

for ax,path,title,dist,hi in [
    (axes[0],s_path,'Static Route\n(No Density Awareness)',sc['static_dist'],sc['static_high']),
    (axes[1],a_path,'Density-Aware Route\n(With ML Predictions)',sc['aware_dist'],sc['aware_high']),
]:
    ax.set_facecolor('#F8F9FA')
    for u,v in G_base.edges():
        edge  = (u,v) if (u,v) in density_map else (v,u)
        lbl   = density_map.get(edge,'Low')
        col   = color_map[lbl]
        x0,y0 = nodes[u]; x1,y1 = nodes[v]
        ax.plot([x0,x1],[y0,y1],color=col,linewidth=3,alpha=0.35,zorder=1)

    for i in range(len(path)-1):
        u,v   = path[i],path[i+1]
        x0,y0 = nodes[u]; x1,y1 = nodes[v]
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),
            arrowprops=dict(arrowstyle='-|>',color='#2E4057',lw=3.0),zorder=3)

    for node,(x,y) in nodes.items():
        col = '#E74C3C' if node in [path[0],path[-1]] else \
              '#F39C12' if node in path else '#2E4057'
        size = 700 if node in path else 400
        ax.scatter(x,y,s=size,c=col,zorder=4,edgecolors='white',linewidths=2)
        ax.text(x,y,node,fontsize=10,ha='center',va='center',
                color='white',fontweight='bold',zorder=5)

    patches = [mpatches.Patch(color=c,label=f'{l} Density')
               for l,c in color_map.items()]
    ax.legend(handles=patches,loc='upper right',fontsize=9,framealpha=0.9)
    ax.set_title(f"{title}\nDistance: {dist:.2f} km  |  High-Density Segs: {hi}",
                 fontsize=11,fontweight='bold')
    ax.set_xlim(-0.8,9.2); ax.set_ylim(-0.8,7.5); ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/Step3_Route_Comparison.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step3_Route_Comparison.png")

# Chart 3: All Scenarios Summary
labels  = [f"Scenario {i+1}" for i in range(len(all_results))]
s_highs = [r['static_high'] for r in all_results]
a_highs = [r['aware_high']  for r in all_results]
s_dists = [r['static_dist'] for r in all_results]
a_dists = [r['aware_dist']  for r in all_results]

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,6))
fig.suptitle('Route Optimisation — All Scenarios Summary',fontsize=14,fontweight='bold')
x=np.arange(len(labels)); w=0.35

ax1.bar(x-w/2,s_dists,w,label='Static Route',color='#E74C3C',edgecolor='black',lw=0.8)
ax1.bar(x+w/2,a_dists,w,label='Density-Aware',color='#2ECC71',edgecolor='black',lw=0.8)
ax1.set_ylabel('Route Distance (km)',fontsize=11)
ax1.set_title('Route Distance Comparison',fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.legend(); ax1.yaxis.grid(True,linestyle='--',alpha=0.6); ax1.set_axisbelow(True)
for i,(s,a) in enumerate(zip(s_dists,a_dists)):
    ax1.text(i-w/2,s+0.05,f'{s:.1f}km',ha='center',fontsize=9,fontweight='bold')
    ax1.text(i+w/2,a+0.05,f'{a:.1f}km',ha='center',fontsize=9,fontweight='bold')

ax2.bar(x-w/2,s_highs,w,label='Static Route',color='#E74C3C',edgecolor='black',lw=0.8)
ax2.bar(x+w/2,a_highs,w,label='Density-Aware',color='#2ECC71',edgecolor='black',lw=0.8)
ax2.set_ylabel('High-Density Segments on Route',fontsize=11)
ax2.set_title('High-Density Segments Encountered',fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.legend(); ax2.yaxis.grid(True,linestyle='--',alpha=0.6); ax2.set_axisbelow(True)
for i,(s,a) in enumerate(zip(s_highs,a_highs)):
    ax2.text(i-w/2,s+0.02,str(s),ha='center',fontsize=10,fontweight='bold')
    ax2.text(i+w/2,a+0.02,str(a),ha='center',fontsize=10,fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/Step3_Scenarios_Summary.png',dpi=150,bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step3_Scenarios_Summary.png")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
print(f"""
{"="*60}
  STEP 3 COMPLETE — ROUTE OPTIMISATION SUMMARY
{"="*60}

  Road Network : {G_base.number_of_nodes()} nodes | {G_base.number_of_edges()} edges
  Density      : Low={low_cnt} | Medium={med_cnt} | High={high_cnt} segments

  ┌────────────┬──────────────────┬──────────────────┬──────────┐
  │ Scenario   │ Static Route     │ Aware Route      │ Avoided  │
  ├────────────┼──────────────────┼──────────────────┼──────────┤""")

for r in all_results:
    sc = r['scenario'].split(':')[0]
    print(f"  │ {sc:<10} │ {r['static_dist']:.2f}km ({r['static_high']} high) │"
          f" {r['aware_dist']:.2f}km ({r['aware_high']} high)  │ {r['high_avoided']:+d} segs  │")

print(f"""  └────────────┴──────────────────┴──────────────────┴──────────┘

  Results : outputs/routing_results.json
  Charts  : outputs/Step3_*.png

  ✅ Ready for Step 4: System Integration
""")
