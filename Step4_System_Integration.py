# =============================================================================
# VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION
# Step 4: Full System Integration
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
print("  Step 4: Full System Integration")
print("="*60)
print("""
  This script runs the COMPLETE system end-to-end:
  ┌──────────────────────────────────────────────────┐
  │  INPUT: Road segment + time conditions           │
  │    ↓                                             │
  │  PREDICTION: ML model classifies density         │
  │    ↓                                             │
  │  ROUTING: Dijkstra finds optimal route           │
  │    ↓                                             │
  │  OUTPUT: Recommended route + density map         │
  └──────────────────────────────────────────────────┘
""")

# =============================================================================
# STAGE 1: LOAD ALL COMPONENTS
# =============================================================================
print("─"*60+"\nSTAGE 1: LOADING SYSTEM COMPONENTS\n"+"─"*60)

required = ['models/random_forest_model.pkl',
            'models/decision_tree_model.pkl',
            'models/feature_cols.pkl',
            'data/clean_traffic_data.csv',
            'outputs/model_results.json']
for r in required:
    if not os.path.exists(r):
        print(f"❌ Missing: {r}\n   Run Steps 1–3 first."); exit()

with open('models/random_forest_model.pkl','rb') as f:  rf_model  = pickle.load(f)
with open('models/decision_tree_model.pkl','rb') as f:  dt_model  = pickle.load(f)
with open('models/feature_cols.pkl','rb') as f:         feature_cols = pickle.load(f)
with open('outputs/model_results.json','r') as f:       model_results = json.load(f)
df = pd.read_csv('data/clean_traffic_data.csv')

print(f"✅ Random Forest model    — loaded")
print(f"✅ Decision Tree model    — loaded")
print(f"✅ Feature columns        — {len(feature_cols)} features")
print(f"✅ Traffic dataset        — {len(df):,} records")
print(f"✅ Model results          — loaded")

# =============================================================================
# STAGE 2: DEFINE ROAD NETWORK
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 2: INITIALISING ROAD NETWORK\n"+"─"*60)

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

print(f"✅ Road network: {G_base.number_of_nodes()} intersections, "
      f"{G_base.number_of_edges()} road segments")
print(f"   Valid origins/destinations: {', '.join(sorted(nodes.keys()))}")

# =============================================================================
# STAGE 3: DENSITY PREDICTION ENGINE
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 3: DENSITY PREDICTION ENGINE\n"+"─"*60)

label_decode = {0:'Low', 1:'Medium', 2:'High'}
penalty_vals = {'Low':1.0, 'Medium':1.5, 'High':2.5}
color_map    = {'Low':'#2ECC71', 'Medium':'#F39C12', 'High':'#E74C3C'}

# Separate dataset by density class for realistic edge assignment
low_recs  = df[df['density_level']=='Low'][feature_cols].values
med_recs  = df[df['density_level']=='Medium'][feature_cols].values
high_recs = df[df['density_level']=='High'][feature_cols].values

def predict_network_density(seed=42):
    """Predict density level for every edge in the road network."""
    np.random.seed(seed)
    edges_list  = list(G_base.edges())
    n_edges     = len(edges_list)
    n_low       = n_edges // 3
    n_med       = n_edges // 3
    n_high      = n_edges - n_low - n_med
    assignments = ['Low']*n_low + ['Medium']*n_med + ['High']*n_high
    np.random.shuffle(assignments)

    density_map = {}
    for edge, assigned in zip(edges_list, assignments):
        if assigned == 'Low':
            rec = low_recs[np.random.randint(len(low_recs))].reshape(1,-1)
        elif assigned == 'Medium':
            rec = med_recs[np.random.randint(len(med_recs))].reshape(1,-1)
        else:
            rec = high_recs[np.random.randint(len(high_recs))].reshape(1,-1)
        pred = rf_model.predict(rec)[0]
        density_map[edge] = label_decode[int(pred)]
    return density_map

def build_aware_graph(density_map):
    """Build a graph with density-penalised edge weights."""
    G = G_base.copy()
    for u,v in G.edges():
        edge = (u,v) if (u,v) in density_map else (v,u)
        base = G[u][v]['base_weight']
        pen  = penalty_vals.get(density_map.get(edge,'Low'), 1.0)
        G[u][v]['weight'] = round(base * pen, 3)
    return G

def get_route_stats(path, density_map):
    """Compute density breakdown and base distance for a given path."""
    counts = {'Low':0,'Medium':0,'High':0}
    dist   = 0.0
    for i in range(len(path)-1):
        u,v    = path[i],path[i+1]
        edge   = (u,v) if (u,v) in density_map else (v,u)
        lbl    = density_map.get(edge,'Low')
        counts[lbl] += 1
        dist  += G_base[u][v]['base_weight']
    return counts, round(dist,2)

print("✅ Prediction engine initialised")
print(f"   Low  records available : {len(low_recs):,}")
print(f"   Med  records available : {len(med_recs):,}")
print(f"   High records available : {len(high_recs):,}")

# =============================================================================
# STAGE 4: INTERACTIVE ROUTING FUNCTION
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 4: ROUTE QUERY ENGINE\n"+"─"*60)

def query_route(origin, destination, seed=42, verbose=True):
    """
    Main system function:
    1. Predicts density for all road segments
    2. Builds density-aware weighted graph
    3. Runs Dijkstra for static and aware routes
    4. Returns comparison results
    """
    if origin not in nodes:
        return f"❌ Invalid origin '{origin}'. Choose from: {', '.join(sorted(nodes.keys()))}"
    if destination not in nodes:
        return f"❌ Invalid destination '{destination}'. Choose from: {', '.join(sorted(nodes.keys()))}"
    if origin == destination:
        return "❌ Origin and destination must be different."

    density_map = predict_network_density(seed=seed)
    G_aware     = build_aware_graph(density_map)

    # Static route (ignores traffic)
    static_path = nx.dijkstra_path(G_base,  origin, destination, weight='base_weight')
    static_cost = nx.dijkstra_path_length(G_base, origin, destination, weight='base_weight')
    s_dens, s_dist = get_route_stats(static_path, density_map)

    # Density-aware route
    aware_path  = nx.dijkstra_path(G_aware, origin, destination, weight='weight')
    a_dens, a_dist = get_route_stats(aware_path, density_map)

    overhead  = (a_dist - s_dist) / s_dist * 100 if s_dist > 0 else 0
    avoided   = s_dens['High'] - a_dens['High']

    result = {
        'origin'          : origin,
        'destination'     : destination,
        'density_map'     : density_map,
        'static_path'     : static_path,
        'static_dist'     : s_dist,
        'static_density'  : s_dens,
        'aware_path'      : aware_path,
        'aware_dist'      : a_dist,
        'aware_density'   : a_dens,
        'high_avoided'    : avoided,
        'dist_overhead_pct': round(overhead,1),
    }

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  🗺  ROUTE QUERY: {origin}  →  {destination}")
        print(f"  {'─'*50}")
        print(f"\n  📍 STATIC ROUTE (no traffic awareness)")
        print(f"     Path     : {' → '.join(static_path)}")
        print(f"     Distance : {s_dist:.2f} km")
        print(f"     Density  : Low={s_dens['Low']} | Med={s_dens['Medium']} | High={s_dens['High']}")

        print(f"\n  🚦 DENSITY-AWARE ROUTE (ML-optimised)")
        print(f"     Path     : {' → '.join(aware_path)}")
        print(f"     Distance : {a_dist:.2f} km")
        print(f"     Density  : Low={a_dens['Low']} | Med={a_dens['Medium']} | High={a_dens['High']}")
        print(f"     Overhead : {overhead:+.1f}% distance vs static route")

        print(f"\n  📊 RESULT")
        if avoided > 0:
            print(f"     ✅ {avoided} high-density segment(s) successfully avoided!")
            print(f"     ✅ Density-aware route is RECOMMENDED.")
        elif avoided == 0 and aware_path == static_path:
            print(f"     ℹ️  Both routes are identical — no congestion on this path.")
        else:
            print(f"     ℹ️  No additional high-density segments avoided on this query.")
        print(f"  {'─'*50}\n")

    return result

# =============================================================================
# STAGE 5: RUN SYSTEM DEMONSTRATION
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 5: SYSTEM DEMONSTRATION — LIVE QUERIES\n"+"─"*60)

demo_queries = [
    ('A', 'L', 42,  'Morning peak — North to East'),
    ('K', 'M', 77,  'Afternoon — South-West to North-East'),
    ('A', 'N', 99,  'Evening peak — Cross-city'),
    ('B', 'O', 55,  'Off-peak — North to South'),
    ('K', 'D', 123, 'Weekday — South-West to North-East corner'),
]

all_demo_results = []
for origin, dest, seed, desc in demo_queries:
    print(f"\n  ► {desc}")
    result = query_route(origin, dest, seed=seed, verbose=True)
    all_demo_results.append({**result,
        'description'   : desc,
        'static_path'   : ' → '.join(result['static_path']),
        'aware_path'    : ' → '.join(result['aware_path']),
        'density_map'   : {f"{k[0]}-{k[1]}":v for k,v in result['density_map'].items()},
    })

# =============================================================================
# STAGE 6: COMPUTE OVERALL SYSTEM PERFORMANCE
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 6: OVERALL SYSTEM PERFORMANCE\n"+"─"*60)

total_queries   = len(all_demo_results)
improved        = sum(1 for r in all_demo_results if r['high_avoided'] > 0)
same            = sum(1 for r in all_demo_results if r['high_avoided'] == 0)
total_avoided   = sum(r['high_avoided'] for r in all_demo_results)
avg_overhead    = np.mean([r['dist_overhead_pct'] for r in all_demo_results])
avg_s_high      = np.mean([r['static_density']['High'] for r in all_demo_results])
avg_a_high      = np.mean([r['aware_density']['High']  for r in all_demo_results])
congestion_reduction = ((avg_s_high - avg_a_high) / avg_s_high * 100) if avg_s_high > 0 else 0

print(f"\n  Total queries run          : {total_queries}")
print(f"  Routes improved            : {improved} / {total_queries}")
print(f"  Routes unchanged           : {same} / {total_queries}")
print(f"  Total high-density avoided : {total_avoided} segments")
print(f"  Avg congestion reduction   : {congestion_reduction:.1f}%")
print(f"  Avg distance overhead      : {avg_overhead:.1f}%")

# =============================================================================
# STAGE 7: SAVE ALL RESULTS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 7: SAVING INTEGRATED RESULTS\n"+"─"*60)

os.makedirs('outputs', exist_ok=True)

# Save demo results
save_results = []
for r in all_demo_results:
    save_results.append({
        'description'      : r['description'],
        'origin'           : r['origin'],
        'destination'      : r['destination'],
        'static_path'      : r['static_path'],
        'static_dist_km'   : r['static_dist'],
        'static_high_segs' : r['static_density']['High'],
        'aware_path'       : r['aware_path'],
        'aware_dist_km'    : r['aware_dist'],
        'aware_high_segs'  : r['aware_density']['High'],
        'high_segs_avoided': r['high_avoided'],
        'dist_overhead_pct': r['dist_overhead_pct'],
    })

integration_summary = {
    'system_name'           : 'Vehicle Density Detection and Route Optimisation',
    'author'                : 'Sanusi Moshood Olanrewaju',
    'institution'           : 'Federal University Oye-Ekiti (FUOYE)',
    'ml_model_used'         : 'Random Forest',
    'dt_accuracy_pct'       : model_results['dt_accuracy'],
    'rf_accuracy_pct'       : model_results['rf_accuracy'],
    'road_nodes'            : G_base.number_of_nodes(),
    'road_edges'            : G_base.number_of_edges(),
    'routing_algorithm'     : "Dijkstra's Algorithm",
    'total_queries'         : total_queries,
    'routes_improved'       : improved,
    'total_high_avoided'    : total_avoided,
    'avg_congestion_reduction_pct': round(congestion_reduction,2),
    'avg_distance_overhead_pct'   : round(avg_overhead,2),
    'demo_queries'          : save_results,
}

with open('outputs/integration_summary.json','w') as f:
    json.dump(integration_summary, f, indent=2)
print("✅ Saved: outputs/integration_summary.json")

# =============================================================================
# STAGE 8: FINAL VISUALISATIONS
# =============================================================================
print("\n"+"─"*60+"\nSTAGE 8: GENERATING FINAL SYSTEM CHARTS\n"+"─"*60)

# ── Chart 1: System Pipeline Diagram ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0,14); ax.set_ylim(0,4); ax.axis('off')
fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#F8F9FA')

pipeline_steps = [
    (1.2, 2, '#2E4057', 'INPUT\nMODULE',       'Road Segment\nTime / Conditions'),
    (4.0, 2, '#1A6B8A', 'PREPROCESSING\nMODULE','Clean · Encode\nScale · Engineer'),
    (6.8, 2, '#2E7D32', 'PREDICTION\nMODULE',   'Random Forest\nLow/Med/High'),
    (9.6, 2, '#B05E00', 'ROUTING\nMODULE',      "Graph + Dijkstra's\nAware Weights"),
    (12.4,2, '#6A1B9A', 'OUTPUT\nMODULE',       'Optimal Route\n+ Density Map'),
]

for i,(x,y,col,title,sub) in enumerate(pipeline_steps):
    rect = plt.Rectangle((x-1.0, y-0.75), 2.0, 1.5,
                          facecolor=col, edgecolor='white', lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y+0.25, title, ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='white', zorder=3)
    ax.text(x, y-0.32, sub, ha='center', va='center', fontsize=7,
            color='#E0E0E0', zorder=3)
    if i < len(pipeline_steps)-1:
        ax.annotate('', xy=(pipeline_steps[i+1][0]-1.05, y),
                    xytext=(x+1.05, y),
                    arrowprops=dict(arrowstyle='-|>', color='#2E4057',
                                    lw=2.5, mutation_scale=18), zorder=4)

ax.set_title('Integrated System Pipeline — Vehicle Density Detection & Route Optimisation',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('outputs/Step4_System_Pipeline.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step4_System_Pipeline.png")

# ── Chart 2: Demo Query Results Summary ──────────────────────────────────────
descs     = [r['description'].split('—')[-1].strip() for r in all_demo_results]
s_highs   = [r['static_density']['High']  for r in all_demo_results]
a_highs   = [r['aware_density']['High']   for r in all_demo_results]
overheads = [r['dist_overhead_pct']       for r in all_demo_results]

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('System Integration — Live Query Performance',
             fontsize=14, fontweight='bold')

y = np.arange(len(descs)); h = 0.35
ax1.barh(y+h/2, s_highs, h, label='Static Route',
         color='#E74C3C', edgecolor='black', lw=0.8)
ax1.barh(y-h/2, a_highs, h, label='Density-Aware',
         color='#2ECC71', edgecolor='black', lw=0.8)
ax1.set_xlabel('High-Density Segments on Route', fontsize=11)
ax1.set_title('High-Density Segments Encountered\nper Query', fontweight='bold')
ax1.set_yticks(y); ax1.set_yticklabels(descs, fontsize=9)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1.legend(); ax1.xaxis.grid(True,linestyle='--',alpha=0.6)
ax1.set_axisbelow(True)

colors2 = ['#2ECC71' if o <= 5 else '#F39C12' if o <= 15 else '#E74C3C'
           for o in overheads]
bars = ax2.barh(y, overheads, color=colors2, edgecolor='black', lw=0.8)
ax2.axvline(0, color='black', lw=1)
ax2.set_xlabel('Distance Overhead vs Static Route (%)', fontsize=11)
ax2.set_title('Distance Overhead of\nDensity-Aware Route', fontweight='bold')
ax2.set_yticks(y); ax2.set_yticklabels(descs, fontsize=9)
ax2.xaxis.grid(True,linestyle='--',alpha=0.6); ax2.set_axisbelow(True)
for bar,val in zip(bars,overheads):
    ax2.text(val+0.2, bar.get_y()+bar.get_height()/2,
             f'{val:+.1f}%', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/Step4_Query_Performance.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step4_Query_Performance.png")

# ── Chart 3: Final System Summary Dashboard ───────────────────────────────────
fig = plt.figure(figsize=(14,8))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('System Summary Dashboard\nVehicle Density Detection and Route Optimisation',
             fontsize=15, fontweight='bold', y=0.98)

# KPI boxes
kpi_data = [
    ('ML Accuracy\n(Random Forest)', f"{model_results['rf_accuracy']}%", '#2E7D32'),
    ('ML Accuracy\n(Decision Tree)', f"{model_results['dt_accuracy']}%", '#1A6B8A'),
    ('Road Segments\nModelled',       f"{G_base.number_of_edges()}",      '#B05E00'),
    ('Routes\nImproved',              f"{improved}/{total_queries}",       '#6A1B9A'),
    ('High-Density\nAvoided',         f"{total_avoided} segs",            '#C0392B'),
    ('Avg Overhead',                  f"{avg_overhead:.1f}%",             '#7F8C8D'),
]

for i,(title,val,col) in enumerate(kpi_data):
    ax = fig.add_subplot(2,6,i+1)
    ax.set_facecolor(col)
    ax.text(0.5,0.62,val,  ha='center',va='center',fontsize=18,
            fontweight='bold',color='white',transform=ax.transAxes)
    ax.text(0.5,0.22,title,ha='center',va='center',fontsize=8.5,
            color='#E0E0E0',transform=ax.transAxes,multialignment='center')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_edgecolor('white'); spine.set_linewidth(2)

# Results table
ax_tbl = fig.add_subplot(2,1,2)
ax_tbl.axis('off')

col_labels = ['Query', 'Static Path', 'Aware Path', 'High Avoided', 'Overhead']
table_data = []
for r in save_results:
    table_data.append([
        r['description'].split('—')[-1].strip()[:22],
        r['static_path'],
        r['aware_path'],
        f"{r['high_segs_avoided']:+d}",
        f"{r['dist_overhead_pct']:+.1f}%",
    ])

tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1, 1.6)
for (row,col),cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2E4057'); cell.set_text_props(color='white',fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    cell.set_edgecolor('#CCCCCC')

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('outputs/Step4_System_Dashboard.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ Saved: outputs/Step4_System_Dashboard.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"""
{"="*60}
  STEP 4 COMPLETE — FULL SYSTEM INTEGRATION SUMMARY
{"="*60}

  ┌─────────────────────────────────────────────────┐
  │  MACHINE LEARNING PERFORMANCE                   │
  │  Decision Tree Accuracy : {model_results['dt_accuracy']:>6}%               │
  │  Random Forest Accuracy : {model_results['rf_accuracy']:>6}%               │
  ├─────────────────────────────────────────────────┤
  │  ROAD NETWORK                                   │
  │  Intersections (nodes)  : {G_base.number_of_nodes():>6}                │
  │  Road segments (edges)  : {G_base.number_of_edges():>6}                │
  │  Routing algorithm      : Dijkstra's Algorithm  │
  ├─────────────────────────────────────────────────┤
  │  SYSTEM PERFORMANCE (5 live queries)            │
  │  Routes improved        : {improved:>2} / {total_queries}                  │
  │  High-density avoided   : {total_avoided:>6} segments          │
  │  Avg congestion reduction: {congestion_reduction:>5.1f}%               │
  │  Avg distance overhead  : {avg_overhead:>5.1f}%               │
  └─────────────────────────────────────────────────┘

  Outputs saved:
  ✅ outputs/integration_summary.json
  ✅ outputs/Step4_System_Pipeline.png
  ✅ outputs/Step4_Query_Performance.png
  ✅ outputs/Step4_System_Dashboard.png

  ════════════════════════════════════════════════════
  🎉 ALL 4 STEPS COMPLETE — SYSTEM FULLY INTEGRATED!
  ════════════════════════════════════════════════════

  Your project outputs folder contains:
  📊 Step1_Exploratory_Analysis.png
  📊 Step2_Confusion_Matrices.png
  📊 Step2_Performance_Comparison.png
  📊 Step2_Feature_Importance.png
  📊 Step2_CrossValidation.png
  📊 Step3_Road_Network_Density.png
  📊 Step3_Route_Comparison.png
  📊 Step3_Scenarios_Summary.png
  📊 Step4_System_Pipeline.png
  📊 Step4_Query_Performance.png
  📊 Step4_System_Dashboard.png
  📋 outputs/model_results.json
  📋 outputs/routing_results.json
  📋 outputs/integration_summary.json

  ✅ Ready for Chapter 4: Results and Discussion
""")
