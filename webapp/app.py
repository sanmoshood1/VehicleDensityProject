# =============================================================================
# VEHICLE DENSITY DETECTION AND ROUTE OPTIMISATION
# Step 5: Flask Web Application — Full Lagos Road Network
# Author: Sanusi Moshood Olanrewaju | FUOYE
# =============================================================================

from flask import Flask, render_template, request, jsonify
import pickle, json, os, numpy as np, networkx as nx
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# =============================================================================
# LAGOS ROAD NETWORK — 40 Real Roads with GPS Coordinates
# =============================================================================
ROADS = {
    'A':  {'name': 'Ikorodu Road (Ojota)',             'lat': 6.5833, 'lng': 3.3833},
    'B':  {'name': 'Maryland Junction',                 'lat': 6.5650, 'lng': 3.3600},
    'C':  {'name': 'Jibowu / Yaba',                    'lat': 6.5150, 'lng': 3.3700},
    'D':  {'name': 'Herbert Macaulay Way',              'lat': 6.5100, 'lng': 3.3600},
    'E':  {'name': 'Carter Bridge (Mainland)',          'lat': 6.4550, 'lng': 3.3900},
    'F':  {'name': 'Lagos Island CMS',                  'lat': 6.4530, 'lng': 3.3950},
    'G':  {'name': 'Ahmadu Bello Way (VI)',             'lat': 6.4280, 'lng': 3.4200},
    'H':  {'name': 'Adeola Odeku Street (VI)',          'lat': 6.4320, 'lng': 3.4150},
    'I':  {'name': 'Ozumba Mbadiwe (VI)',               'lat': 6.4350, 'lng': 3.4350},
    'J':  {'name': 'Lekki Phase 1 Gate',                'lat': 6.4450, 'lng': 3.4750},
    'K':  {'name': 'Lekki-Epe Expressway (Ajah)',       'lat': 6.4650, 'lng': 3.5750},
    'L':  {'name': 'Admiralty Way (Lekki)',             'lat': 6.4480, 'lng': 3.4900},
    'M':  {'name': 'Ikoyi Link Bridge',                 'lat': 6.4480, 'lng': 3.4050},
    'N':  {'name': 'Funsho Williams Avenue',            'lat': 6.4700, 'lng': 3.3600},
    'O':  {'name': 'Apapa-Oshodi Expressway',           'lat': 6.4550, 'lng': 3.3550},
    'P':  {'name': 'Oshodi Interchange',                'lat': 6.5500, 'lng': 3.3350},
    'Q':  {'name': 'Agege Motor Road',                  'lat': 6.6000, 'lng': 3.3150},
    'R':  {'name': 'Ikeja (Obafemi Awolowo Way)',       'lat': 6.6050, 'lng': 3.3490},
    'S':  {'name': 'Mobolaji Bank Anthony Way',         'lat': 6.5750, 'lng': 3.3580},
    'T':  {'name': 'Lateef Jakande Road (Ikeja)',       'lat': 6.5950, 'lng': 3.3400},
    'U':  {'name': 'Lagos-Abeokuta Expressway',         'lat': 6.6300, 'lng': 3.2700},
    'V':  {'name': 'Ipaja Road',                        'lat': 6.6100, 'lng': 3.2500},
    'W':  {'name': 'Egbeda-Idimu Road',                 'lat': 6.5900, 'lng': 3.2750},
    'X':  {'name': 'Akowonjo Road',                     'lat': 6.5700, 'lng': 3.2900},
    'Y':  {'name': 'LASU-Isheri Road',                  'lat': 6.5500, 'lng': 3.2600},
    'Z':  {'name': 'Lagos-Badagry Expressway',          'lat': 6.4700, 'lng': 3.2200},
    'AA': {'name': 'Mile 2 Corridor',                   'lat': 6.4850, 'lng': 3.3050},
    'AB': {'name': 'LASU-Iba Road',                     'lat': 6.4600, 'lng': 3.1900},
    'AC': {'name': 'Ijanikin-Badagry Road',             'lat': 6.4400, 'lng': 3.1200},
    'AD': {'name': 'Third Mainland Bridge (North)',     'lat': 6.5200, 'lng': 3.4000},
    'AE': {'name': 'Third Mainland Bridge (South)',     'lat': 6.4700, 'lng': 3.4050},
    'AF': {'name': 'Ikorodu-Sagamu Road',               'lat': 6.6500, 'lng': 3.5000},
    'AG': {'name': 'Agric Road (Ikorodu)',              'lat': 6.6200, 'lng': 3.5100},
    'AH': {'name': 'College Road Ogba',                 'lat': 6.6150, 'lng': 3.3300},
    'AI': {'name': 'Western Avenue / Eko Bridge',       'lat': 6.4600, 'lng': 3.3700},
    'AJ': {'name': 'Adetokunbo Ademola (VI)',           'lat': 6.4300, 'lng': 3.4100},
    'AK': {'name': 'Epe Road',                          'lat': 6.5800, 'lng': 3.9800},
    'AL': {'name': 'Banana Island Road',                'lat': 6.4600, 'lng': 3.4200},
    'AM': {'name': 'Ijede Road (Ikorodu)',              'lat': 6.6400, 'lng': 3.5400},
    'AN': {'name': 'Lekki-Epe Expressway (Sangotedo)', 'lat': 6.4700, 'lng': 3.6400},
}

EDGES = [
    ('A','B',3.5),  ('B','C',4.0),  ('B','S',3.8),  ('C','D',1.5),
    ('C','AD',4.5), ('D','E',5.0),  ('D','N',3.0),  ('E','F',1.0),
    ('AD','AE',11.5),('AE','F',2.0),('AE','AI',2.5),('F','M',3.0),
    ('F','AI',1.5), ('F','AJ',3.5), ('M','G',2.5),  ('M','AL',2.0),
    ('G','H',1.0),  ('G','AJ',1.5), ('H','I',2.0),  ('I','J',4.0),
    ('J','L',2.5),  ('J','K',12.0), ('K','AN',8.0), ('AN','AK',25.0),
    ('N','O',4.5),  ('O','AA',5.0), ('O','AI',3.0), ('AA','Z',4.0),
    ('AA','AB',6.5),('Z','AB',5.5), ('AB','AC',15.0),
    ('P','O',4.0),  ('P','S',3.5),  ('P','Q',4.5),  ('P','R',5.0),
    ('Q','R',3.0),  ('Q','AH',2.5), ('Q','U',7.0),  ('R','S',2.5),
    ('R','T',2.0),  ('R','AH',3.5), ('S','A',5.0),  ('T','W',5.5),
    ('T','U',5.0),  ('U','V',4.0),  ('U','W',6.0),  ('V','W',3.5),
    ('V','Y',5.0),  ('W','X',3.0),  ('X','AA',7.5), ('Y','AA',8.0),
    ('A','AF',18.0),('AF','AG',3.0),('AG','AM',4.0),('AF','AM',5.0),
]

SPEED_MAP      = {'Low': 55, 'Medium': 28, 'High': 10}
PENALTY_MAP    = {'Low': 1.0, 'Medium': 1.6, 'High': 3.0}
DENSITY_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
DENSITY_COLOURS = {
    'Low':    {'bg':'#d4edda','text':'#155724','badge':'#28a745','map':'#10b981'},
    'Medium': {'bg':'#fff3cd','text':'#856404','badge':'#ffc107','map':'#f59e0b'},
    'High':   {'bg':'#f8d7da','text':'#721c24','badge':'#dc3545','map':'#ef4444'},
}

NIGERIAN_HOLIDAYS = [
    '01-01','02-01','05-01','06-12','10-01','10-14','12-25','12-26'
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
rf_model = feature_cols = df_records = None

def load_assets():
    global rf_model, feature_cols, df_records
    import pandas as pd
    try:
        with open(os.path.join(MODEL_DIR,'random_forest_model.pkl'),'rb') as f: rf_model=pickle.load(f)
        with open(os.path.join(MODEL_DIR,'feature_cols.pkl'),'rb') as f: feature_cols=pickle.load(f)
        df = pd.read_csv(os.path.join(DATA_DIR,'clean_traffic_data.csv'))
        df_records = {
            'Low'   : df[df['density_level']=='Low'][feature_cols].values,
            'Medium': df[df['density_level']=='Medium'][feature_cols].values,
            'High'  : df[df['density_level']=='High'][feature_cols].values,
        }
        print(f"✅ Model loaded | Features: {len(feature_cols)} | Records: {len(df):,}")
        return True
    except Exception as e:
        print(f"❌ Error loading: {e}"); return False

def build_base_graph():
    G = nx.Graph()
    G.add_nodes_from(ROADS.keys())
    for u,v,w in EDGES:
        if u in ROADS and v in ROADS:
            G.add_edge(u, v, base_weight=w, weight=w)
    return G

def predict_density(hour, day_of_week=1, is_holiday=False, is_raining=False):
    np.random.seed(int(hour)*7 + int(day_of_week)*3 + 13)
    G = build_base_graph()
    edges_list = list(G.edges())
    n = len(edges_list)
    is_weekend      = day_of_week >= 5
    is_morning_peak = 6 <= hour <= 10 and not is_weekend
    is_evening_peak = 16 <= hour <= 20 and not is_weekend
    is_night        = hour >= 22 or hour <= 4
    is_peak         = is_morning_peak or is_evening_peak

    if is_holiday:
        dist = ['Low']*20 + ['Medium']*10 + ['High']*5
    elif is_peak and is_raining:
        dist = ['Low']*3  + ['Medium']*7  + ['High']*25
    elif is_peak:
        dist = ['Low']*5  + ['Medium']*10 + ['High']*20
    elif is_raining and not is_night:
        dist = ['Low']*8  + ['Medium']*15 + ['High']*12
    elif is_weekend:
        dist = ['Low']*18 + ['Medium']*12 + ['High']*5
    elif is_night:
        dist = ['Low']*28 + ['Medium']*6  + ['High']*1
    else:
        dist = ['Low']*10 + ['Medium']*15 + ['High']*10

    np.random.shuffle(dist)
    assignments = (dist * ((n // len(dist)) + 1))[:n]
    density_map = {}
    for edge, assigned in zip(edges_list, assignments):
        if df_records and assigned in df_records:
            recs = df_records[assigned]
            rec  = recs[np.random.randint(len(recs))].reshape(1,-1)
            pred = rf_model.predict(rec)[0]
            label = DENSITY_LABELS[int(pred)]
        else:
            label = assigned
        density_map[edge] = label
    return density_map

def build_aware_graph(density_map):
    G = build_base_graph()
    for u,v in G.edges():
        edge = (u,v) if (u,v) in density_map else (v,u)
        base = G[u][v]['base_weight']
        pen  = PENALTY_MAP.get(density_map.get(edge,'Low'), 1.0)
        G[u][v]['weight'] = round(base*pen, 3)
    return G

def get_edge_density(path, density_map):
    counts = {'Low':0,'Medium':0,'High':0}
    for i in range(len(path)-1):
        u,v  = path[i],path[i+1]
        edge = (u,v) if (u,v) in density_map else (v,u)
        counts[density_map.get(edge,'Low')] += 1
    return counts

def calc_travel_time(path, density_map):
    G_base = build_base_graph()
    total  = 0.0
    for i in range(len(path)-1):
        u,v   = path[i],path[i+1]
        dist  = G_base[u][v]['base_weight']
        edge  = (u,v) if (u,v) in density_map else (v,u)
        speed = SPEED_MAP[density_map.get(edge,'Low')]
        total += (dist/speed)*60
    return round(total)

def get_overall_congestion(density_counts, total_edges):
    if total_edges == 0: return 'Low'
    hp = density_counts['High']/total_edges
    mp = density_counts['Medium']/total_edges
    if hp >= 0.4: return 'High'
    elif hp >= 0.2 or mp >= 0.5: return 'Medium'
    return 'Low'

def find_alternative_route(G_aware, origin, dest, best_path):
    G_alt = G_aware.copy()
    for i in range(len(best_path)-1):
        u,v = best_path[i],best_path[i+1]
        if G_alt.has_edge(u,v): G_alt.remove_edge(u,v)
    try: return nx.dijkstra_path(G_alt, origin, dest, weight='weight')
    except nx.NetworkXNoPath: return None

def build_map_segments(path, density_map):
    segments = []
    for i in range(len(path)-1):
        u,v    = path[i],path[i+1]
        edge   = (u,v) if (u,v) in density_map else (v,u)
        label  = density_map.get(edge,'Low')
        segments.append({
            'from'   : {'lat':ROADS[u]['lat'],'lng':ROADS[u]['lng'],'name':ROADS[u]['name']},
            'to'     : {'lat':ROADS[v]['lat'],'lng':ROADS[v]['lng'],'name':ROADS[v]['name']},
            'density': label,
            'colour' : DENSITY_COLOURS[label]['map'],
        })
    return segments

def route_query(origin, dest, hour, day_of_week=1, is_holiday=False, is_raining=False):
    G_base      = build_base_graph()
    density_map = predict_density(hour, day_of_week, is_holiday, is_raining)
    G_aware     = build_aware_graph(density_map)

    best_path = nx.dijkstra_path(G_aware, origin, dest, weight='weight')
    best_dens = get_edge_density(best_path, density_map)
    best_dist = round(sum(G_base[best_path[i]][best_path[i+1]]['base_weight'] for i in range(len(best_path)-1)),1)
    best_time = calc_travel_time(best_path, density_map)
    best_cong = get_overall_congestion(best_dens, len(best_path)-1)
    best_segs = build_map_segments(best_path, density_map)

    alt_path   = find_alternative_route(G_aware, origin, dest, best_path)
    alt_result = None
    if alt_path:
        alt_dens = get_edge_density(alt_path, density_map)
        alt_dist = round(sum(G_base[alt_path[i]][alt_path[i+1]]['base_weight'] for i in range(len(alt_path)-1)),1)
        alt_time = calc_travel_time(alt_path, density_map)
        alt_cong = get_overall_congestion(alt_dens, len(alt_path)-1)
        alt_segs = build_map_segments(alt_path, density_map)
        alt_result = {
            'path':alt_path, 'path_names':[ROADS[n]['name'] for n in alt_path],
            'path_str':' → '.join(ROADS[n]['name'] for n in alt_path),
            'distance':alt_dist,'time_mins':alt_time,'congestion':alt_cong,
            'colours':DENSITY_COLOURS[alt_cong],'density':alt_dens,'map_segments':alt_segs,
        }

    density_summary = {'Low':0,'Medium':0,'High':0}
    for v in density_map.values(): density_summary[v] += 1

    return {
        'origin':origin, 'origin_name':ROADS[origin]['name'],
        'origin_coords':{'lat':ROADS[origin]['lat'],'lng':ROADS[origin]['lng']},
        'dest':dest, 'dest_name':ROADS[dest]['name'],
        'dest_coords':{'lat':ROADS[dest]['lat'],'lng':ROADS[dest]['lng']},
        'hour':hour, 'is_peak':(6<=hour<=10) or (16<=hour<=20),
        'is_raining':is_raining, 'is_holiday':is_holiday,
        'best':{
            'path':best_path,'path_names':[ROADS[n]['name'] for n in best_path],
            'path_str':' → '.join(ROADS[n]['name'] for n in best_path),
            'distance':best_dist,'time_mins':best_time,'congestion':best_cong,
            'colours':DENSITY_COLOURS[best_cong],'density':best_dens,'map_segments':best_segs,
        },
        'alternative':alt_result,
        'network_density':density_summary,
        'total_roads':len(EDGES),
        'all_roads':{k:{'lat':v['lat'],'lng':v['lng'],'name':v['name']} for k,v in ROADS.items()},
    }

# ── Flask Routes ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('home.html', roads={k:v['name'] for k,v in ROADS.items()}, active='home')

@app.route('/query', methods=['GET'])
def query_page():
    all_coords = {k:{'lat':v['lat'],'lng':v['lng'],'name':v['name']} for k,v in ROADS.items()}
    all_edges  = [{'from':u,'to':v} for u,v,_ in EDGES if u in ROADS and v in ROADS]
    return render_template('query.html',
        roads={k:v['name'] for k,v in ROADS.items()},
        all_coords=json.dumps(all_coords),
        all_edges=json.dumps(all_edges),
        active='query')

@app.route('/query', methods=['POST'])
def query_api():
    try:
        data        = request.get_json()
        origin      = data.get('origin','').strip().upper()
        dest        = data.get('destination','').strip().upper()
        hour        = int(data.get('hour',8))
        day_of_week = int(data.get('day_of_week',1))
        is_holiday  = bool(data.get('is_holiday',False))
        is_raining  = bool(data.get('is_raining',False))

        if not origin or origin not in ROADS:
            return jsonify({'error':'Invalid source road.'}),400
        if not dest or dest not in ROADS:
            return jsonify({'error':'Invalid destination road.'}),400
        if origin == dest:
            return jsonify({'error':'Source and destination must be different.'}),400

        result = route_query(origin, dest, hour, day_of_week, is_holiday, is_raining)
        return jsonify({'success':True,'result':result})
    except nx.NetworkXNoPath:
        return jsonify({'error':'No route found. These roads may not be connected.'}),400
    except Exception as e:
        return jsonify({'error':f'System error: {str(e)}'}),500

@app.route('/analysis')
def analysis():
    return render_template('analysis.html', active='analysis')

@app.route('/performance')
def performance():
    return render_template('performance.html', active='performance')

@app.route('/about')
def about():
    return render_template('about.html', active='about',
        total_roads=len(ROADS), total_edges=len(EDGES))

@app.route('/health')
def health():
    return jsonify({'status':'running','model_loaded':rf_model is not None,
        'roads':len(ROADS),'edges':len(EDGES)})

if __name__ == '__main__':
    print("="*60)
    print("  VEHICLE DENSITY DETECTION & ROUTE OPTIMISATION")
    print(f"  Lagos Road Network — {len(ROADS)} Roads, {len(EDGES)} Connections")
    print("="*60)
    load_assets()
    print("\n  Open: http://127.0.0.1:5000\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
