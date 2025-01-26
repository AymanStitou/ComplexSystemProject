# https://energydata.info/dataset/vietnam-electricity-transmission-network/resource/2b096a8f-17ab-432e-916c-82a8c1d4ce94

import json
import networkx as nx

def geojson_to_networkx(geojson_str):
    """
    Converts a GeoJSON FeatureCollection of MultiLineStrings into
    a NetworkX Graph. Each (lon, lat) coordinate is a node. Every
    pair of consecutive points forms an edge. Feature properties
    are stored as edge attributes.
    """
    data = json.loads(geojson_str)
    G = nx.Graph()

    for feature in data['features']:
        props = feature['properties']
        geometry = feature['geometry']

        if geometry['type'] == 'MultiLineString':
            # Each MultiLineString can have multiple sets of coordinates
            for line in geometry['coordinates']:
                # 'line' is a list of [lon, lat] pairs
                # Iterate over consecutive pairs
                for i in range(len(line) - 1):
                    # Current node is (lon1, lat1)
                    lon1, lat1 = line[i]
                    # Next node is (lon2, lat2)
                    lon2, lat2 = line[i + 1]

                    node_a = (lon1, lat1)
                    node_b = (lon2, lat2)

                    # Add the nodes to the graph
                    G.add_node(node_a)
                    G.add_node(node_b)

                    # Add an edge between these consecutive points
                    G.add_edge(node_a, node_b, **props)

    return G

with open('transmissionlinekv.json', 'r', encoding='utf-8') as f:
    geojson_str = f.read()


G = geojson_to_networkx(geojson_str)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Each edge has the original properties stored:
# for u, v, data in G.edges(data=True):
#     print(u, "--", v, "with attributes:", data)
