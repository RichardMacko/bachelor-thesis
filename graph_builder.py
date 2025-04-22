import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx.algorithms.community as nx_comm
import community.community_louvain as community_louvain  # Louvainov algoritmus na detekciu komunít
import random
from collections import Counter, defaultdict
import copy
import numpy as np
import seaborn as sns
from pyvis.network import Network
import json


class GraphVisualizer:
    def __init__(self, following_dict, interaction_dict):
        self.following_dict = following_dict
        self.interaction_dict = interaction_dict
        self.main_user = "richard_macko"
        self.graph = None
        self.partition = None

    def build_graph(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(self.main_user)

        for user_a, followers in self.following_dict.items():
            if user_a not in self.graph:
                self.graph.add_node(user_a)
            if user_a != self.main_user:
                self.graph.add_edge(self.main_user, user_a, weight=0)

            for user_b in followers:
                weight = self.interaction_dict.get((user_a, user_b), 0) + self.interaction_dict.get((user_b, user_a), 0)
                if weight > 2:
                    self.graph.add_edge(user_a, user_b, weight=weight)

        weights = [d["weight"] for _, _, d in self.graph.edges(data=True) if d["weight"] > 0]
        if weights:
            avg_weight = sum(weights) / len(weights)
            print(f"Priemerná váha hrán: {avg_weight:.2f}")
        else:
            print("Žiadne hrany s váhou > 0")
        nodes_to_remove = [node for node in self.graph.nodes() if self.graph.degree(node) <= 2]
        self.graph.remove_nodes_from(nodes_to_remove)
        self.partition = community_louvain.best_partition(self.graph.to_undirected())

    def draw_communities_pyvis(self):
        community_graph = self.generate_community_graph()
        self.draw_community_graph_pyvis(community_graph, "komunity.html", self.graph)

    def draw_communities_plotly(self):
        community_graph = self.generate_community_graph()
        self.draw_community_graph_plotly(community_graph)

    def draw_full_graph_matplotlib(self):
        unique_communities = list(set(self.partition.values()))
        community_colors = {comm: (random.random(), random.random(), random.random()) for comm in unique_communities}
        node_colors = [community_colors[self.partition[node]] for node in self.graph.nodes()]

        edge_colors = []
        edge_widths = []

        for u, v, d in self.graph.edges(data=True):
            w = d.get('weight', 0)
            if w <= 2:
                edge_colors.append('gray')
                edge_widths.append(1)
            elif 3 <= w <= 5:
                edge_colors.append('orange')
                edge_widths.append(2)
            elif 6 <= w <= 9:
                edge_colors.append('green')
                edge_widths.append(3)
            else:
                edge_colors.append('red')
                edge_widths.append(4)

        pos = nx.spring_layout(self.graph, seed=42)
        centrality = nx.betweenness_centrality(self.graph)
        top_users = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 najvplyvnejších používateľov:", top_users)

        edge_betweenness = nx.edge_betweenness_centrality(self.graph)
        top_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top mosty v grafe (hrany):", top_edges)

        community_sizes = Counter(self.partition.values())
        print("Veľkosť komunít:", community_sizes)

        for comm_id in unique_communities:
            members = [n for n in self.partition if self.partition[n] == comm_id]
            subgraph = self.graph.subgraph(members)
            weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
            avg_weight = sum(weights) / len(weights) if weights else 0
            print(f"Komunita {comm_id}: {len(members)} uzlov, priemerná váha hrán: {avg_weight:.2f}")

        print("Počet uzlov:", self.graph.number_of_nodes())
        print("Počet hrán:", self.graph.number_of_edges())
        print("Počet komunít:", len(unique_communities))

        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
                width=edge_widths, arrows=True,
                node_size=[1000 * centrality[n] + 200 for n in self.graph.nodes()])

        plt.scatter([], [], color="gray", label="1-2 interakcie")
        plt.scatter([], [], color="orange", label="3-5 interakcií")
        plt.scatter([], [], color="green", label="6-9 interakcií")
        plt.scatter([], [], color="red", label="10+ interakcií")
        plt.legend(title="Sila interakcií")
        plt.show()
    def generate_community_graph(self):
        community_graph = nx.Graph()
        edge_weights = defaultdict(int)

        for u, v, data in self.graph.edges(data=True):
            comm_u = self.partition[u]
            comm_v = self.partition[v]
            if comm_u != comm_v:
                edge_weights[(comm_u, comm_v)] += data.get('weight', 1)

        community_members = defaultdict(list)
        for node, comm_id in self.partition.items():
            community_members[comm_id].append(node)


        for comm_id, members in community_members.items():
            community_graph.add_node(
                comm_id,
                members=members,
                member_count=len(members),  

            )


        for (comm_u, comm_v), weight in edge_weights.items():
            community_graph.add_edge(comm_u, comm_v, weight=weight)

        return community_graph

    def draw_community_graph_pyvis(self, community_graph, output_filename, original_graph=None):
        net = Network(height="800px", width="100%", bgcolor="#1e1e1e", font_color="white", notebook=False)
        net.barnes_hut()

        for node, data in community_graph.nodes(data=True):
            community_id = data.get("community_id", node)
            members = data.get("members", [])
            member_count = data.get("member_count")
            color = data.get("color", "#{:06x}".format(random.randint(0, 0xFFFFFF)))

            title = f"<b>Komunita {community_id}</b><br>Počet členov: {member_count}<br>Top členovia:<br>"
            title += "<br>".join(members[:5]) if members else "--"

            net.add_node(
                node,
                label=f"Komunita {community_id}",
                size=15 + member_count * 2,
                title=title,
                color=color,
                borderWidth=2,
            )

            if original_graph and members:
                self.draw_individual_community_graph(community_id, members, original_graph)

        for source, target, data in community_graph.edges(data=True):
            weight = data.get("weight", 1)
            net.add_edge(
                source,
                target,
                value=weight,
                title=f"Interakcie: {weight}",
                width=1 + weight * 0.5,
                color="#888888"
            )


        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 16,
              "face": "Arial"
            }
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": false
          },
          "interaction": {
            "navigationButtons": true,
            "multiselect": true
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          }
        }
        """)

        net.show(output_filename, notebook=False)

    def draw_individual_community_graph(self, community_id, members, original_graph):
        subgraph = original_graph.subgraph(members)
        net = Network(height="750px", width="100%", bgcolor="#f4f4f4", font_color="#000000", notebook=False)
        net.barnes_hut()

        for node in subgraph.nodes():
            net.add_node(node, label=node, title=node, size=15, color="#3498db")

        for u, v, data in subgraph.edges(data=True):
            weight = data.get("weight", 1)
            net.add_edge(u, v, value=weight, title=f"Interakcie: {weight}")

        filename = f"komunita_{community_id}.html"
        net.show(filename, notebook=False)
        print(f"✅ Vygenerovaný detailný graf pre komunitu {community_id}: {filename}")

    def get_pagerank(self):
        pagerank = nx.pagerank(self.graph)
        sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_pagerank


    def draw_community_graph_plotly(self, community_graph):
        pos = nx.spring_layout(community_graph, seed=42)

        edge_x = []
        edge_y = []
        edge_weights = []

        for u, v, data in community_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_weights.append(data['weight'])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        sizes = []
        labels = []

        for node in community_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            sizes.append(community_graph.nodes[node]['member_count'] * 10)
            labels.append(f'Komunita {node}<br>Veľkosť: {community_graph.nodes[node]["member_count"]}')

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[f'K{n}' for n in community_graph.nodes()],
            hovertext=labels,
            textposition="bottom center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=sizes,
                color=sizes,
                colorbar=dict(
                    thickness=15,
                    title='Veľkosť komunity',
                    xanchor='left',

                ),
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(
                                text='Vizualizácia komunít (superuzly)',
                                font=dict(size=20)
                            ),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Hrany reprezentujú silu interakcií medzi komunitami.",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                        )
        fig.show()

    def show_heatmap(self):
        unique_communities = list(set(self.partition.values()))
        matrix = np.zeros((len(unique_communities), len(unique_communities)))

        for u, v, weight in self.graph.edges(data=True):
            comm_u = self.partition[u]
            comm_v = self.partition[v]
            matrix[comm_u][comm_v] += weight["weight"]
            matrix[comm_v][comm_u] += weight["weight"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, cmap="coolwarm", xticklabels=unique_communities, yticklabels=unique_communities)
        plt.xlabel("Komunita")
        plt.ylabel("Komunita")
        plt.title("Heatmapa interakcií medzi komunitami")
        plt.show()

    def show_histogram_of_communities(self):
        community_sizes = Counter(self.partition.values())
        unique_communities = list(set(self.partition.values()))
        plt.figure(figsize=(8, 5))
        plt.bar(community_sizes.keys(), community_sizes.values(), color='skyblue')
        plt.xlabel("ID komunity")
        plt.ylabel("Počet členov")
        plt.title("Rozloženie veľkosti komunít")
        plt.xticks(range(len(unique_communities)))
        plt.show()

    def show_density(self):
        undirected_graph = self.graph.to_undirected()
        communities = {}
        for node, comm in self.partition.items():
            communities.setdefault(comm, set()).add(node)


        density_dict = {} 
        intensity_ratio_dict = {} 


        for comm, nodes in communities.items():

            n = len(nodes)

            max_possible_edges = n * (n - 1) / 2 if n > 1 else 1

            internal_edge_count = 0
            internal_weight_sum = 0


            for u in nodes:
                for v in nodes:
                    if u < v and undirected_graph.has_edge(u, v):
                        internal_edge_count += 1
                        internal_weight_sum += undirected_graph[u][v].get('weight', 1)  

            density = internal_edge_count / max_possible_edges if max_possible_edges != 0 else 0
            density_dict[comm] = density


            total_incident_weight = 0
            for u in nodes:
                for v in undirected_graph.neighbors(u):
                    total_incident_weight += undirected_graph[u][v].get('weight', 1)

            intensity_ratio = internal_weight_sum / total_incident_weight if total_incident_weight != 0 else 0
            intensity_ratio_dict[comm] = intensity_ratio


        print("\nMeranie intenzity bublín:")
        for comm in sorted(communities.keys()):
            print(f"Komunita {comm}: počet uzlov = {len(communities[comm])}, "
                f"hustota = {density_dict[comm]:.3f}, "
                f"intensity ratio = {intensity_ratio_dict[comm]:.3f}")


        plt.subplot(1, 2, 2)
        communities_sorted = sorted(communities.keys())
        density_values = [density_dict[comm] for comm in communities_sorted]
        intensity_values = [intensity_ratio_dict[comm] for comm in communities_sorted]

        bar_width = 0.35
        indices = range(len(communities_sorted))
        plt.bar([i - bar_width / 2 for i in indices], density_values, width=bar_width, label='Hustota')
        plt.bar([i + bar_width / 2 for i in indices], intensity_values, width=bar_width, label='Intensity Ratio')
        plt.xticks(indices, [f"Kom {comm}" for comm in communities_sorted])
        plt.xlabel("Komunita")
        plt.ylabel("Hodnota")
        plt.title("Meranie intenzity bublín")
        plt.legend()

        plt.tight_layout()
        plt.show()