import os.path
import networkx as nx
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def colorize_graph(
    G: nx.classes.graph.Graph = None, node_path_csv: str = "nodes.csv"
) -> nx.classes.graph.Graph:
    df_nodes = pd.read_csv(os.path.join("datasets", node_path_csv))
    # Add nodes to the graph with color labels based on their original labels
    color_map = {
        "L1": "blue",
        "L2": "red",
        "L3": "yellow",
        "L4": "green",
        "L5": "grey",
        "L6": "black",
        "L7": "orange",
    }
    initial_labels = {
        "L1": 1,  # sickest node
        "L2": 0,  # sickest node
        "L3": 0,  # sickest node
        "L4": 0,  # sickest node
        "L5": 0,  # sickest node
        "L6": 0,  # sickest node
        "L7": 0,  # sickest node
    }
    for index, row in df_nodes.iterrows():
        if row["labels"] in color_map:
            G.add_node(
                row["node_id"],
                color=color_map[row["labels"]],
                label=row["labels"],
                status=initial_labels[row["labels"]],
            )
        else:
            G.add_node(row["node_id"], color="white", label=row["labels"], status=0)
    return G


def check_is_community_detection_good(
    G: nx.classes.graph.Graph = None, method: object = None, type_method: str = ""
) -> None:
    """

    :param G:
    :param method:
    :param type_method:
    :return:
    """
    communities = method
    modularity = nx.algorithms.community.modularity(G, communities)
    coverage = sum([len(c) for c in communities]) / len(G)
    print(f"The modularity of the network is in {type_method}: {modularity}")
    print(f"The coverage of the network is: {coverage}")


def plot_draw_network_with_label(G: nx.classes.graph.Graph = None) -> None:
    """

    :param G:
    :return:
    """
    # Draw the graph with colored nodes
    pos = nx.spring_layout(G)
    centrality = nx.degree_centrality(G)
    # Set node size proportional to degree centrality
    node_sizes = [centrality[node] * 1000 for node in G]
    node_colors = [node[1]["color"] for node in G.nodes(data=True)]
    # Create a figure with a custom size
    plt.figure(figsize=(15, 15))  # Change the width and height as you like
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5,
        arrows=True,  # add arrows to indicate directed edges
        edge_color="black",  # set edge color to black
    )
    plt.show()


def plot_degree_dist(G: nx.classes.graph.Graph = None, title: str = "") -> None:
    """
    this method visualize plot degree dist of graph
    :param G:
    :return:
    """
    degrees = [G.degree(n) for n in G.nodes()]
    plt.figure(figsize=(8, 6))
    plt.title(title)
    sns.histplot(
        degrees,
        kde=False,
        stat="density",
        linewidth=0,
        bins=50,
        color=sns.xkcd_rgb["marine blue"],
    )
    plt.xlabel("Degree")
    plt.ylabel("Density (log scale)")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def calculate_the_probability_to_unknown_node_get_sick(G):
    # Calculate the probability of each Unknown node getting sick
    unknown_nodes = [n for n in G.nodes() if G.nodes[n]["status"] == 0]
    p_sick_given_L1_sick = {}
    for node in unknown_nodes:
        neighbors = list(G.neighbors(node))
        num_infected_neighbors = len(
            [n for n in neighbors if G.nodes[n]["status"] == 1]
        )
        p_sick_given_L1_sick[node] = num_infected_neighbors / len(neighbors)
    for k, v in p_sick_given_L1_sick.items():
        if v > 0.0 and G.nodes[k]["label"] == "Unknown":
            print(k, " | ", v)
    return p_sick_given_L1_sick


def calculate_modularity_and_community(
        G: nx.classes.graph.Graph = None, network_community_method: object = None
) -> None:
    start_time = time()
    communities = network_community_method
    fig, axs = plt.subplots(
        len(communities) // 3 + 1, 3,
        figsize=(20, len(communities) // 3 + 1)
    )  # Increase the figure size
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    community_differences = []
    community_differences_sd = []
    for i, comm in enumerate(communities):
        subgraph = G.subgraph(comm)
        # Create a figure with a custom size
        labels = [subgraph.nodes[item]["label"] for item in subgraph.nodes()]
        unique_label = set(labels)
        if len(unique_label) > 1:
            community_differences.append(
                max([labels.count(label) for label in unique_label]) / len(subgraph.nodes())
            )
            community_differences_sd.append(
                stdev([labels.count(label) for label in unique_label])
            )
        elif unique_label == 1:
            community_differences.append(
                1.0
            )
            community_differences_sd.append(
                0.0
            )
        # Calculate degree centrality for each node
        centrality = nx.degree_centrality(subgraph)
        # Set node size proportional to degree centrality
        node_sizes = [centrality[node] * 1000 for node in subgraph.nodes()]
        node_colors = [node[1]["color"] for node in subgraph.nodes(data=True)]
        pos = nx.circular_layout(subgraph)

        # Create a subplot with 3 columns
        if len(unique_label) > 1:
            axs[i // 3, i % 3].set_title(
                f"Community {i} avg : {max([labels.count(label) for label in unique_label]) / len(subgraph.nodes())}  sd : {stdev([labels.count(label) for label in unique_label])}")
        elif len(unique_label) == 1:
            axs[i // 3, i % 3].set_title(
                f"Community {i} avg : {max([labels.count(label) for label in unique_label]) / len(subgraph.nodes())} sd : {0.0}")
        axs[i // 3, i % 3].axis("off")

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=axs[i // 3, i % 3],  # Add this line to draw nodes on a specific axis
        )
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edge_color="gray",
            alpha=0.5,
            ax=axs[i // 3, i % 3],  # Add this line to draw edges on a specific axis
        )

    fig.tight_layout()
    print(
        "TIME FINISHED : ",
        time() - start_time,
        " \n Average difference of each community labels : ",
        sum(community_differences) / len(community_differences),
        "\n Average of each community Standard deviation : ",
        sum(community_differences_sd) / len(community_differences_sd),
    )



def visualize(
    params: list = None,
    spaceFromLeftNumber: int = 20,
    widthNumber: int = 110,
    title: str = " GITHUB STARGAZERS",
    G: nx.classes.graph.Graph = None,
) -> None:
    """
    this method is used to visualize metrics of graph
    :param params:
    :param spaceFromLeftNumber:
    :param widthNumber:
    :param title:
    :return:
    """
    spaceFromLeft = " " * spaceFromLeftNumber
    width = widthNumber
    print(spaceFromLeft + title)
    print("*" * width)
    for item in params:
        nexLine = "*" + spaceFromLeft + item
        numberOfSpaceForNexLine = width - len(nexLine) - 1
        print(nexLine, end="")
        print(" " * numberOfSpaceForNexLine + "*")
    print("*" * width)
    plot_degree_dist(G=G, title=title)


def fetch_hugest_subgraph(graph_):
    """

    :param graph_:
    :return:
    """
    Gcc = max(nx.connected_components(graph_), key=len)
    giantC = graph_.subgraph(Gcc)
    return giantC


def most_likely_unknown_to_L1(G: nx.classes.graph.Graph = None):
    G = colorize_graph(G)
    unknown_nodes = [n for n in G.nodes if G.nodes[n]["label"] == "Unknown"]
    disease_nodes = [n for n in G.nodes if G.nodes[n]["label"] == "L1"]
    probs = {}
    for n in unknown_nodes:
        coefficients = [
            coeff
            for m in disease_nodes
            for coeff in nx.jaccard_coefficient(G, [(n, m)])
        ]
        prob = max(coefficients)
        probs[n] = prob
    return probs


def clustering_unknowns(G: nx.classes.graph.Graph = None):
    G = colorize_graph(G)
    unknown_nodes = [n for n in G.nodes if G.nodes[n]["labels"] == "Unknown"]
    disease_nodes = [n for n in G.nodes if G.nodes[n]["labels"] == "L1"]
    probs = {}
    for n in unknown_nodes:
        coefficients = [
            coeff
            for m in disease_nodes
            for coeff in nx.jaccard_coefficient(G, [(n, m)])
        ]
        prob = max(coefficients)
        probs[n] = prob
    return probs


def average_shortest_path_length_for_all(graph_):
    """

    :param graph_:
    :return:
    """
    lengths = []
    for component in nx.connected_components(graph_):
        component_ = graph_.subgraph(component)
        length = nx.average_shortest_path_length(component_)
        lengths.append(length)
    myLength = len(lengths)
    if myLength == 0:
        return 0
    return sum(lengths) / myLength


def calculateMetrics(G: nx.classes.graph.Graph = None) -> list:
    """
    this method calculate metrics of graph
    :param G: graph
    :return: list of metrics to visualize
    """
    startTime = time()
    params = []
    diam = []
    for component in nx.strongly_connected_components(G):
        subgraph = G.subgraph(component)
        diam.append(nx.diameter(subgraph))
    diameter = "Diameter : " + str(max(diam))
    components = [G.subgraph(c).copy() for c in nx.strongly_connected_components(G)]
    size = len(G.nodes())
    weighted_avg_shortest_path = sum(
        nx.average_shortest_path_length(c) * len(c) / size for c in components
    )
    averageShortestPath = "Average shortest path : " + str(weighted_avg_shortest_path)
    numberOfNodes = "Number of nodes : " + str(G.number_of_nodes())
    numberOfEdges = "Number of edges : " + str(G.number_of_edges())
    density = "Density : " + str(nx.density(G))
    clusteringCoefficient1 = "Clustering coefficient 1 " + str(nx.transitivity(G))
    clusteringCoefficient2 = "Clustering coefficient 2 " + str(nx.average_clustering(G))

    assortativity = "Assortativity : " + str(
        nx.degree_pearson_correlation_coefficient(G)
    )
    bc = {
        k: v
        for k, v in sorted(
            nx.betweenness_centrality(G).items(), key=lambda item: item[1]
        )
    }
    betweennessCentrality = "Top 5 betweenness centrality : " + str(list(bc)[:5])
    cc = {
        k: v
        for k, v in sorted(nx.closeness_centrality(G).items(), key=lambda item: item[1])
    }
    closenessCentrality = "Top 5 closeness centrality : " + str(list(cc)[:5])
    dc = {
        k: v
        for k, v in sorted(nx.degree_centrality(G).items(), key=lambda item: item[1])
    }
    degreeCentrality = "Top 5 degree centrality path : " + str(list(dc)[:5])

    params.append(numberOfNodes)
    params.append(numberOfEdges)
    params.append(density)
    params.append(clusteringCoefficient1)
    params.append(clusteringCoefficient2)
    params.append(diameter)
    params.append(assortativity)
    params.append(betweennessCentrality)
    params.append(closenessCentrality)
    params.append(degreeCentrality)
    params.append(averageShortestPath)
    print("TIME FINISHED : ", time() - startTime)

    return params
