import os
import argparse
import pickle

import matplotlib.pyplot as plt
import jax
import numpy as np
import numpyro
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

jax.config.update("jax_enable_x64", True)


def main(args):
    # Set initial RNG Key:
    initial_key = jax.random.PRNGKey(args.seed)

    # Load trained model:
    filename = args.model_path
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    with open(filepath, 'rb') as handle:
        mcmc, config, train_args = pickle.load(handle)

    samples = mcmc.get_samples()

    w_1 = samples['w1']
    w_2 = samples['w2']
    w_3 = samples['w3']

    avg_w_1 = np.mean(w_1, axis=0).T
    avg_w_2 = np.mean(w_2, axis=0).T
    avg_w_3 = np.mean(w_3, axis=0)


    G = nx.DiGraph()

    layer_1_nodes = avg_w_1.shape[-1]
    layer_2_nodes = avg_w_2.shape[-1]
    layer_3_nodes = avg_w_2.shape[-1]
    layer_4_nodes = avg_w_3.shape[-1]

    layer_1_node_range = (0, layer_1_nodes)
    layer_2_node_range = (layer_1_nodes, layer_1_nodes + layer_2_nodes)
    layer_3_node_range = (layer_1_nodes + layer_2_nodes, layer_1_nodes + layer_2_nodes + layer_3_nodes)
    layer_4_node_range = (layer_1_nodes + layer_2_nodes + layer_3_nodes, layer_1_nodes + layer_2_nodes + layer_3_nodes + layer_4_nodes)

    # Input Layer:
    row = 0
    col = 0
    for i in range(*layer_1_node_range):
        for j in range(*layer_2_node_range):
            G.add_edge(i, j, weight=avg_w_1[i, j])
            col += 1
        col = 0
        row += 1

    # Hidden Layer:
    row = 0
    col = 0
    for i in range(*layer_2_node_range):
        for j in range(*layer_3_node_range):
            G.add_edge(i, j, weight=avg_w_2[row, col])
            col += 1
        col = 0
        row += 1

    # Output Layer:
    row = 0
    col = 0
    for i in range(*layer_3_node_range):
        for j in range(*layer_4_node_range):
            G.add_edge(i, j, weight=avg_w_3[row, col])
            col += 1
        col = 0
        row += 1

    adjacency_matrix = nx.adjacency_matrix(G, weight='weight').todense()
    laplacian_matrix = nx.directed_laplacian_matrix(G, weight='weight')

    # Graph Eigenvalues:
    _, s_a, _ = np.linalg.svd(adjacency_matrix)
    _, s_l, _ = np.linalg.svd(laplacian_matrix)
    number_of_connections = np.where(s_l >= 1e-6)[0].shape[0]

    # Dedensify Graph:
    C, C_nodes = nx.dedensify(G, threshold=2)
    C_a = nx.adjacency_matrix(C, weight='weight').todense()
    C_l = nx.directed_laplacian_matrix(C, weight='weight')

    # Graph Singular Values:
    fig, ax = plt.subplots()
    ax.plot(s_a, linestyle='none', label='Adjacency', marker='*')
    ax.plot(s_l, linestyle='none', label='Laplacian', marker='.')
    ax.set_ylabel('Singular Values')
    ax.legend(loc='upper right')
    filename = 'singular_values.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)

    # Graph Adjacency Matrix:
    fig, ax = plt.subplots()
    ax.imshow(adjacency_matrix)
    filename = 'adjacency_matrix.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)

    # Graph Laplacian Matrix:
    fig, ax = plt.subplots()
    ax.imshow(laplacian_matrix)
    filename = 'laplacian_matrix.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)

    # Graph:
    fig, ax = plt.subplots()
    pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
    nx.draw(G, with_labels=True, pos=pos, ax=ax, font_weight='bold')
    filename = 'graph.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)

    # Dedensified Graph:
    fig, ax = plt.subplots()
    pos = graphviz_layout(C, prog='dot', args="-Grankdir=LR")
    nx.draw(C, with_labels=True, pos=pos, ax=ax, font_weight='bold')
    filename = 'dedensified_graph.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)

    # Dedensified Graph Adjacency Matrix:
    fig, ax = plt.subplots()
    ax.imshow(C_a)
    filename = 'dedensified_adjacency_matrix.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)

    # Dedensified Graph Laplacian Matrix:
    fig, ax = plt.subplots()
    ax.imshow(C_l)
    filename = 'dedensified_laplacian_matrix.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sensitivity analysis of model.")
    parser.add_argument("--seed", nargs="?", default=0, type=int)
    parser.add_argument("--model-path", nargs="?", default="mass_spring_damper.pickle", type=str, help='path to trained model.')
    parser.add_argument("--num-devices", nargs="?", default=1, type=int, help='number of devices')
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.enable_x64(use_x64=True)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_devices)

    main(args)
