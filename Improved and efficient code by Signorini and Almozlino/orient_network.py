import math
import networkx
import numpy
import pandas
import scipy
import sklearn.linear_model

NETWORK_FILENAME = "H_sapiens.net"
DIRECTED_INTERACTIONS_FILENAME = "consensus.net"
SOURCES_FILENAME = "drug_targets.txt"
TERMINALS_FILENAME = "drug_expressions.txt"

PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

ORIENTATION_EPSILON = 0.01

def read_network():
    return pandas.read_table(NETWORK_FILENAME, header=None, usecols=[0,1,2], index_col=[0,1])

def read_directed_interactions():
    directed_interactions = pandas.read_table(DIRECTED_INTERACTIONS_FILENAME, header=None, skiprows=1, usecols=[0,1,2,3], index_col=[0,1])
    directed_interactions = directed_interactions[directed_interactions[2] != "biogrid"]

    return directed_interactions[[3]].rename(columns={3:2})

def read_priors(priors_filename):
    return pandas.read_table(priors_filename, header=None).groupby(0)[1].apply(set).to_dict()

def read_data():
    network = read_network()
    directed_interactions = read_directed_interactions()
    sources = read_priors(SOURCES_FILENAME)
    terminals = read_priors(TERMINALS_FILENAME)

    return network, directed_interactions, sources, terminals

def generate_similarity_matrix(graph):
    genes = sorted(graph.nodes)
    matrix = networkx.to_scipy_sparse_matrix(graph, genes, weight=2)

    norm_matrix = scipy.sparse.diags(1 / numpy.sqrt(matrix.sum(0).A1))
    matrix = norm_matrix * matrix * norm_matrix

    return PROPAGATE_ALPHA * matrix, genes

def propagate(seeds, matrix, gene_indexes, num_genes):
    F_t = numpy.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - PROPAGATE_ALPHA) * F_t

    for _ in range(PROPAGATE_ITERATIONS):
        F_t_1 = F_t
        F_t = matrix.dot(F_t_1) + Y

        if math.sqrt(scipy.linalg.norm(F_t_1 - F_t)) < PROPAGATE_EPSILON:
            break

    return F_t

def generate_propagate_data(network):
    graph = networkx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)
    matrix, genes = generate_similarity_matrix(graph)
    num_genes = len(genes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])

    return gene_indexes, matrix, num_genes

def generate_feature_columns(network, sources, terminals):
    gene_indexes, matrix, num_genes = generate_propagate_data(network)
    gene1_indexes, gene2_indexes = map(lambda x: tuple([x]), zip(*[[(gene_indexes[gene]) for gene in pair] for pair in network.index]))
    experiments = sorted(sources.keys() & terminals.keys())

    def generate_column(experiment):
        source_scores = propagate(sources[experiment], matrix, gene_indexes, num_genes)
        terminal_scores = propagate(terminals[experiment], matrix, gene_indexes, num_genes)

        gene1_sources = source_scores[gene1_indexes]
        gene2_sources = source_scores[gene2_indexes]
        gene1_terminals = terminal_scores[gene1_indexes]
        gene2_terminals = terminal_scores[gene2_indexes]

        return (gene1_sources * gene2_terminals)/(gene1_terminals * gene2_sources)

    feature_columns = pandas.DataFrame(numpy.column_stack([generate_column(experiment) for experiment in experiments]), index=network.index).fillna(0)
    reverse_columns = (1/feature_columns).replace(numpy.inf, numpy.nan).fillna(0)

    return feature_columns, reverse_columns

def score_network(feature_columns, reverse_columns, directed_interactions):
    training_columns = pandas.concat([feature_columns.loc[directed_interactions.index], reverse_columns.loc[directed_interactions.index]])

    target_labels = numpy.ones(len(directed_interactions))
    reverse_labels = numpy.zeros(len(directed_interactions))
    training_scores = numpy.append(target_labels, reverse_labels)

    classifier = sklearn.linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=0.001).fit(training_columns, training_scores)

    scores = numpy.append(target_labels, classifier.predict_proba(feature_columns.drop(directed_interactions.index))[:,1])
    reverse_scores = numpy.append(reverse_labels, classifier.predict_proba(reverse_columns.drop(directed_interactions.index))[:,1])

    return pandas.DataFrame(numpy.column_stack([scores, reverse_scores]), index=feature_columns.index, columns=[2,3])

def orient_network(network, scores):
    network = network.reset_index()
    scores = scores.reset_index()

    inverted = network[scores[2] < scores[3]]
    network[3] = scores[2]/scores[3]
    network.loc[inverted.index, 3] = scores[3]/scores[2]

    oriented = network[network[3] > 1 + ORIENTATION_EPSILON]
    network[4] = 0
    network.loc[oriented.index, 4] = 1

    swapped = inverted.index & oriented.index
    network.loc[swapped, [0,1]] = network.loc[swapped, [1,0]].values

    return network.set_index([0,1])

network, directed_interactions, sources, terminals = read_data()
merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
feature_columns, reverse_columns = generate_feature_columns(merged_network, sources, terminals)
scores = score_network(feature_columns, reverse_columns, directed_interactions)
oriented_network = orient_network(merged_network, scores)
