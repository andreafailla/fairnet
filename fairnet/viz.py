import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "plot_GA_eval",
    "plot_marginalization_scores",
    "plot_marginalization_scores_by_attr",
]


def plot_GA_eval(logbook, fitness):
    """
    _summary_

    :param logbook: _description_
    :param fitness: _description_
    """
    sns.set_style("whitegrid")
    minFitnessValues, meanFitnessValues = logbook.select("best", "avg")

    plt.plot(minFitnessValues, color="blue", label="best")
    plt.plot(meanFitnessValues, color="green", label="avg")

    plt.xlabel("Generation")
    if fitness == "nodes":
        plt.ylabel("Marginalized Nodes")
    elif fitness == "marg":
        plt.ylabel("Marginalization Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_marginalization_scores(marg_dict):
    """
    _summary_

    :param marg_dict: _description_
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.kdeplot(list(marg_dict.values()), ax=ax)
    plt.xlabel("Marginalization Scores")
    ax.set_xlim(-1, 1)
    plt.show()


def plot_marginalization_scores_by_attr(attrs, marg_dict):
    """
    _summary_

    :param attrs: _description_
    :param marg_dict: _description_
    """
    _, ax = plt.subplots()
    for label in set(list(attrs.values())):
        labeled_nodes = [k for k, v in attrs.items() if v == label]
        labeled = [v for k, v in marg_dict.items() if k in labeled_nodes]
        sns.kdeplot(labeled, ax=ax, label=label)

    plt.xlabel("Marginalization Scores")
    ax.set_xlim(-1, 1)
    plt.legend()
    plt.show()
