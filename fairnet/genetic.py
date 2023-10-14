import numpy as np
from deap import creator, base, tools
import copy
from .marginalization import *

__all__ = ["reduce_marginalization_genetic", "replace_missing_values_genetic"]


def random_individual(fn):
    """
    generates a random individual for the GA
    :param fn: the FairNet object
    :return: a list of 0s and 1s, where 1s
    """

    return list(np.random.choice(a=[0, 1], size=(len(fn.candidates))))


def evaluate_marginalization(
        individual, fn, return_net,
):
    """
    Evaluation function for the GA.
    It computes the marginalization score of the network after applying the solution.
    :param individual:
    :param fn:
    :param return_net:
    :return:
    """
    individual = individual[0]  # <- because DEAP

    eva_g = fn.g.copy()  # copy of OG network, modified for testing the solution

    indexes = [i for i, j in enumerate(individual) if j == 1]

    all_edges = [fn.candidates[i][:2] for i in indexes]

    affected = set()  # nodes affected by tested solution
    for u, v in all_edges:
        affected.add(u)
        affected.add(v)

    for e in all_edges:
        if eva_g.has_edge(*e):
            eva_g.remove_edge(*e)
        else:
            eva_g.add_edge(*e)

    num_marg_nodes = (
        0  # amount of marginalized nodes in network after applying the solution
    )

    fair_marg = []  # marg score of network after applying the solution
    for node in eva_g.nodes():
        marg = individual_marginalization_score(eva_g, node, fn.attrs, fn.weights)
        fair_marg.append(abs(marg))
        if abs(marg) > fn.thresh:
            num_marg_nodes += 1

    budget = sum(individual)

    if return_net:
        return eva_g, budget, individual

    if fn.fitness == "nodes":
        return num_marg_nodes, budget, np.mean(fair_marg)
    else:  # fn.fitness == "marg":
        return np.mean(fair_marg), budget, num_marg_nodes
    # elif fn.fitness == "round":
    #    return round(np.mean(fair_marg), 2), budget, num_marg_nodes


def reduce_marginalization_genetic(fn, GA_params):
    """
    _summary_

    :param fn: _description_
    :param GA_params: _description_
    :return: _description_
    """
    if GA_params is None:
        GA_params = {
            "NUM_GENERATIONS": 30,
            "POPULATION_SIZE": 150,
            "CXPB": 0.5,
            "MUTPB": 0.25,
        }

    return_net = False  # if True, eva function returns the fair network. INSIDE THE GA, IT MUST BE FALSE

    creator.create(
        "Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0)
    )  # fitness function
    creator.create(
        "Individual", list, fitness=creator.Fitness
    )  # individual class (list) with fitness attribute

    toolbox = base.Toolbox()  # creiamo il toolbox

    toolbox.register("random_individual", random_individual, fn)

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.random_individual,
        n=1,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate", evaluate_marginalization, fn=fn, return_net=return_net
    )  # funzione di valutazione. Vedi quanto detto sopra
    toolbox.register("mate", tools.cxUniform, indpb=0.50)  # funzione di crossover
    toolbox.register(
        "mutate", tools.mutFlipBit, indpb=0.2
    )  # funzione di mutazione custom
    toolbox.register("select", tools.selTournament, tournsize=3)

    NUM_GENERATIONS = GA_params["NUM_GENERATIONS"]  # numero di generazioni
    POPULATION_SIZE = GA_params["POPULATION_SIZE"]  # popolazione per gen

    CXPB, MUTPB = (
        GA_params["CXPB"],
        GA_params["MUTPB"],
    )  # crossover and mutation probability

    n_HOF = 1  # number of individuals to keep in the hall of fame

    pop = toolbox.population(n=POPULATION_SIZE)

    hof = tools.HallOfFame(n_HOF)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("best", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "best", "avg", "other", "budget"

    invalid_individuals = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(
        gen=0, budget=hof[0].fitness.values[1], other=hof[0].fitness.values[2], **record
    )
    print(logbook.stream)

    for gen in range(1, NUM_GENERATIONS + 1):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random_sample() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random_sample() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(
            gen=gen,
            budget=hof[0].fitness.values[1],
            other=hof[0].fitness.values[2],
            **record
        )
        print(logbook.stream)

    hof.update(pop)
    g, _, individual = evaluate_marginalization(hof.items[0], fn=fn, return_net=True)
    return g, logbook, individual


def random_individual_missing(fn):
    """
    generates a random individual for the GA (missing values)
    :param fn: the FairNet object
    :return: a list of 0s and 1s 
    """
    return list(np.random.choice(a=list(fn.attrs.values()), size=len(fn.missing)))


def mutate_missing(individual, indpb, fn):
    """
    _summary_

    :param individual: _description_
    :param indpb: _description_
    :param fn: _description_
    :return: _description_
    """

    for i in range(len(individual)):
        if np.random.random_sample() < indpb:
            individual[i] = np.random.choice([attr for attr in set((list(fn.attrs.values()))) if attr != individual[i]])

    return (individual,)


def evaluate_missing(individual, fn, return_net):
    """
    Evaluation function for the GA (missing values).
    :param individual:
    :param fn: 
    :param return_net: 
    :return: 
    """
    individual = individual[0]  # <- because DEAP

    for node, attr in zip(fn.missing, individual):
        fn.attrs[node] = attr

    weights = compute_weights(fn.attrs)

    marg_dict = compute_marginalization_scores(fn.g, fn.attrs, weights)

    disc_nodes = [k for k, v in marg_dict.items() if abs(v) > fn.thresh]

    overall_marg = np.mean(
        [abs(v) for v in marg_dict.values()]
    )  # network marginalization score

    if return_net:
        return fn.attrs

    if fn.fitness == "marg":
        return overall_marg, len(disc_nodes)
    else:  # fn.fitness == "nodes":
        return len(disc_nodes), overall_marg
    # elif fn.fitness == "round":
    #   return round(np.mean(fair_marg), 2), budget, num_marg_nodes


def replace_missing_values_genetic(fn, GA_params):
    """
    Runs the GA for replacing missing values.
    :param fn: the FairNet object
    :param GA_params: the GA parameters
    :return: 
    """

    if GA_params is None:
        GA_params = {
            "NUM_GENERATIONS": 30,
            "POPULATION_SIZE": 150,
            "CXPB": 0.5,
            "MUTPB": 0.25,
        }

    creator.create(
        "Fitness", base.Fitness, weights=(-1.0, -1.0)
    )
    creator.create(
        "Individual", list, fitness=creator.Fitness
    )

    toolbox = base.Toolbox()

    toolbox.register("random_individual_missing", random_individual_missing, fn=fn)


    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.random_individual_missing,
        n=1,
    )


    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate_missing", evaluate_missing, fn=fn, return_net=False,
    )
    toolbox.register("mate", tools.cxUniform, indpb=0.50)
    toolbox.register(
        "mutate_missing", mutate_missing, indpb=0.05, fn=fn
    )
    toolbox.register("select", tools.selTournament, tournsize=3)


    print("Fitness:", fn.fitness)
    NUM_GENERATIONS = GA_params["NUM_GENERATIONS"]
    POPULATION_SIZE = GA_params["POPULATION_SIZE"]

    CXPB, MUTPB = (
        GA_params["CXPB"],
        GA_params["MUTPB"],
    )

    n_HOF = 1

    pop = toolbox.population(n=POPULATION_SIZE)

    hof = tools.HallOfFame(n_HOF)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("best", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen"] + stats.fields

    invalid_individuals = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate_missing, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, **record)

    print(logbook.stream)

    for gen in range(1, NUM_GENERATIONS + 1):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random_sample() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random_sample() < MUTPB:
                toolbox.mutate_missing(mutant[0])
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate_missing, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, **record)

        print(logbook.stream)

    hof.update(pop)

    attrs = evaluate_missing(hof.items[0], fn=fn, return_net=True)

    return attrs, logbook
