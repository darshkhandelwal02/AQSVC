from deap import base, creator, tools, algorithms
import fitness
import random
import numpy as np
import sklearn

def gsvm(nqubits, depth, nparameters, X, y,
         mu=100, lambda_=150, cxpb=0.7, mutpb=0.3, ngen=2000,
         use_pareto=True, verbose=True, weights=[-1.0,1.0],
         debug=True):
    print('multi')
    bits_puerta = 5
    long_cadena = depth * nqubits * bits_puerta
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness = creator.FitnessMulti, statistics=dict)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("Individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, long_cadena)
    toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.2)
    toolbox.register('select', tools.selNSGA2)
    toolbox.register("evaluate", fitness.Fitness_acc(nqubits, nparameters, X, y, debug=debug))
    pop = toolbox.Population(n=mu)
    stats1 = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats1.register('media',np.mean)
    stats1.register('std',np.std)
    stats1.register('max',np.max)
    stats1.register('min',np.min)
    logbook = tools.Logbook()
    pareto = tools.ParetoFront(similar = np.array_equal)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                             mu, lambda_, cxpb, mutpb, ngen,
                                             stats=stats1,
                                             halloffame=pareto, verbose=verbose)
    pareto.update(pop)
    return pop, pareto, logbook

def gsvm_recall(nqubits, depth, nparameters, X, y,
         mu=100, lambda_=150, cxpb=0.7, mutpb=0.3, ngen=2000,
         use_pareto=True, verbose=True, weights=[-1.0,1.0],
         debug=True):
    print('multi')
    bits_puerta = 5
    long_cadena = depth * nqubits * bits_puerta
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness = creator.FitnessMulti, statistics=dict)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("Individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, long_cadena)
    toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.2)
    toolbox.register('select', tools.selNSGA2)
    toolbox.register("evaluate", fitness.Fitness_recall(nqubits, nparameters, X, y, debug=debug))
    pop = toolbox.Population(n=mu)
    stats1 = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats1.register('media',np.mean)
    stats1.register('std',np.std)
    stats1.register('max',np.max)
    stats1.register('min',np.min)
    logbook = tools.Logbook()
    pareto = tools.ParetoFront(similar = np.array_equal)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                             mu, lambda_, cxpb, mutpb, ngen,
                                             stats=stats1,
                                             halloffame=pareto, verbose=verbose)
    pareto.update(pop)
    return pop, pareto, logbook

def gsvm_recall_single(nqubits, depth, nparameters, X, y,
                       mu=100, lambda_=150, cxpb=0.7, mutpb=0.3, ngen=2000,
                       verbose=True, debug=True):

    # Define the number of bits per gate
    bits_per_gate = 5
    chromosome_length = depth * nqubits * bits_per_gate

    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except AttributeError:
        pass  # Avoid redefinition error

    # Define the genetic algorithm toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("Individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=chromosome_length)
    toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
    population = toolbox.Population(n=mu)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # Flip mutation
    toolbox.register("select", tools.selNSGA2) 
    fitness_function = fitness.Fitness_recall_single(nqubits, nparameters, X, y, debug=debug)
    toolbox.register("evaluate", fitness_function)
    # Initialize the population
    # Set up statistics to track progress
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register('mean', np.mean)
    stats.register('std', np.std)
    stats.register('max', np.max)
    stats.register('min', np.min)

    # Logbook for tracking progress
    logbook = tools.Logbook()
    pareto = tools.ParetoFront(similar = np.array_equal)
    # Run the genetic algorithm
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox,
        mu=mu, lambda_=lambda_,
        cxpb=cxpb, mutpb=mutpb,
        ngen=ngen, stats=stats, halloffame= pareto,
        verbose=verbose
    )
    pareto.update(population)
    # Return the final population and logbook
    return population, logbook
