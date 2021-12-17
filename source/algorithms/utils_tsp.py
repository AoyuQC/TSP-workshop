# imports
import dimod
import dwave_networkx as dnx
import networkx as nx
import numpy as np


# helper function to compute distance from route
def get_distance(route, data):
    """
    find distance for given route = [0, 4, 3, 1, 2] and original data
    """
    # get the total distance without return
    total_dist = 0
    for idx, node in enumerate(route[:-1]):
        dist = data[route[idx + 1]][route[idx]]
        total_dist += dist

    print("Total distance (without return):", total_dist)

    # add distance between start and end point to complete cycle
    return_distance = data[route[0]][route[-1]]
    # print('Distance between start and end:', return_distance)

    # get distance for full cyle
    distance_with_return = total_dist + return_distance
    print("Total distance (including return):", distance_with_return)

    return total_dist, distance_with_return


# helper function for solving TSP with D-Wave adapted from Ocean
# including some heuristical filling if not all contraints have been met
def traveling_salesperson(
    G, sampler=None, lagrange=None, weight="weight", start=None, **sampler_args
):
    """Returns an approximate minimum traveling salesperson route.

    Defines a QUBO with ground states corresponding to the
    minimum routes and uses the sampler to sample
    from it.

    A route is a cycle in the graph that reaches each node exactly once.
    A minimum route is a route with the smallest total edge weight.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum traveling salesperson route.
        This should be a complete graph with non-zero weights on every edge.

    sampler :
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    lagrange : number, optional (default None)
        Lagrange parameter to weight constraints (visit every city once)
        versus objective (shortest distance route).

    weight : optional (default 'weight')
        The name of the edge attribute containing the weight.

    start : node, optional
        If provided, the route will begin at `start`.

    sampler_args :
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    route : list
       List of nodes in order to be visited on a route

    Examples
    --------

    >>> import dimod
    ...
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from({(0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1),
    ...                            (1, 3, .5), (2, 3, .1)})
    >>> dnx.traveling_salesperson(G, dimod.ExactSolver(), start=0) # doctest: +SKIP
    [0, 1, 2, 3]

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # get lists with all cities
    list_cities = list(G.nodes())

    # Get a QUBO representation of the problem
    Q = dnx.traveling_salesperson_qubo(G, 3*lagrange, weight)

    # use the sampler to find low energy states
    # response = sampler.sample_qubo(Q, **sampler_args)
    # sample = response.first.sample
    # sample = sampler.sample_qubo(Q, **sampler_args).first.sample
    # response = sampler.sample_qubo(Q, **sampler_args)
    # aggregate = response.aggregate()
    # print(aggregate)
    # sample = response.first.sample

    slice = sampler.sample_qubo(Q, **sampler_args).aggregate().slice(2000)

    sample_set = slice

    route_list = []

    for sample in sample_set:
        print(dict(sample))


        # fill route with None values
        # route = [None] * len(G)
        # get cities from sample
        routes = []
        for i in range(len(G)):
            routes.append([])
        # NOTE: Prevent duplicate city entries by enforcing only one occurrence per city along route
        for (city, time), val in sample.items():
            # if val and (city not in route):
            #     route[time] = city
            if val:
                routes[time].append(city)
        print(f"routes {routes}")
        
        def gen_multiple_route(routes, up_route, complete_routes):
            # print(f"iterate routes {routes}, up_route {up_route}, complte_route {complete_routes}")
            current_time_city = routes[0]
            current_len = len(routes)
            if len(current_time_city) == 0:
                # update sequence
                current_route = up_route.copy()
                current_route.append(None)
                if current_len > 1:
                    gen_multiple_route(routes[1:], current_route, complete_routes)
                elif current_len == 1:
                    complete_routes.append(current_route)

            for city in current_time_city:
                current_route = up_route.copy()
                if city in current_route:
                    for i in range(2):
                        if i == 0:
                            update_current_route = [None if ci == city else ci for ci in current_route]
                            update_current_route.append(city)
                            
                            if current_len > 1:
                                gen_multiple_route(routes[1:], update_current_route, complete_routes)
                            elif current_len == 1:
                                complete_routes.append(update_current_route)
                        else:
                            current_route.append(None)
                else:
                    current_route.append(city)
                            
                if current_len > 1:
                    gen_multiple_route(routes[1:], current_route, complete_routes)
                elif current_len == 1:
                    complete_routes.append(current_route)

            return 0
        
        filter_routes_list = []
        gen_multiple_route(routes, [], filter_routes_list)

        print(f"fitler_routes_list {filter_routes_list}")

        # run heuristic replacing None values
        for route in filter_routes_list:
            if None in route:
                # print(f"found None! route: {route}")
                # get not assigned cities
                cities_unassigned = [city for city in list_cities if city not in route]
                cities_unassigned = list(np.random.permutation(cities_unassigned))
                for idx, city in enumerate(route):
                    if city == None:
                        route[idx] = cities_unassigned[0]
                        cities_unassigned.remove(route[idx])

            # cycle solution to start at provided start location
            if start is not None and route[0] != start:
                # rotate to put the start in front
                idx = route.index(start)
                route = route[idx:] + route[:idx]
        
            route_list.append(route)

    return route_list
