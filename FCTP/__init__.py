# -*- coding: utf-8 -*-
"""
Package FCTP - Heuristic methods for solving the FCTP.
"""
__author__ = """Andreas Klose"""
__email__ = 'aklose@math.au.dk'
__version__ = '1.4 (20/03/20)'

# __all__ = ['param', 'sol','optim','FCTP']

from . import api
from . import param
from . import sol

import numpy as np
import networkx as nx

if api.CplexOk: import cplex
if api.CplexOk: from . import optim

BASIC = 1
NON_BASIC = 0
NB_UPPER = 2

from abc import ABC, abstractmethod


class fctp(ABC):
    """
    Abstract base class providing non-abstract methods to access functions
    and procedures from libFCTPy via the "api" module as well as a number
    of abstract method that need to be implemented by classes that extend
    this base class.

    `Attributes:`

        m       : (int) number of suppliers
        n       : (int) number of customers
        nnodes  : (int) number of nodes=m+n
        narcs   : (int) number of arcs = m*n
        err     : (int) error constant (=0 means no error)
        solution: (class "solution") None
        cputime : (float) cpu time used by a method
        lobnd   : (float) a lower bound on the optimal objective value

    `Methods:`

        reset_data        : pass a new data to the FCTP object
        set_start_time    : set time stamp for measuring CPU time
        get_CPU_time      : obtain CPU seconds gone since set_start_time
        get_num_suppliers : return number of suppliers
        get_num_customers : return number of customers
        get_supply        : return a single or all supply quantities
        get_demand        : return a single or all demand quantities
        get_cap           : return a single or all arc capacities
        get_fcost         : return fixed cost on a single or on all arcs
        get_tcost         : return unit cost on a single or on all arcs
        set_supply        : overwrite a single or all supply quantities
        set_demand        : overwrite a single or all demand quantities
        set_cap           : overwrite a single or all arc capacities
        set_fcost         : overwrite a single or all arc fixed cost
        set_tcost         : overwrite a single or all arc unit cost
        greedy            : apply greedy heuristic to construct initial solution
        lp_heuris         : apply LP heuristic to construct initial solution
        local_search      : apply a local search on current basic feasible solution
        grasp             : apply GRASP
        get_obj_val       : return objective value of current solution
        get_flow          : return flow on a single or all arcs in current solution
        get_status        : return status (basic or not) of a single or all arcs
        set_flow          : set the flow on a single or all arcs
        set_base          : build internal basis tree of current solution
        comp_cost         : (re-)compute the cost of current solution
        get_cost_sav      : compute cost saving resulting from putting an arc in the basis
        get_leaving_arcN1 : return index of supplier of arc leaving the basis
        get_leaving_arcN2 : return index of customer of arc leaving the basis
        get_leaving_arc   : return index of arc leaving the basis
        remember_move     : remember the move just investigate by get_cost_sav
        do_move           : apply the move remembered by remember_move
        end               : free memory allocated by the FCTP object
    """

    def initFCTP(self, data_file=None, ini_file=None, **kwargs):
        """
        Initilizes the class fctp.

        `Parameters:`

            data_file : str (optional)
                if present (not None) data are read from the file with
                the name (and path) data_file. This is even done, if
                the initialization file (ini_file) specifies some (other)
                file for the input data.

            ini_file : str (optional)
                if not present (that is None), parameters are specificied
                by the default initialization file named "FCTPheur.ini".
                (Default values apply if even this file is not present).
                Otherwise, parameters are read from the ini-file "ini_file".

            keyword parameters:
                If no datafile is specified, data have to be passed using
                the following type of a dictionary:
                    {"SUPPLY":s,
                     "DEMAND":d,
                     "CAPACITY":c,
                     "FIXEDCOST":f,
                     "UNITCOST";t}
                Therewith,
                    n is the number of customers,
                    m the number of suppliers,
                    s is the numpy int array of suppliers,
                    d is the numpy int array of demands,
                    c is the numpy int array of arc capacities
                      (can be missing or None if the problem is uncapacitated),
                    f is the numpy float array of arc fixed costs, and
                    t is the numpy float array of unit transportation costs.
                    The arc (i,j) from supplier i=0,...,m-1 to customer
                    j=0,...,n-1 has thereby index i*n + j.
        """
        self.history = None  # Trajectory of solution values in one run
        self.all_hist = None  # Trajectory of solution values in all runs
        self.solution = None
        if not data_file is None: param.set(param.input_file, data_file)
        if not ini_file is None:
            param.set(param.init_file, ini_file)
        else:
            param.set(param.init_file, 'FCTPheur.ini')
        param.read_ini_file()
        data_file = param.get(param.input_file)

        if not data_file is None:
            self.err = api.read_data(data_file)
        else:
            s = kwargs.get('SUPPLY', None)
            d = kwargs.get('DEMAND', None)
            c = kwargs.get('CAPACITY', None)
            f = kwargs.get('FIXEDCOST', None)
            t = kwargs.get('UNITCOST', None)
            if not (s is None or d is None or f is None or t is None) \
                    and len(f) == len(t):
                m, n = len(s), len(d)
                mn = m * n if c is None else len(c)
                if m * n == len(f) and mn == len(t):
                    self.err = api.reset_data(m, n, s, d, c, f, t)
                else:
                    self.err = 88
            else:
                self.err = 77
        if self.err == 0:
            self.m = api.get_num_suppliers()
            self.n = api.get_num_customers()
            self.nnodes = self.m + self.n
            self.narcs = self.m * self.n
            self.lobnd = 0.0
            self.cputime = 0.0

    def reset_data(self, m, n, s, d, c, f, t):
        """
        Reset the library's data.

        `Parameters:`

            m : int
                number of suppliers
            n : int
                number of customers
            s : numpy array of int
                supply vector
            d : numpy array of int
                demand vector
            c : numpy array of int
                arc capacities; None if uncapacitated.
                Note that the arc from i to j
                has index i*n+j
            f : numpy array of float
                fixed costs on the arcs, where
                arc no. i*n + j is the arc from
                supplier i to customer j,
                i=0,...,m-1 and j=0,...,n-1
            t : numpy array of float
                unit transp. costs on the arcs

        `Returns:`

            err : int
                  0 if no error; otherwise some positive constant.
        """
        self.m = m
        self.n = n
        self.nnodes = m + n
        self.narcs = m * n
        return api.reset_data(m, n, s, d, c, f, t)

    def set_start_time(self):
        """
        Set a time stamp for measuring computation time.
        """
        api.set_start_time()

    def get_CPU_time(self):
        """
        Return CPU time in seconds gone since last call to set_start_time.
        """
        return api.get_CPU_time()

    def get_num_suppliers(self):
        """
        Returns number of supply nodes in the current FCTP instance
        """
        return api.get_num_suppliers()

    def get_num_customers(self):
        """
        Returns number of customer nodes in current FCTP instance
        """
        return api.get_num_customers()

    def get_supply(self, i=None):
        """
        Returns supplier's i supply if i is not None; otherwise
        the whole array (as numpy array of int) is returned.
        """
        if i is None: return api.get_Supplies()
        return api.get_supply(i)

    def get_demand(self, j=None):
        """
        Returns customer j's demand (as int) if j is not None.
        Otherwise the whole array of demands (as numpy array
        of int) is returned.
        """
        if j is None: return api.get_Demands()
        return api.get_demand(j)

    def get_cap(self, ij=None, arc=None):
        """
        Returns the capacity (as int) of an arc from supplier i to
        customer j or the whole array of arc capacities (as numpy
        array of int).

        `For example:`

            get_cap((1,2)) returns capacity on arc (1,2) that is,
                           from supplier 1 to customer 2;
            get_cap(arc=7) returns capacity on arc no. 7, that is
                           the arc from supplier 7//n to customer 7%n
            get_cap()      returns array of arc capacities
        """
        if not ij is None: return api.get_cap(ij[0], ij[1])
        if not arc is None: return api.get_arc_cap(arc)
        return api.get_Capacities()

    def get_fcost(self, ij=None, arc=None):
        """
        Returns the fixed cost (as float) of the arc (i,j) or the whole array
        of fixed costs (as numpy array of float).

        `For example:`

            get_fcost((1,2)) returns fixed cost on arc from supplier 1
                             to customer 2
            get_fcost(arc=7) returns fixed cost on arc no. 7, that is
                             the arc from supplier 7//n to customer 7%n
            get_fcost()      returns array of fixed cost
        """
        if not ij is None: return api.get_fcost(ij[0], ij[1])
        if not arc is None: return api.get_arc_fcost(arc)
        return api.get_Fcosts()

    def get_tcost(self, ij=None, arc=None):
        """
        Returns the unit transportation cost (as float) of an arc
        or the whole array of these costs (as numpy array of float).

        `For example:`

            get_fcost((1,2)) returns unit cost on arc from supplier 1
                             to customer 2
            get_fcost(arc=7) returns unit cost on arc no. 7, that is
                             the arc from supplier 7//n to customer 7%n
            get_fcost()      returns array of unit costs
        """
        if not ij is None: return api.get_tcost(ij[0], ij[1])
        if not arc is None: return api.get_arc_tcost(arc)
        return api.get_Tcosts()

    def set_supply(self, s, i=None):
        """
        Overwrites the library's supply data with the numpy array
        s of int if i is None. If i is not none, then s should
        be an int and just supplier i's supply is set to the value s.
        """
        if i is None:
            api.set_supplies(s)
        else:
            api.set_supply(i, s)

    def set_demand(self, d, j=None):
        """
        Overwrites the library's demand data with the numpy array
        d of int if j is None. If j is not none, then d should
        be an int and just customer j's demand is set to the value d.
        """
        if j is None:
            api.set_demands(d)
        else:
            api.set_demand(j, d)

    def set_cap(self, c, ij=None, arc=None):
        """
        Sets the capacity on an arc to the value c (int) or sets
        all arc capacities to the values given by the numpy
        array c of int.

        `For example`

            set_cap(c, (1,2)) sets capacity on arc from supplier 1
                              to customer 2 to the value c.
            set_cap(c,arc=7)  sets capacity on arc no. 7 to value c
            set_cap(c)        overwrites library's capacity data with c.
        """
        if not ij is None:
            api.set_cap(ij[0], ij[1], c)
        elif not arc is None:
            api.set_arc_cap(arc, c)
        else:
            api.set_capacities(c)

    def set_fcost(self, f, ij=None, arc=None):
        """
        Sets the fixed cost on an arc to the value f (float) or sets
        all arc fixed costs to the values given by the numpy
        array f of float.

        `For example`

            set_fcost(f, (1,2)) sets fixed cost on arc from supplier 1
                                to customer 2 to the value f.
            set_fcost(f,arc=7)  sets fixed cost on arc no. 7 to value c
            set_fcost(f)        overwrites library's fixed cost data with f.
        """
        if not ij is None:
            api.set_fcost(ij[0], ij[1], f)
        elif not arc is None:
            api.set_arc_fcost(arc, f)
        else:
            api.set_fcosts(f)

    def set_tcost(self, t, ij=None, arc=None):
        """
        Sets the unit cost on an arc to the value t (float) or sets
        all arc unit costs to the values given by the numpy
        array t of float.

        `For example`

            set_tcost(f, (1,2)) sets unit cost on arc from supplier 1
                                to customer 2 to the value t.
            set_fcost(f,arc=7)  sets unit cost on arc no. 7 to value c
            set_fcost(f)        overwrites library's unit cost data with t.
        """
        if not ij is None:
            api.set_tcost(ij[0], ij[1], t)
        elif not arc is None:
            api.set_arc_tcost(arc, t)
        else:
            api.set_tcosts(t)

    def greedy(self, what_meas):
        """
        Greedily constructs a solution for the FCTP and builds
        thereafter the basis tree belonging to this solution.

        `Parameters:`

            what_meas: int
                specifies how to evaluate arcs:
                1 : costs per unit with fixed cost linearized by arc capacity
                2 : costs per unit with fixed cost linearized by remaining arc capacity
                3 : total cost of supplying the remaining quantity on an arc

        `Returns:`

            err: int
                0 if the basis tree was set up successfully; otherwise
                some positive integer.
        """
        api.greedy(what_meas - 1)
        return api.set_base()

    def lp_heuris(self, useCplex=False):
        """
        Computes initial feasible basic solution using Balinski's linear
        programming heuristic which is to solve the transportation problem
        after linearizing the fixed costs. If useCplex is True, the
        transportation problem is solved by means of Cplex's solver;
        otherwise networkx is used. After a solution is obtained, the
        basis tree is set up and 0 returned if this is successful;
        otherwise some positive integer is returned.
        """
        s = api.get_Supplies()
        d = api.get_Demands()
        cap = api.get_Capacities()
        c = api.get_Tcosts() + api.get_Fcosts() / cap
        err = 0

        if useCplex and api.CplexOk:
            cpx = cplex.Cplex()
            ub = lambda arc: cap[arc] if cap[arc] < min(s[arc // self.n], d[arc % self.n]) \
                else cplex.infinity
            cpx.objective.set_sense(cpx.objective.sense.minimize)
            ubnd = [float(ub(arc)) for arc in range(self.narcs)]
            x = cpx.variables.add(obj=c, ub=ubnd)
            # Supply constraints
            lhs = [cplex.SparsePair(x[i * self.n:(i + 1) * self.n], [1] * self.n) for i in self.m]
            cpx.linear_constraints.add(lin_expr=lhs, senses=['E'] * self.m, rhs=s)
            # Demand constraints
            lhs = [cplex.SparsePair([x[i * self.n + j] for i in range(self.m)], [1] * self.m) \
                   for j in range(self.n)]
            cpx.linear_constraints.add(lin_expr=lhs, senses=['E'] * self.n, rhs=d)
            cpx.solve()
            if cpx.solution.is_primal_feasible():
                flows = (np.array(cpx.solution.get_values()) + 0.001).astype(int)
                api.set_flows(flows)
                err = api.set_base()
            else:
                err = 777
            cpx.end()
        else:
            G = nx.DiGraph()
            G.add_nodes_from([i for i in range(self.nnodes)])
            G.add_edges_from([(i, j) for i in range(self.m) \
                              for j in range(self.m, self.nnodes)])
            ndem = {k: {'demand': -s[k] if k < self.m else d[k - self.m]} for k in G.nodes}
            ecost = {e: {'weight': int(round(c[e[0] * self.n + e[1] - self.m] * 1000))} for e in G.edges}
            ecap = {e: {'capacity': cap[e[0] * self.n + e[1] - self.m]} for e in G.edges
                    if cap[e[0] * self.n + e[1] - self.m] < min(s[e[0]], d[e[1] - self.m])}
            nx.set_node_attributes(G, ndem)
            nx.set_edge_attributes(G, ecost)
            if ecap: nx.set_edge_attributes(G, ecap)
            flowCst, flowDict = nx.network_simplex(G)
            flows = np.array([flowDict[e // self.n][self.m + e % self.n] \
                              for e in range(self.narcs)], dtype=int)
            api.set_flows(flows)
            objval = api.comp_cost()
            err = api.set_base()
        return err

    def local_search(self, createBasTree=False):
        """
        Invokes the library's local search procedure to be applied on the
        internally stored current basic feasible solution.
        If createBasTree is true, then the basis tree is first setup
        Otherwise, it is assumed that this has already been done (the default)!
        If the procedure fails to "createBasTree" then a positive error code
        is returned; otherwise 0.
        """
        return api.local_search(createBasTree)

    def grasp(self, what_meas, max_iter, alpha):
        """
        Applies a Grasp procedure to the FCTP. See FCTPgreedy regarding the meaning
        of the parameters "what_meas" and "alpha". The GRASP procedure stops if
        after max_iter subsequent iterations the best solution found so far is not
        improved. If an error occurs (when trying to set a solution as basis),
        a positive error code is returned; otherwise 0.
        """
        return api.grasp(what_meas, max_iter, alpha)

    def get_obj_val(self):
        """
        Return object value of current solution.
        """
        return api.get_obj_val()

    def get_flow(self, ij=None, arc=None):
        """
        Return flow (as int) on arc i-> in current solution or the
        numpy array of all flows (array of int). For example,
        get_flow( (1,2) ) returns the flow in the library's
        current solution from supplier 1 to customer 2,
        get_flow(arc=7) does the similar for arc no. 7,
        and get_flow returns the whole array of flows.
        """
        if not ij is None: return api.get_flow(ij[0], ij[1])
        if not arc is None: return api.get_arc_flow(arc)
        return api.get_Flows()

    def get_status(self, ij=None, arc=None):
        """
        Returns the status of the arc i->j in the current basic solution
        or the stati of all arcs in an numpy array of int.
        For example, get_status( (1,2) ) returns the status of the arc
        from supplier 1 to customer 2; get_status( arc=7 ) returns
        the status of arc no. 7, and get_status() returns a numpy array
        of int indicating the status of each arc.
        """
        if not ij is None: return api.get_status(ij[0], ij[1])
        if not arc is None: return api.get_arc_status(arc)
        return api.get_arc_stats()

    def set_flow(self, flow, ij=None, arc=None):
        """
        Set the flow on a single arc arc i->j to the int value flow or
        overwrites the library's flows in the current solution
        by the values given in the numpy array flow of int.
        For example, set_flow( flow, ij=(1,2)) overwrites the
        current flow on the arc from supplier 1 to customer 2
        with the int value flow; set_flow( flow, arc=7) does
        the similar for arc no. 7, and set_flow(flow) expects
        that flow is a numpy array of int of dimension at least
        equal to the number of arcs and overwrites all flow data
        with the values given in the array "flow".
        """
        if not ij is None:
            api.set_flow(ij[0], ij[1], flow)
        elif not arc is None:
            api.set_arc_flow(arc, flow)
        else:
            api.set_flows(flow)

    def comp_cost(self):
        """
        (Re-)computes the cost of the current basic feasible solution
        """
        api.comp_cost()

    def set_base(self):
        """
        Builds the basis tree corresponding to the current solution.
        Returns 0 if successful; otherwise a positive error code
        """
        return api.set_base()

    def get_cost_sav(self, ij=None, arc=None):
        """
        Compute and return the cost savings that can be achieved by
        introducing the arc i->j into the basis. For example,
        get_cost_sav( (i,j) ) and get_cost_sav( arc=i*n+j ) both
        return the cost saving achieved if the arc from supplier i
        to customer j is made a basic arc (0 is returned if
        the arc is already in the basis).
        """
        if not ij is None: return api.get_cost_sav(ij[0], ij[1])
        if not arc is None: return api.get_arc_cost_sav(arc)
        return 0

    def get_leaving_arcN1(self):
        """
        Return the index of the supplier belonging to the arc leaving the basis.
        Note that the leaving arc is computed after a call to get_cost_sav!
        """
        return api.get_leaving_arcN1()

    def get_leaving_arcN2(self):
        """
        Return the index of the customer belonging to the arc leaving the basis.
        Note that the leaving arc is computed after a call to get_cost_sav!
        """
        return api.get_leaving_arcN2()

    def get_leaving_arc(self):
        """
        Return the index of the arc leaving the basis
        """
        return api.get_leaving_arc()

    def remember_move(self):
        """
        Stores the data of a basic exchange (move) investigated by the
        most recent call to get_cost_sav. This method need to be
        called if later the moven (that is basic exchange) should
        perhaps also be performed.
        """
        api.remember_move()

    def do_move(self):
        """
        Applies the move remembered by method "remember_move".
        """
        api.do_move()

    def is_degenerated(self):
        """
        Returns true if a tentative basic exchange that just was
        evaluated using function get_cost_sav() is degenerate.
        Note tht this function may only be called after get_cost_sav
        was called before.
        """
        return api.is_degenerated()

    def end(self):
        """
        Free all memory allocated by the library to keep all direct
        and intermediate data.
        """
        api.FCTPclose()

    def solve(self):

        # First construct an initial feasible solution
        greed = param.get(param.greedy_meas)
        if greed == 0:
            # Construct solution by means of LP-heuristic
            err = self.lp_heuris()
        else:
            # Greedily construct initial feasible solution
            err = self.greedy(greed)
        if err > 0: return err

        # Get the improvement method to apply
        imp_meth = param.get(param.improve_method)
        nruns = param.get(param.num_runs)
        do_restart = nruns > 1 and param.get(param.do_restart) == param.yes
        do_optim = api.CplexOk and imp_meth >= param.cpx_opt
        imp_meth %= param.cpx_opt

        # Store the start solution if required
        self.solution = sol.solution()
        start_sol = sol.solution(self.solution) if do_restart else None
        best_sol = sol.solution(start_sol) if nruns > 1 else None
        start_time = self.get_CPU_time()

        if imp_meth == param.ls:
            self.local_search()
            self.solution.over_write()
        else:
            self.all_hist = []
            for itr in range(nruns):
                self.history = None
                if imp_meth == param.msls:
                    self.msls()
                elif imp_meth == param.ils or imp_meth == param.ils_rtr:
                    self.ils()
                elif imp_meth == param.sa:
                    self.sa()
                elif imp_meth == param.sa_osman:
                    self.Osman_sa()
                elif imp_meth == param.rtr or imp_meth == param.rtr_ils:
                    self.rtr()
                else:
                    break
                if not self.history is None: self.all_hist.append(self.history)
                if nruns == 1: break
                if self.solution.tot_cost < best_sol.tot_cost:
                    best_sol.over_write(self.solution)
                if do_restart == param.yes:
                    # Reset solution to start solution
                    start_sol.make_basic()
                else:
                    best_sol.make_basic()
            if nruns > 1:
                best_sol.make_basic()
                self.solution.over_write(best_sol)
            self.cputime = self.get_CPU_time() - start_time

        # Eventually call Cplex to try to find better or optimal solution
        if do_optim: optim.cpxSolve(self)

        return 0

    # Abstract methods follow. These methods need to be implemented
    # in classes that inherit methods from this class.
    @abstractmethod
    def msls(self):
        """
        Multi-start local search. Initial solution need to be available
        at the library and in the instance self.solution of the solution
        class. The best solution reached is also stored in this solution
        instance.
        """
        pass

    @abstractmethod
    def ils(self):
        """
        Iterated Local Search. W.r.t initial and final solution the same
        holds as in case of method msls.
        """
        pass

    @abstractmethod
    def sa(self):
        """
        Simulated annealing procedure.
        """
        pass

    @abstractmethod
    def Osman_sa(self):
        """
        Simulated annealing procedure for the FCTP, similar to Osman's
        (1995) SA procedure for the generalised assignment problem.

        Osman I. (1995). Heuristics for the Generalized Assignment Problem:
        Simulated Annealing and Tabu Search Approaches. Oper. Res. Spectrum
        17(4):211--225.
        """
        pass

    @abstractmethod
    def rtr(self):
        """
        Record-to-record travel procedure for the FCTP. The procedure
        works in a similar way as the RTR heuristic for the dynamic
        travelling salesman problem suggested in
        Li, Golden, Wasil (1995).
        """
        pass