"""
   Class extFCTP that is used to implement abstract methods 
   from the base class FCTP.fctp.

   author:
   version:
       01/01/20
       21/02/20 ( 2nd Tutorial, F2020)
       02/03/20 (10th Lecture,  F2020)     
       06/03/20 ( 3rd Tutorial, F2020)  
"""

import FCTP
import numpy as np
import math

class extFCTP( FCTP.fctp ):
    """
    Class that extends the base class and provides implementations
    of abstract methods of the base class.
    """

    #------------------------------------------------------------------------------ 
    
    def __init__(self, data_file=None, ini_file=None, **kwargs ):
        """
        Initializes the base class.

        Parameters:
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
                Therewith, n is the number of customers, m the number of suppliers,
                s is the numpy int array of suppliers, d is the numpy int array of
                demands, c is the numpy int array of arc capacities (can be missing or
                be None if the problem is uncapacitated), f is the numpy float array 
                of arc fixed costs, and t is the numpy float array of unit transportation 
                costs on  the arcs. An arc (i,j) from supplier i=0,...,m-1 to customer 
                j=0,...,n-1 has thereby index i*n + j.             
        """
        super().initFCTP( data_file=data_file,ini_file=ini_file, **kwargs )
        self.headlen = None
    
    #------------------------------------------------------------------------------    
    
    def give_info( self, *args, title=None ):
        if title:
            print('-'*60)
            print(title)
            print('-'*60)
            self.headlen = [ len(a) for a in args]       
            for a in args: print(a,end=' ')
        else:
            for i, a in enumerate(args): 
                l = self.headlen[i]
                formstr = ''.join(['{0:>',str(l),'s}'])
                astr = str(a)
                if len(astr) > l: astr=astr[:l]
                print(formstr.format(astr),end=' ')
        print()
    
    #------------------------------------------------------------------------------                        
    
    def greedy(self, what_meas, alpha=0.0 ):
        """       
        We overwrite the built-in Greedy method with our own version.
        
        Parameters:
            what_meas : int
                Indicates which type of greedy function should be applied.
                1 : costs per unit with fixed cost linearized by arc capacity
                2 : costs per unit with fixed cost linearized by remaining arc capacity
                3 : total cost of supplying the remaining quantity on an arc
                4 : total cost of supplying the given capacity on an arc
            alpha : float
                alpha=0.0 -> greedily create basic feasible solution
                alpha=1.0 -> randomly create basic feasible solution a    
        """
        tcost = self.get_tcost().reshape(self.m,self.n)
        fcost = self.get_fcost().reshape(self.m,self.n)
        capaci = self.get_cap().reshape(self.m,self.n)
        lincst = fcost + tcost*capaci if what_meas>2 else tcost + fcost/capaci
        flows = np.zeros( self.narcs, dtype=int )
        d_rem = self.get_demand()
        s_rem = self.get_supply()
        customers = list( range(self.n) )
        suppliers = list( range(self.m) )
        
        if alpha == 1.0:
            choose_ij = lambda mn: (np.random.randint(mn[0]),np.random.randint(mn[1]))
        else:
            choose_ij = lambda mn : (np.unravel_index( np.argmin(lincst[:mn[0],:mn[1]], axis=None), mn) )     

        mm = self.m # number of suppliers (still) showing positive supply
        nn = self.n # number of customers (still) showing positive demand
        while mm > 0 or nn > 0:
             
            #i, j = np.unravel_index( np.argmin(lincst, axis=None), lincst.shape )
            i,j = choose_ij( ( mm, nn ) )
            ii, jj = suppliers[i], customers[j]
            xij  = min( s_rem[i], d_rem[j] )
            flows[ ii*self.n+jj ] = xij
            d_rem[j] -= xij
            s_rem[i] -= xij
            d_zero = d_rem[j]==0
            s_zero = s_rem[i]==0
            if d_zero:
                # Remove "column"/customer j from the problem by overwriting it with last column
                nn -= 1
                customers[j] = customers[nn]
                lincst[:,j] = lincst[:,nn]
                d_rem[j] = d_rem[nn]
                if 1 < what_meas < 4:
                    # Greedy measure is dynamic => we have to adjust the linearized costs 
                    tcost[:,j]  = tcost[:,nn]
                    fcost[:,j]  = fcost[:,nn]
                    capaci[:,j] = capaci[:,nn] 
                    if not s_zero:
                        capaci[i,:nn] = np.where( d_rem[:nn] < s_rem[i], d_rem[:nn], s_rem[i] )
                        lincst[i,:nn] = fcost[i,:nn] + tcost[i,:nn]*capaci[i,:nn] if what_meas==3 else\
                                        tcost[i,:nn] + fcost[i,:nn]/capaci[i,:nn]
            if s_zero:
                # Remove "row"/supplier i from the problem by overwriting with last row
                mm -= 1
                suppliers[i] = suppliers[mm]
                lincst[i] = lincst[mm]
                s_rem[i] = s_rem[mm]
                if 1 < what_meas < 4:
                    # Greedy measure is dynamic => we have to adjust the linearized costs
                    tcost[i]  = tcost[mm]
                    fcost[i]  = fcost[mm]
                    capaci[i] = capaci[mm] 
                    if not d_zero:
                        capaci[:mm,j] = np.where( s_rem[:mm] < d_rem[j], s_rem[:mm], d_rem[j] )
                        lincst[:mm,j] = fcost[:mm,j] + tcost[:mm,j]*capaci[:mm,j] if what_meas==3 else\
                                        tcost[:mm,j] + fcost[:mm,j]/capaci[:mm,j]                                         
                
        self.set_flow( flows )
        self.comp_cost()
        err = self.set_base( )
        return err        
    
    #------------------------------------------------------------------------------
    
    def local_search ( self ):
        """
        We overwrite the local search method of the base class FCTP
        """
        if FCTP.param.get(FCTP.param.ls_type)==FCTP.param.first_accept:
            self.fa_ls()
        else:
            self.ba_ls()    
        return 0    
   
    #------------------------------------------------------------------------------ 
    
    def fa_ls( self ):
        """
        We implement a first-accept local search:
        """
        num_checked = 0
        arc = -1
        while num_checked < self.narcs:
            arc = (arc+1) % self.narcs
            num_checked += 1
            if self.get_status(arc=arc) != FCTP.BASIC and self.get_cost_sav(arc=arc) > 0.0:
                self.remember_move()
                self.do_move()
                num_checked = 0
          
    #------------------------------------------------------------------------------                       
    
    def ba_ls( self ):
        """
        We implement a best-accept local search
        """
        nb_arcs = np.where( self.get_status() != FCTP.BASIC )[0] 
        while True:
            saving = np.fromiter((self.get_cost_sav( arc=a) for a in nb_arcs),float)
            indx = np.argmax( saving )
            arc = nb_arcs[indx]
            if saving[indx] < FCTP.param.tol_default: break
            self.get_cost_sav(arc=arc)
            self.remember_move()
            self.do_move()
            nb_arcs[indx] = self.get_leaving_arc()

    #------------------------------------------------------------------------------
                              
    def msls( self ):
        """
        Multi-start local search. Starts from a random initial feasible basic
        solution and applies local search to it. The first solution is, however,
        obtained by applying local search to the initial solution obtained
        by the construction heuristic determined in the configuration file.
        """        
        # Apply local search to the initial solution stored in the library

    #------------------------------------------------------------------------------             

    def ils( self ):
        """
        Iterated Local Search applied to the FCTP.
        """
        self.local_search()
        self.history = [self.get_obj_val()]
        k_step = 5
        max_iter = FCTP.param.get(FCTP.param.max_iter)
        best_sol = FCTP.sol.solution()
        inform = FCTP.param.get(FCTP.param.screen) == FCTP.param.on
        if inform:
            self.give_info("Iter", "Before LS", "After LS", "Best_sol", title="Multi-start local search")
        for itr in range(max_iter):
            for k in range(k_step):
                nb_arcs = np.where(self.get_status() != FCTP.BASIC)[0]  # Get edges to look at
                costs = np.zeros_like(nb_arcs)
                i = 0
                for arc in nb_arcs:
                    saving = self.get_cost_sav(arc=arc)
                    if saving < 0:
                        costs[i] = -saving
                    i += 1
                arcs_to_choose_from = nb_arcs[costs != 0]
                inv_costs = 1 / (costs[costs != 0])
                probs = inv_costs / np.sum(inv_costs)
                choice = np.random.choice(arcs_to_choose_from, p=probs)
                self.get_cost_sav(arc=choice)
                self.remember_move()
                self.do_move()
            before_LS = self.get_obj_val()
            self.local_search()
            after_LS = self.get_obj_val()
            if after_LS < best_sol.tot_cost:
                best_sol.over_write()
            if inform:
                self.give_info(itr, before_LS, after_LS, best_sol.tot_cost)
            self.history.append(after_LS)
        best_sol.make_basic()
        self.solution.over_write(best_sol)
    
    #------------------------------------------------------------------------------      
    
    def sa( self ):
        """
        Applies a standard simulated annealing procedure to the FCTP. 
        """    
    
        # Intialize iteration counter, best solution, non-basic arcs, sample size
        num_moves = 0 
        iterat = 0
        best_sol = FCTP.sol.solution( self.solution )      
        nb_arcs = np.where(self.get_status()!=FCTP.BASIC)[0].astype(int)
        sample_size = num_nb = nb_arcs.shape[0]
        
        # Retrieve parametes used in the SA
        sample_growth = FCTP.param.get( FCTP.param.sample_growth )
        sa_beta  = FCTP.param.get( FCTP.param.sa_cool_beta )
        min_rate = FCTP.param.get( FCTP.param.min_acc_rate )
        ini_rate = FCTP.param.get( FCTP.param.ini_acc_rate )
        max_fail = FCTP.param.get( FCTP.param.max_no_imp )

        # Fix initial temperature. so that initial acceptance rate is 
        # about FCTPparam.ini_acc_rate*100 %
        mean = sum( min(0.0,self.get_cost_sav(arc=a)) for a in nb_arcs )/num_nb
        temp = mean/math.log( ini_rate )
          
        # Say hello
        inform = FCTP.param.get(FCTP.param.screen) == FCTP.param.on;
        if inform: 
            self.give_info ("Iter","Temperature","Sample_size","Acc_rate",\
                            "Current_Obj","Incumbent",title="Simulated annealing") 
    
        self.history = [ best_sol.tot_cost ]
        # Main loop  
        num_fail = 0
        go_on = True;
        while go_on:
            iterat += 1
            # Sample at current temperature 
            improve = False
            non_degen = num_nb
            num_accepted = 0
            count = 0
            while count < sample_size:
                count += 1
                if non_degen == 0: break
                # Make a random basic exchange but avoid degenerate ones 
                is_degen = True;
                while is_degen and non_degen > 0: 
                    indx = np.random.randint(non_degen)
                    saving = self.get_cost_sav( arc=nb_arcs[indx] )
                    is_degen = self.is_degenerated()
                    if is_degen:
                        non_degen -= 1
                        nb_arcs[indx], nb_arcs[non_degen] = nb_arcs[non_degen], nb_arcs[indx]
                accept = (saving > 0.0) or ( (not is_degen) \
                                        and math.log(np.random.rand()) < saving/temp )
                # Apply the move if accept and record new set of non-basic arcs
                if accept:
                    num_moves += 1
                    num_accepted += 1
                    self.remember_move()
                    self.do_move()
                    nb_arcs[indx] = self.get_leaving_arc()
                    non_degen = num_nb
                    cur_obj = self.get_obj_val()
                    if cur_obj < best_sol.tot_cost: 
                        improve = True
                        best_sol.over_write()
                    self.history.append( cur_obj )    
            acc_rate = num_accepted/sample_size
            if inform: self.give_info(iterat,temp,sample_size,acc_rate,cur_obj,best_sol.tot_cost) 
            num_fail += 1
            if improve : num_fail = 0
            # Set sample_size at next temperature level
            sample_size += int( max( sample_size*sample_growth, 1 ) ) 
            # Adjust the temperature 
            temp *= sa_beta
            # Stop if acceptance rate below minimum and no improved solution in recent iterations
            go_on = acc_rate > min_rate or num_fail < max_fail

        # Reset solution to best one found by procedure above and apply deterministic local search
        best_sol.make_basic( )
        self.local_search()
        self.solution.over_write( best_sol )
        
    #------------------------------------------------------------------------------
           
    def Osman_sa(self):    
        """
        Applies simulated annealing to the FCTP in a similar way as Osman (1995) 
        did for the generalised assignment problem, that is, by moving through the 
        neighbourhood in an implicitly given order and moving to the first accepted 
        neighbouring solution.
        """
        # First transfer the library's solution (which is thus assumed to exist)
        # to a local optimum 
        self.local_search()
    
        # Store this local optimum as best one found so far
        best_sol = FCTP.sol.solution()
        cur_obj  = self.get_obj_val()
        self.history = [cur_obj]
    
        # Initialise start and end temperature as largest and smallest deterioation 
        # in objective value observed when moving from the current point to a 
        # neighbouring solution
        nb_arcs = np.where(self.get_status()!=FCTP.BASIC)[0]
        num_nb  = nb_arcs.shape[0]
        #delta   = list( max(0.0,-self.get_cost_sav(arc=a)) for a in nb_arcs )
        delta   = list( -self.get_cost_sav(arc=a) for a in nb_arcs )
        Tstart  = max( delta )
        Tfinal  = max( min( delta ), 1.0 )
    
        # Initialise parameter beta of cooling schedule T'=T/(1+beta*T)
        Tcurr = Tbest = Treset = Tstart
        beta0 = (Tstart-Tfinal)/(Tstart*Tfinal)/(self.nnodes+1)
        
        # Say hello
        inform = FCTP.param.get(FCTP.param.screen) == FCTP.param.on;
        if inform: 
            self.give_info ("#Accepted","Temperature","Current_Obj","Incumbent",\
                            title="Osman style simulated annealing") 
            self.give_info( 0,Tcurr,cur_obj,cur_obj)
    
        num_reset = iterat = 0    
        max_iter  = FCTP.param.get(FCTP.param.max_iter)
        while num_reset < max_iter:
            # Sweep through neighbourhood. Each time a move is accepted, reduce temp. 
            # But reset temperature, if no move was accepted.
            ncheck  = num_nb
            num_acc = 0
            while ncheck > 0:
                arc_num = np.random.randint(ncheck)
                arc = nb_arcs[arc_num]
                saving = self.get_cost_sav( arc=arc )
                accept = ( saving > 0.0 ) or \
                         ( saving < 0.0 and math.log(np.random.rand()) < saving/Tcurr )
                if accept:   
                    self.remember_move()
                    self.do_move()
                    nb_arcs[arc_num] = self.get_leaving_arc()
                    cur_obj = self.get_obj_val()
                    self.history.append( cur_obj )
                    if cur_obj < best_sol.tot_cost:
                        best_sol.over_write()
                        Tbest = Tcurr
                        num_reset = 0
                    iterat += 1    
                    beta = beta0/(num_nb + math.sqrt(iterat)) 
                    Tcurr = Tcurr/(1.0+beta*Tcurr);
                    ncheck = num_nb;
                    num_acc +=1
                else:
                    ncheck -= 1
                    nb_arcs[arc_num], nb_arcs[ncheck] = nb_arcs[ncheck], arc
            # No move accepted -> Reanneal and stop if #reannealings reaches maximum
            Treset /= 2.0;
            if  Treset < Tbest: Treset = Tbest
            Tcurr = Treset;
            num_reset += 1
            if inform: self.give_info(num_acc,Tcurr,self.get_obj_val(),best_sol.tot_cost)
    
        # Reset libary's solution to best one found above
        best_sol.make_basic()
        self.solution.over_write( best_sol )
