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
import time

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

        err = self.local_search()
        self.history = [self.get_obj_val()]

        if err > 0: return err

        best_sol = FCTP.sol.solution()
        max_iter = FCTP.param.get(FCTP.param.max_iter)
        inform = FCTP.param.get(FCTP.param.screen) == FCTP.param.on
        if inform: self.give_info("Iter", "Before LS", "After LS", "Best_sol", title="Multi-start local search")
        for itr in range(max_iter):
            if self.greedy(0, 1) != 0: break
            before_LS = self.get_obj_val()
            self.local_search()
            after_LS = self.get_obj_val()
            if after_LS < best_sol.tot_cost: best_sol.over_write()
            if inform: self.give_info(itr, before_LS, after_LS, best_sol.tot_cost)
            self.history.append(after_LS)
        best_sol.make_basic()
        self.solution.over_write(best_sol)

    #------------------------------------------------------------------------------             

    def ils( self ):
        if FCTP.param.get(FCTP.param.ils_type)==FCTP.param.ils_standard:
            self.ils_standard()
        elif FCTP.param.get(FCTP.param.ils_type)==FCTP.param.ils_kstep:
            self.ils_k_step()
        else:
            raise NameError('Unknown ILS type, input 0 for Standard ILS and 1 for K-step')

    def kick_solution(self, num_exchanges=0):
        """
        Perturb a solution by sequentially introducing up to "num_exchanges"
        non-basic arcs into the basis.

        `Parameters:`
            num_exchanges : int  (optional)
                            number of non-basic arcs that the procedures
                            tries to put into the basis. If equal to zero,
                            the default, the procedures selects a random number
                            between 5 and 20% of the number of basic variables.
        """
        # Record indices of non-basic arcs.
        nb_arcs = np.where(self.get_status() != FCTP.BASIC)[0]

        # If number of exchanges unspecified, then randomly decide on it
        if num_exchanges == 0:
            num_basic = self.nnodes - 1
            num_exchanges = 3 if num_basic // 5 <= 5 else 5 + np.random.randint(num_basic // 5 - 4)

        # Apply "numexchanges" random basic exchanges.
        num_nb = nb_arcs.shape[0]
        for _ in range(num_exchanges):
            # Pick a non-basic arc at random and make it a basic arc
            in_arc = np.random.randint(num_nb)
            self.get_cost_sav(arc=nb_arcs[in_arc])
            self.remember_move()
            self.do_move()
            num_nb -= 1
            nb_arcs[in_arc] = nb_arcs[num_nb]

    def ils_standard(self):
        """
        Iterated Local Search applied to the FCTP.
        """
        # Check if instead of an ordinary local search a RTR search should
        # be used for improving perturbed solutions.
        do_RTR = FCTP.param.get(FCTP.param.improve_method) == FCTP.param.ils_rtr

        # Initialise parameter controlling when to reset the current solution
        beta = max(5, (self.nnodes - 1) // 10)

        #  Initialise iteration counters
        num_fail = 0;
        max_fail = FCTP.param.get(FCTP.param.max_no_imp)
        max_iter = FCTP.param.get(FCTP.param.max_iter)
        iterat = 0;

        # Display something on the screen, so that we can see that something happens
        do_info = FCTP.param.get(FCTP.param.screen)
        inform = do_info == FCTP.param.on
        if inform: self.give_info("Iter", "OBJ (before LS)", "OBJ (after LS)", \
                                  "BEST_OBJ", title="Iterated local search")

        # Save the initial solution as both the "current" and incumbent solution
        best_sol = FCTP.sol.solution()
        cur_sol = FCTP.sol.solution(best_sol)
        self.history = [cur_sol.tot_cost]

        # If RTR is applied as local search method switch of the screen and
        # reduce number of iterations for the RTR procedure
        if do_RTR:
            FCTP.param.set(FCTP.param.max_no_imp, 10)
            FCTP.param.set(FCTP.param.max_iter, 10)
            FCTP.param.set(FCTP.param.screen, FCTP.param.off)

        # Do the actual ILS:
        for _ in range(max_iter):
            iterat += 1
            # Improve solution using local search
            before_LS = self.get_obj_val()
            if do_RTR:
                self.rtr()
            else:
                self.local_search()
            after_LS = self.get_obj_val()
            accept = after_LS < cur_sol.tot_cost
            self.history.append(after_LS)
            # Check if new overall best solution has been detected
            num_fail += 1
            if after_LS < best_sol.tot_cost:
                best_sol.over_write()
                num_fail = 0;
            # Stop if max. number of failed subsequent iterations is reached
            if num_fail == max_fail: break
            # Display objective values after local search
            if inform: self.give_info(iterat, before_LS, after_LS, best_sol.tot_cost)
            # Every beta iterations, reset the "current" solution to the best one.
            if iterat % beta == 0:
                accept = False
                cur_sol.over_write(best_sol)
            # If solution is accepted, overwrite "current solution".
            # Otherwise, overwrite the actual solution with the "current solution".
            if accept:
                cur_sol.over_write()
            else:
                cur_sol.make_basic()

            # Apply a random kick to the Library's solution
            self.kick_solution()

        # ILS is finished. Set library's solution to best one found above
        best_sol.make_basic()
        self.solution.over_write(best_sol)

        # Reset iterations and screen parameter if changed
        if do_RTR:
            FCTP.param.set(FCTP.param.max_no_imp, max_fail)
            FCTP.param.set(FCTP.param.max_no_imp, max_iter)
            FCTP.param.set(FCTP.param.screen, do_info)

    def ils_k_step(self):
        self.local_search()
        self.history = [self.get_obj_val()]
        k_step = FCTP.param.get(FCTP.param.kstep)
        max_iter = FCTP.param.get(FCTP.param.max_iter)
        max_fail = FCTP.param.get(FCTP.param.max_no_imp)
        num_fail = 0

        reset = FCTP.param.get(FCTP.param.reset)
        max_before_reset = FCTP.param.get(FCTP.param.max_before_reset)
        num_no_improvement = 0

        if FCTP.param.get(FCTP.param.weight_func)=='linear':
            transform = lambda weights: weights
        elif FCTP.param.get(FCTP.param.weight_func)=='sqrt':
            transform = lambda  weights: np.sqrt(weights)
        elif FCTP.param.get(FCTP.param.weight_func)=='power':
            transform = lambda  weights: weights**2
        else:
            raise NameError('Weight Function not found, define weight function as linear, sqrt or power, got ' + FCTP.param.get(FCTP.param.weight_func))
        weight_func = lambda x: transform(x) / np.sum(transform(x))

        best_sol = FCTP.sol.solution()
        inform = FCTP.param.get(FCTP.param.screen) == FCTP.param.on
        if inform:
            self.give_info("Iter", "Before LS", "After LS", "Best_sol", title="Multi-start local search")
        for itr in range(max_iter):
            leaving_arcs = []
            for k in range(k_step):
                nb_arcs = np.where(self.get_status() != FCTP.BASIC)[0]  # Get edges to look at
                arcs_to_choose_from = list(set(nb_arcs) - set(leaving_arcs))
                savings = np.zeros_like(arcs_to_choose_from)
                i = 0
                for arc in arcs_to_choose_from:
                    saving = self.get_cost_sav(arc=arc)
                    savings[i] = saving
                    i += 1
                weights = savings-np.min(savings) # Make savings non-negative
                weights = weight_func(weights)
                choice = np.random.choice(arcs_to_choose_from, p=weights)
                self.get_cost_sav(arc=choice)
                leaving_arcs.append(self.get_leaving_arc())
                self.remember_move()
                self.do_move()
            before_LS = self.get_obj_val()
            self.local_search()
            after_LS = self.get_obj_val()
            num_fail += 1
            num_no_improvement += 1
            if after_LS < best_sol.tot_cost:
                num_no_improvement = 0
                num_fail = 0
                best_sol.over_write()
            if num_no_improvement >= max_before_reset and reset:
                num_fail = 0
                self.solution.over_write(best_sol)
            if inform:
                self.give_info(itr, before_LS, after_LS, best_sol.tot_cost)
            self.history.append(after_LS)
            if num_fail >= max_fail:
                break
        best_sol.make_basic()
        self.solution.over_write(best_sol)


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

    def rtr_travel(self, record):
        """
        Application of a single record to record travel that makes basic exchanges
        and either applies the first improving one or the best accepted one.
        Non-basic arcs are scanned by looping over suppliers and customers;
        best-moves for non-basic arcs adjacent to the current supplier node
        are thereby searched.

        `Parameters:`
            record : float
                best objective value found so far

        `Returns:`
            moved : boolean
                True if a move has been applied; otherwise False
        """
        supplier = np.random.permutation(self.m)
        customer = np.random.permutation(self.n)
        move_made = False
        deviat = FCTP.param.get(FCTP.param.rtr_percent) * record

        for s in supplier:
            best_sav = -np.inf
            for c in customer:
                if self.get_status(ij=(s, c)) != FCTP.BASIC:
                    saving = self.get_cost_sav(ij=(s, c))
                    if saving > best_sav:
                        best_sav = saving
                        self.remember_move()
                        if best_sav > 0.0: break
            # Apply first improving move of best acceptable non-improving
            # basic exchange involving non-basic arcs of supplier s
            if best_sav > 0.0 or self.get_obj_val() - best_sav < record + deviat:
                self.do_move()
                move_made = True
        return move_made

    # ------------------------------------------------------------------------------

    def rtr(self):
        """
        Applies a Record-to-Record travel to the FCTP.
        """
        do_info = FCTP.param.get(FCTP.param.screen)
        inform = do_info == FCTP.param.on
        if inform:
            self.give_info("Iter", "Before_LS", "After_LS", "Best_Objval", title="RTR procedure")

            # Check if instead of an ordinary local search an iterated search should
        # be used for improving start solutions provided by RTR_travel
        do_ILS = FCTP.param.get(FCTP.param.improve_method) == FCTP.param.rtr_ils

        # self.local_search() # Maybe, make first ordinary local search
        best_sol = FCTP.sol.solution()
        self.history = []

        # Main loop: Repeatedly do local search on an initial solution constructed by RTR
        iterat = 0
        num_fail = 0
        max_fail = FCTP.param.get(FCTP.param.max_no_imp)
        max_iter = FCTP.param.get(FCTP.param.max_iter)

        # If ILS is applied as local search method switch of the screen and
        # reduce number of iterations for the iterated local search
        if do_ILS:
            FCTP.param.set(FCTP.param.max_no_imp, 10)
            FCTP.param.set(FCTP.param.screen, FCTP.param.off)

        while num_fail < max_fail:
            iterat += 1
            # Apply up to max_iter single record-to-record travels
            for _ in range(max_iter):
                if not self.rtr_travel(best_sol.tot_cost):  break
            # Improve RTR solution using some local search method
            before_LS = self.get_obj_val()
            if do_ILS:
                self.ils()
            else:
                self.local_search()
            #  Check if new record is obtained
            num_fail += 1
            after_LS = self.get_obj_val()
            self.history.append(after_LS)
            if after_LS < best_sol.tot_cost:
                num_fail = 0
                best_sol.over_write()
            # Send some output to the screen
            if inform: self.give_info(iterat, before_LS, after_LS, best_sol.tot_cost)

        # Reset libary's solution to best one found above
        best_sol.make_basic()
        self.solution.over_write(best_sol)

        # Reset iterations and screen parameter if changed
        if do_ILS:
            FCTP.param.set(FCTP.param.max_no_imp, max_fail)
            FCTP.param.set(FCTP.param.screen, do_info)
