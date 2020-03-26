"""
  Tries exact solution to an instance of the FCTP by means of 
  CPLEX's MIP solver.

  author:
      Andreas Klose
  version:
      1.0 (01/01/20)    
"""
import cplex
import numpy as np
from cplex.callbacks import UserCutCallback
from . import api
from . import param

# Identifiers of different types of cuts used by Cplex
__cuttypes = [107,108,110,111,112,115,117,119,120,122,126,133,134]

#---------------------------------------------------------------------

def __setupFCTP ( d, s, f, t, c ):
    """
    Build the model, where d is array of demands, s array of supplies,
    f array of arc fixed cost, t array of unit arc transportation cost,
    c array of arc capacities.
    """
    model = cplex.Cplex() 
    n = len(d)
    m = len(s)
    narcs = m*n

    # Variables
    x = model.variables.add( obj=t )
    y = model.variables.add( obj=f, types=['B']*narcs )

    # Supply constraints
    lhs = [cplex.SparsePair(x[i*n:(i+1)*n], [1]*n) for i in range(m)]
    model.linear_constraints.add( lin_expr=lhs, senses=['E']*m, rhs=s )
    
    # Demand constraints           
    lhs = [cplex.SparsePair([x[i*n+j] for i in range(m)], [1]*m) for j in range(n)]
    model.linear_constraints.add(lin_expr=lhs,senses=['E']*n, rhs=d)

    # Variable upper bounds: x(ij) - y(ij)*capacity(i,j) <= 0
    lhs = [cplex.SparsePair([x[a],y[a]],[1,-c[a]]) for a in range(narcs)]       
    model.linear_constraints.add(lin_expr=lhs,senses=['L']*narcs, rhs=[0]*narcs)
    
    return model, x, y 
    
#---------------------------------------------------------------------

def SNFGomoryCut( D, c, x, y, yfrac ):
    """
    Derives (fractional) Gomory cuts for the single node flow model
    
               \sum_{i\in I} x_i = D, 0 <= x_i <= c_i*y_i. 

    Let S be a subset of I and let h \in S. A basis representation of
    a basic variable y_h is given by
    
    y_h + \sum_{i notin S} x_i/c_h  - \sum_{i in S\{h}} (c_i/c_h) (1-y_i)
     - \sum_{i in S} (c_i y_i - x_i)/c_h  = (c_h - lambda)/c_h =: alpha

    where \lambda = \sum_{i\in S} c_i - D.

    The intersection cut derived from this row is

    \sum_{i notin S} x_i/c_h - alpha/(1-alpha)\sum_{i in S\{h}} (c_i/c_h)(1-y_i) 
    - alpha/(1-alpha)\sum_{i in S} (c_i y_i - x_i)/c_h - alpha >= 0
    
    It can be shown that the cut above is equivalent to the "weak flow
    cover inequality"

    \sum_{i\in I\S} x_i >= (c_h - lambda)(1-y_h)

    or 

    \sum_{i\in I\S} x_i - (c_h - lambda)(1-y_h) >= 0

    or
 
    \sum_{i\in I\S} x_i - (D-\sum_{i\in S\{h}}c_i)(1-y_h) >= 0


    For each given h\in I, we then select the subset S such that
    the left hand side of the last inequality is as small as possible.
    Thereafter the fractional Gomory cut belonging to the basis representation
    of y_h is derived. If the cut is violated, its coefficients in the 
    variables (x,y) and the right-hand side are returned. Otherwise None is
    returned.
    
    
    `Parameters:`      
    
        D : float  
          demand of the demand node in the single-node flow model  
        c : numpy array of float     
          capacities of the arcs from the source nodes to the sink  
        x : numpy array of float    
          flows on the arcs from the source nodes to the sink  
        y : numpy array of float    
           (fractional) solution to the binary arc variables  
        yfrac : list of int    
            indices of fractional values in y      
    
    `Returns:`  
    
        lhs : list of float    
            Coefficients of flow  and binary variables. (None if no cut found)    
        rhs : float  
            right-hand side of >=-inequality. (None if no cut found)
                        
    """

    zero = 1.0E-5
    one  = 1.0-zero

    # Determine "best" index h
    smallest = 1.0E33
    n = len( c )
    for k in yfrac:
        tmp = [min(0.0, (1.0-y[k])*c[i]-x[i]) for i in range(n) if i != k]
        lhs = D*y[k]- x[k] + sum(tmp)
        if lhs < smallest: h, smallest = k, lhs

    # Given the index h above, the set S\{h} consists of every index i != h
    # for which (1-y_h)*c_i - x_i < 0
    in_S = [ (i!=h) and (x[i] > (1-y[h])*c[i]) for i in range(n) ]
    S_h = [ i for i in range(n) if in_S[i] ]
    in_S[h] = True

    # Set of nodes I \ S
    I_S = [ i for i in range(n) if not in_S[i] ]

    # Excess of capacity of set S over demand D
    excess = c[h] + sum( c[i] for i in S_h ) - D
    if excess < zero: return None, None
    alpha = (c[h]-excess)/c[h]
    if alpha <= zero: return None, None

    # Coefficient of flow variable x_i, i notin S, in the Gomory cut
    xcoeff = min(1.0/c[h], (1.0-1.0/c[h])*alpha/(1.0-alpha) )

    # Coefficients of binary variables (1-y_i), i from S but i!=h, in the cut
    fracs = np.modf([ c[i]/c[h] for i in S_h])[0]
    ycoeff = [ min( 1-f, f*alpha/(1.0-alpha) ) for f in fracs ]

    # Coefficients of slack variables w_i := c_i y_i - x_i in the cut
    wcoeff = min( 1.0-1.0/c[h], alpha/((1-alpha)*c[h]) )

    # Check if Gomory cut is violated
    lhs =  xcoeff*sum(x[i] for i in I_S)\
         + wcoeff * (sum(c[i]*y[i]-x[i] for i in S_h) + c[h]*y[h]-x[h]) \
         + sum( coef*(1-y[S_h[i]]) for i,coef in enumerate(ycoeff) )      
    if ( lhs/alpha > one ): return None, None

    # Collect the coefficients for the flow variables x
    xlhs = [xcoeff*(1-int(in_S[i]))-wcoeff*int(in_S[i]) for i in range(n) ]

    # Collect the coefficients for the binary variables y
    ylhs = [0.0]*n
    ylhs[h] = wcoeff*c[h]
    for num, i in enumerate(S_h): ylhs[i] = wcoeff*c[i] - ycoeff[num]

    # Rhs of the cut
    rhs = alpha - sum( ycoeff )

    return xlhs+ylhs, rhs

#---------------------------------------------------------------------

class __flowGomory( UserCutCallback ):
    """
    Include Gomory cuts derived from the single node flow model
    """
    def __call__(self):

        #if self.is_after_cut_loop() == False: return

        y = self.y
        x = self.x
        d = self.d
        s = self.s
        c = self.c
        m = len(s)
        n = len(d)
       
        """ 
        added = sum(self.get_num_cuts(typ) for typ in cuttypes)
        self.ncuts = added - self.ncuts
        if self.ncuts > 0: return
        """
        xv = self.get_values( x[0], x[-1] )
        yv = self.get_values( y[0], y[-1] )
        feas = self.get_feasibilities( y[0], y[-1] )

        for i in range(m):
            # Obtain set of fractional y[i][j] for given i
            yfrac = [j for j in range(n) if feas[i*n+j] > 0]
            if len(yfrac) > 0:
                lhs, rhs = SNFGomoryCut(s[i], c[i*n:(i+1)*n], xv[i*n:(i+1)*n],\
                               yv[i*n:(i+1)*n], yfrac)
                if not lhs is None:
                    xy = list(x[i*n:(i+1)*n])+list(y[i*n:(i+1)*n])
                    self.add(cplex.SparsePair(xy,lhs), sense='G', rhs=rhs )    
               
        for j in range(n):
            yfrac = [i for i in range(m) if feas[i*n+j] > 0]
            if len( yfrac ) > 0:
                xvj = [xv[i*n+j] for i in range(m)]
                yvj = [yv[i*n+j] for i in range(m)]
                cj = [c[i*n+j] for i in range(m)]
                lhs, rhs = SNFGomoryCut( d[j], cj, xvj, yvj, yfrac )
                if not lhs is None:
                    xj = [x[i*n+j] for i in range(m)]             
                    yj = [y[i*n+j] for i in range(m)] 
                    self.add(cplex.SparsePair(xj+yj,lhs), sense='G',rhs=rhs)

#---------------------------------------------------------------------

def __setStartSol ( model ):
    """
    Add solution stored in libFCTP as feasible initial solution to Cplex
    """
    x = api.get_Flows().astype(float)
    y = np.where( x > 0.0, 1.0, 0.0 )
    r = range( 0, model.variables.get_num() )
    model.MIP_starts.add( cplex.SparsePair( r, list(x)+list(y)),\
          model.MIP_starts.effort_level.no_check )

#---------------------------------------------------------------------

def cpxSolve( fctp ):
    """
    Uses CPLEX to solve instances of the CFLP. The necessary data
    are taken from the data stored in the library libFCTP
 
    Parameters :
    
        fctp : class FCTP.fctp
               the instance of class FCTP.fctp representing
               the current instance of the FCTP.
    """
    d = api.get_Demands().astype(float)
    s = api.get_Supplies().astype(float)
    c = api.get_Capacities().astype(float)
    f = api.get_Fcosts()
    t = api.get_Tcosts()
    model, x, y = __setupFCTP( d, s, f, t, c )

    timLim    = param.get(param.cpx_time)
    nodeLim   = param.get(param.cpx_nodelim)
    useLB     = param.get(param.cpx_localbranch)==param.yes
    aggress   = param.get(param.cpx_cut_aggress)==param.yes
    useDisCut = param.get(param.cpx_dis_cut)==param.yes
    useSNFGom = param.get(param.cpx_use_SNF_Gomory)==param.yes
            
    # Set node limit
    if nodeLim > 0: model.parameters.mip.limits.nodes.set( nodeLim )

    # Set time limit
    if timLim > 0: model.parameters.timelimit.set(timLim)

    # Activate local branching
    if useLB: model.parameters.mip.strategy.lbheur.set( 1 )

    # Maybe even emphasis=3 instead of 2
    model.parameters.emphasis.mip.set(2)

    model.parameters.mip.strategy.search.set( \
        model.parameters.mip.strategy.search.values.traditional)
   
    # Switch off "advanced" start 
    # model.parameters.advance.set(0)
    
    # With the following Rothberg's solution polishing (GA method) is invoked
    #model.parameters.mip.limits.nodes.set( 2 )
    #model.parameters.mip.polishafter.time.set(0)

    # Use cuts (very) aggressively
    if aggress:
        model.parameters.mip.cuts.covers.set(3)
        model.parameters.mip.cuts.gubcovers.set(2)
        model.parameters.mip.cuts.flowcovers.set(2)
        model.parameters.mip.cuts.mircut.set(2)
        model.parameters.mip.cuts.gomory.set(2)
        model.parameters.mip.cuts.implied.set(2)
        if useDisCut: model.parameters.mip.cuts.disjunctive.set(3)
        model.parameters.mip.cuts.liftproj.set(3)

    # Install the user cut callback for generating Gomory cuts from
    # the single node flow model
    if useSNFGom:
        model.register_callback( __flowGomory )
        __flowGomory.d = d
        __flowGomory.s = s
        __flowGomory.c = c
        __flowGomory.x = x
        __flowGomory.y = y
        __flowGomory.ncuts = 0
    
    __setStartSol( model )

    tstart = model.get_time()
    model.solve()
    fctp.cputime = model.get_time()-tstart 

    if model.solution.is_primal_feasible(): 
        flow = (np.array(model.solution.get_values(x[0],x[-1]))+0.1).astype(int)
        api.set_flows(flow)
        api.comp_cost()
        api.set_base()
        fctp.solution.over_write()
    
    fctp.lobnd = model.solution.MIP.get_best_objective()
    
    model.end()
