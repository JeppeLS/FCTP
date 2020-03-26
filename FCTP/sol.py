"""
    Class "FCTPy.sol" belonging to package FCTPy.
    
    Defines attributes and methods to treat solutions.
    
    author:
        Andreas Klose
    version:
        1.0 (01/01/20)
"""
import numpy as np
from . import api
from . import param

class solution:
    
    def __init__( self, source = None ):
        """
        Initializes an instance of a solution object either by
        copying a "source" solution or by copying the FCTP
        library's solution data to this object (if source=None).
        """
        self.tree_p = None
        self.tree_t = None
        self.arc_stat = None
        self.tot_cost = 0.0
        self.flow = None
        self.m = 0
        self.n = 0
        if source:
            # Copy "source" solution to this object
            self.m = source.m
            self.n = source.n
            self.tot_cost = source.tot_cost
            self.tree_p = source.tree_p.copy()
            self.tree_t = source.tree_t.copy()
            self.arc_stat = source.arc_stat.copy()
            self.flow = source.flow.copy()
        else:
            # Store the library's internal solution in this new object
            self.m = api.get_num_suppliers()
            self.n = api.get_num_customers() 
            self.tot_cost, self.flow, self.arc_stat, self.tree_p, self.tree_t =\
                api.get_basis_info()
            
    def over_write( self, source = None ):
        """
        Overwrites this solution with the data of a "source" solution or with
        the data from the library's solutions. 
        """
        if source:
            self.m = source.m
            self.n = source.n
            self.tot_cost = source.tot_cost
            self.tree_p[:] = source.tree_p[:]
            self.tree_t[:] = source.tree_t[:]
            self.arc_stat[:] = source.arc_stat[:]
            self.flow[:] = source.flow[:]
        else:
            api.get_basis_info( self ) 
                
    def compute_cost( self, fcost=None, tcost=None ):
        """
        Computes this solution's total cost if the fixed
        and unit cost assume the values passed to this method.
        If any of the cost is None, they are taken as
        currently stored in the library.
        
        `Parameters:`
        
            fcost : numpy array of float
                    fixed costs on the arcs
            tcost : numpy array of float
                    unit transportation costs on the arcs    
        """
        if fcost is None: fcost = api.get_Fcosts()
        if tcost is None: tcost = api.get_Tcosts()
        positive = np.where( self.flow > 0)[0]
        tot_fc = np.sum( fcost[positive] )
        self.tot_cost = tot_fc + np.dot( tcost, self.flow )

    def return_cost( self, fcost=None,  tcost=None ):
        """
        Essentially the same as **compute_cost** but instead
        of storing the computed total cost in the solution
        object's attribute **tot_cost** the computed cost
        are just returned.
        """
        if fcost is None: fcost = api.get_Fcosts()
        if tcost is None: tcost = api.get_Tcosts() 
        positive = np.where( self.flow > 0)[0]
        tot_fc = np.sum( fcost[positive] )
        return tot_fc + np.dot( tcost, self.flow )
 
    def get_flow_cost ( self, flow, fcost=None, tcost=None ):
        """
        Computes the total cost of the flow **flow**.
        
        `Parameters:`
        
            flow  : numpy array of int
                    the flow on the arcs
            fcost : numpy array of float (optional)
                    the fixed arc cost (if None, the fixed cost
                    stored by the library are used)
            tcost : numpy array of float (optional)
                    the unit transportation costs on the arcs 
                    (if None, the costs stored by the library are used)
                    
        `Returns:` The cost (float) of the flow
                                
        """
        if fcost is None: fcost = api.get_Fcosts()
        if tcost is None: tcost = api.get_Tcosts()
        positive = np.where( self.flow > 0)[0]
        tot_fc = np.sum( fcost[positive] )
        return tot_fc + np.dot( tcost, self.flow )
 
    def make_basic ( self ):
        """
        Overwrites the library's internal basic feasible solution
        with this (basic) solution. At least the flows in the
        solution need to be specified. If all components of the
        solution are specified, these data are used to set up
        the basis tree, otherwise the basis tree is recomputed.
        If this fails, a positive integer is returned. Otherwise
        zero is returned. 
        """
        if self.flow is None: return(7777)
        if self.tree_p is None or self.tree_t is None or\
           self.arc_stat is None:
            api.set_flows( self.flow )
            err = api.set_base() 
            if err > 0: return( err )
            api.get_basis_info( self )
            self.compute_cost()
        else:
            if self.tot_cost is None or self.tot_cost == 0.0: self.compute_cost()
            api.set_basis(self.tot_cost, self.flow, self.arc_stat, self.tree_p, self.tree_t)
   
    def print_flows( self, lobnd=None ):
        """
        Print the flow on each arc to the screen.
        """
        methName = param.get_proc_name(param.get(param.improve_method))
        print('-'*60)
        print("Solution obtained with",methName)
        print("Transportation quantities of total cost =",self.tot_cost )
        if not lobnd is None and lobnd > 0.0:
            print("Lower bound on optimal objective value  =",lobnd)    
        print('-'*60)            
        print("i -> j : Flow")
        print('-'*15)
        positive = np.where( self.flow > 0)[0]
        for a in positive:
            print( a//self.n,"->",a%self.n,":",self.flow[a] )
        print('-'*60)  

    def equalTo( self, other_sol ):
        """
        Returns true if this solution equal to the solution "other_sol".
        Two solutions are assumed to be identical if the flows on the
        arcs are all the same.
        """ 
        if self.flow.shape[0] != other_sol.flow.shape[0]: return ( False )
        return not np.any ( self.flow != other_sol.flow )

    def containedIn ( self, pool ):
        """
        Returns true if this solution object is equivalent (equalTo) any
        of the solutions in the list "pool" of solutions.
        """
        result = next( (s for s in pool if self.equalTo(s) ), None )
        return (not result is None)

    def distanceTo ( self, other_sol ):
        """
        Returns the distance (L1-norm) between the flow vector of this 
        solution and solution's other_sol flow vector.
        """
        if self.flow.shape[0] != other_sol.shape[0] : return 2147483647
        return np.linalg.norm( self.flow-other_sol.flow, ord=1 )
   
    def basDistTo( self, other_sol ):
        """
        Return the number of arcs showing a different status in this solution
        as in other_sol
        """
        if self.flow.shape[0] != other_sol.shape[0] : return 2147483647
        return np.sum( np.where(self.arc_stat != other_sol.arc_stat,1,0) )
