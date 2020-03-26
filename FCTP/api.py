"""
    Python interface to libFCTP - Basic procedures for the FCTP

    author:  
        Andreas Klose 
    version: 
        1.0 (01/01/20)    
        1.1 (27/02/20): Function is_degenerated() added,
                        call to library's Greedy adjusted,
                        c_int to int transformation corrected                         
"""
import numpy as np
import ctypes, pathlib, platform

CplexOk = True
__FCTP_Lib = None

# Some shortcuts of data types used within ctypes to interface C
__int = ctypes.c_int
__double = ctypes.c_double
__bool = ctypes.c_bool
__char = ctypes.c_char
__p_void = ctypes.c_void_p
__p_int = ctypes.POINTER( __int )
__p_double = ctypes.POINTER( __double )
__p_bool = ctypes.POINTER( __bool )
__p_char = ctypes.c_char_p

# Dictionary used for storing data belonging to a move (basic exchange)
__moveData = { "in_arc":0, "arc_stat":0, "out_arc":0, "to_upper":0, \
               "apex":0, "flow_chg":0, "i_path":0, "saving":0.0}

__stored_move = dict()

__BASIC = 1

#---------------------------------------------------------------------

def __loadLib():
    """ 
    Find and load the c-library libFCTP.
    """    

    FCTPlib = None

    onWindows = 'Windows' in platform.system()
    onMac = 'darwin' in platform.system().lower()
    FCTP_DLL = 'FCTPy.dll' if onWindows else 'libFCTPy.so'
   
    # The dynamic link library libFCTP.so (FCTP.dll on Windows)
    # is expected to reside in the same directory as this module.
    if onWindows:
        prefix = 'Lib//win//'
    elif onMac:
        prefix = 'Lib/mac/'
    else:
        prefix = 'Lib/linux/'
    FCTPhome = pathlib.Path(__file__).resolve().parent
    FCTPfile = pathlib.PurePath.joinpath(FCTPhome,prefix+FCTP_DLL)
    if FCTPfile.is_file():
        FCTPlib = str( FCTPfile )
    else:
        print("Could not find dynamic link library libFCTP/FCTP.")
        return( False )

    # Library found. Now try to load them
    global __FCTP_Lib
    
    __FCTP_Lib = ctypes.CDLL(FCTPlib, mode=ctypes.RTLD_GLOBAL)
    if __FCTP_Lib is None:
        print("Error occured when loading library ",FCTPlib )
        return( False )

    return( True )
    
#---------------------------------------------------------------------

"""
Obsolete here, but maybe useful at other places
def __pcint_to_nparray( cptr, arr_len ):
    
    Retrieves the content of the array to which the C integer pointer 
    cptr points to and returns it as a numpy array of int.

    Parameters:
        cptr : ctypes.POINTER( ctypes.c_int )
            C pointer to the array
        arr_len : int
            length of the array

    mem_buf = ctypes.pythonapi.PyBuffer_FromMemory
    mem_bub.restype = ctypes.py_object
    buf = buffer_from_memory( cptr, np.dtype(__int).itemsize*arr_len )
    return np.frombuffer( buf, int )
"""
#---------------------------------------------------------------------
"""
Obsolete here, but maybe useful at other places  
def __pcdouble_to_nparray( cptr, arr_len ):
    
    Retrieves the content of the array to which the C double pointer 
    cptr points to and returns it as a numpy array of float.

    Parameters:
        cptr : ctypes.POINTER( ctypes.c_double )
            C pointer to the array
        arr_len : int
            length of the array
    
    mem_buf = ctypes.pythonapi.PyBuffer_FromMemory
    mem_bub.restype = ctypes.py_object
    buf = buffer_from_memory( cptr, np.dtype(__double).itemsize*arr_len )
    return np.frombuffer( buf, float )
"""
#---------------------------------------------------------------------

def read_data( fname ):
    """
    Read the data of an FCTP instance from file named "fname".

    Parameters:
        fname : str
            name (including path if necessary) of data file
    Returns : 
        error_code : int
            error_code may attain the following values:                
            - 0 no error detected
            - 2 input file could not be opened
            - 3 an error occurred on reading number of supply/demand nodes
            - 4 nonpositive number of nodes
            - 5 insufficient memory
            - 6 error occurred on reading supply values
            - 7 error occurred on reading demand values
            - 8 error occurred on reading unit transportation cost values
            - 9 error occurred on reading fixcost values
    """
    fn = fname.encode('utf-8')
    __FCTP_Lib.FCTPreaddat.argtypes = [__p_char]
    __FCTP_Lib.FCTPreaddat.restype = __int
    status = __FCTP_Lib.FCTPreaddat( fn )
    return( status )

#---------------------------------------------------------------------

def reset_data( m, n, s, d, c, fc, tc ):
    """
    Passes data of a new FCTP instance to the library. Can also be
    used to pass the first instance to the library instead of using
    read_data.

    Parameters:
        m : int
            number of suppliers
        n : int
            number of customers
        s : numpy array of int
            array of supply amounts
        d : numpy array of int
            array of customer demands
        c : numpy array of int 
            array of arc capacities. Can be NULL so that
            min{ s[i], d[j]} is the flow limit on arc i->j
        fc: numpy array of float
            fixed cost of each arc 
        tc: numpy array of float
            unit transp. cost on each arc  

    Returns:
       err: int
           0 if no error; otherwise some postive integer.                              
    """
    __FCTP_Lib.FCTPresetData.restype = __int
    s_np = s.astype(__int)
    d_np = d.astype(__int)
    c_np = c.astype(__int)
    f_np = fc.astype(__double)
    t_np = tc.astype(__double)
    p_s = s_np.ctypes.data_as(ctypes.POINTER(__int ))
    p_d = d_np.ctypes.data_as(ctypes.POINTER(__int ))
    if c_np:
        p_c = c_np.ctypes.data_as(ctypes.POINTER(__int ))
    else:
        p_c = __p_void(None)     
    p_f = f_np.ctypes.data_as(ctypes.POINTER(__double))
    p_t = t_np.ctypes.data_as(ctypes.POINTER(__double))
    __FCTP_Lib.FCTPresetData.restype = __int
    err = __FCTP_Lib.FCTPresetData(__int(m), __int(n), p_s, p_d, p_c, p_f, p_t  )
    return err
#---------------------------------------------------------------------

def set_start_time( ):
    """
    Set a time stamp for measuring computation time.
    """
    __FCTP_Lib.FCTPsetStarttime()

#---------------------------------------------------------------------

def get_CPU_time ():
    """
    Returns CPU time passed since last call to setStartTime.
    """
    __FCTP_Lib.FCTPgetCPUtime.restype = __double
    return __FCTP_Lib.FCTPgetCPUtime() 

#---------------------------------------------------------------------

def get_num_suppliers ():
    """
    Returns number of supply nodes in the current FCTP instance
    """
    __FCTP_Lib.FCTPgetnumsupplier.restype = __int
    return __FCTP_Lib.FCTPgetnumsupplier()

#---------------------------------------------------------------------

def get_num_customers ():
    """
    Returns number of customer nodes in current FCTP instance
    """
    __FCTP_Lib.FCTPgetnumcustomer.restype = __int
    return __FCTP_Lib.FCTPgetnumcustomer()

#---------------------------------------------------------------------

def get_supply( i ):
    """
    Returns the supply (as int) of supplier i.
    """
    __FCTP_Lib.FCTPgetsupply.restype = __int
    return __FCTP_Lib.FCTPgetsupply( __int(i) )

#---------------------------------------------------------------------

def get_Supplies( ):
    """
    Returns the array of all supply amounts (as numpy array of int)
    """
    s_c = np.zeros(get_num_suppliers(), dtype=__int)
    ps_c = s_c.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPgetSupplies(ps_c)
    return s_c.astype(int)

#---------------------------------------------------------------------

def get_demand( j ):
    """
    Returns customer j's demand (as int).
    """
    __FCTP_Lib.FCTPgetdemand.restype = __int
    return __FCTP_Lib.FCTPgetdemand( __int(j) )

#---------------------------------------------------------------------

def get_Demands():
    """
    Returns array of all customer demands (as numpy array of int)
    """
    d_c = np.zeros(get_num_customers(), dtype=__int)
    pd_c = d_c.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPgetDemands(pd_c)
    return d_c.astype(int)

#---------------------------------------------------------------------

def get_cap( i,j ):
    """
    Returns capacity (as int) of the arc from supplier i to customer j.
    """
    __FCTP_Lib.FCTPgetcap.restype = __int
    return __FCTP_Lib.FCTPgetcap(__int(i), __int(j) )

#---------------------------------------------------------------------

def get_arc_cap( arc ):
    """
    Return arc's "arc" capacity (as int).
    """
    __FCTP_Lib.FCTPgetacap.restype = __int
    return __FCTP_Lib.FCTPgetacap( __int(arc) )

#---------------------------------------------------------------------

def get_Capacities():
    """
    Returns whole array of arc capacities (as numpy array of int)
    """
    narcs = get_num_customers()*get_num_suppliers()
    c_c = np.zeros( narcs, dtype=__int)
    pc_c = c_c.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPgetCaps(pc_c)
    return c_c.astype(int)
    
#---------------------------------------------------------------------

def get_fcost( i, j ):
    """
    Return fixed cost (as float) on arc from supplier i to customer j
    """
    __FCTP_Lib.FCTPgetfcost.restype = __double
    fij =__FCTP_Lib.FCTPgetfcost( __int(i), __int(j) )
    return( float(fij) )

#---------------------------------------------------------------------

def get_arc_fcost( arc ):
    """
    Return fixed cost (as float) on arc "arc".
    """
    __FCTP_Lib.FCTPgetafcost.restype = __double
    return __FCTP_Lib.FCTPgetafcost( __int(arc) )

#---------------------------------------------------------------------

def get_Fcosts():
    """
    Returns array of all fixed costs (as numpy array of float)
    """
    narcs = get_num_customers()*get_num_suppliers()
    f_c = np.zeros( narcs, dtype=__double)
    pf_c = f_c.ctypes.data_as( ctypes.POINTER( __double ) )
    __FCTP_Lib.FCTPgetFcosts(pf_c)
    return f_c

#---------------------------------------------------------------------

def get_tcost( i, j ):
    """
    Return unit transportation cost (as float) on arc i->j
    """
    __FCTP_Lib.FCTPgettcost.restype = __double
    return __FCTP_Lib.FCTPgettcost( __int(i), __int(j) )

#---------------------------------------------------------------------

def get_arc_tcost( arc ):
    """
    Return unit transportation cost (as float) on arc "arc".
    """
    __FCTP_Lib.FCTPgetatcost.restype = __double
    return __FCTP_Lib.FCTPgetatcost( __int(arc) )

#---------------------------------------------------------------------

def get_Tcosts():
    """
    Returns array of all unit transp. costs (as numpy array of float)
    """
    narcs = get_num_customers()*get_num_suppliers()
    t_c = np.zeros( narcs, dtype=__double)
    pt_c = t_c.ctypes.data_as( ctypes.POINTER( __double ) )
    __FCTP_Lib.FCTPgetTcosts(pt_c)
    return t_c

#---------------------------------------------------------------------

def get_obj_val():
    """
    Return object value of current solution
    """
    __FCTP_Lib.FCTPgetobjval.restype = __double
    return __FCTP_Lib.FCTPgetobjval() 

#---------------------------------------------------------------------

def get_flow( i, j ):
    """
    Return flow (as int) on arc i-> in current solution.
    """
    __FCTP_Lib.FCTPgetflow.restype = __int
    return __FCTP_Lib.FCTPgetflow( __int(i),  __int(j) )

#---------------------------------------------------------------------

def get_arc_flow( arc ):
    """
    Return flow (as int) on arc "arc" in current solution
    """
    __FCTP_Lib.FCTPgetaflow.restype = __int
    return __FCTP_Lib.FCTPgetaflow( __int(arc) )

#---------------------------------------------------------------------

def get_Flows():
    """
    Return array of flows (as numpy array of int) in current solution.
    """
    narcs = get_num_customers()*get_num_suppliers()
    f_c = np.zeros( narcs, dtype=__int)
    pf_c = f_c.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPgetFlows(pf_c)
    return f_c.astype(int)

#---------------------------------------------------------------------

def get_basis_info( solution=None ):
    """
    Retrieves the data used for describing the current basic solution
    to the FCTP. The data are either returned by this function 
    or, if solution is not None, stored in the object solution
    (see class solution in module sol.py).
    
    Returns (if solution=None):
        objval : float
            objective value of the basic solution
        flows : numpy array of int
            the flows on the arcs in the basic solution
        arc_stat : numpy array of int
            the array of status of each arc (basic or not)
        tree_p : numpy array of int
            the predecessors of each node in the basis tree
        tree_t : numpy arra of int
            the depth of the subtree rooted at each node
    """

    n = get_num_customers()
    m = get_num_suppliers()
    narcs = n*m  
    
    flows = np.zeros( narcs, dtype = __int )
    pflows = flows.ctypes.data_as(ctypes.POINTER(__int ))
    arc_stat = np.zeros( narcs, dtype=__int)
    parc_stat = arc_stat.ctypes.data_as(ctypes.POINTER(__int ))
    tree_p = np.zeros( narcs, dtype = __int )
    ptree_p = tree_p.ctypes.data_as(ctypes.POINTER(__int ))
    tree_t = np.zeros( narcs, dtype = __int )
    ptree_t = tree_t.ctypes.data_as(ctypes.POINTER(__int ))

    __FCTP_Lib.FCTPgetbasis.restype = __double
    objval = __FCTP_Lib.FCTPgetbasis( pflows, parc_stat, ptree_p, ptree_t );
    
    if not solution is None:
        solution.m = m
        solution.n = n
        solution.tot_cost = objval
        solution.arc_stat[:] = arc_stat[:]
        solution.tree_p[:] = tree_p[:]
        solution.tree_t[:] = tree_t[:]
        solution.flow[:] = flows[:]
    else:    
        return objval,flows.astype(int),arc_stat.astype(int),tree_p.astype(int),\
               tree_t.astype(int)  
  
#---------------------------------------------------------------------

def set_supply( i, s ):
    """
    Set supplier i's supply to the value s.
    """
    __FCTP_Lib.FCTPsetsupply( __int(i), __int(s) )


#---------------------------------------------------------------------

def set_supplies( s ):
    """
    Set the supply vector to the supplies given by the numpy array s
    of int.
    """
    s32 = s.astype( __int )
    ps32 = s32.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPsetSupplies( ps32 )
    
#---------------------------------------------------------------------

def set_demand( j, d ):
    """
    Set customer j's demand to the value d
    """
    __FCTP_Lib.FCTPsetdemand( __int(j), __int(d) )
    
#---------------------------------------------------------------------

def set_demands( d ):
    """
    Set the demand vector to the demands given by the numpy array d
    of int.
    """
    d32 = d.astype( __int )
    pd32 = d32.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPsetDemands( pd32 )

#---------------------------------------------------------------------

def set_cap( i, j, c ):
    """
    Set capacity of arc from supplier i to customer j to the value c
    """
    __FCTP_Lib.FCTPsetcap( __int(i), __int(j), __int( c ) )

#---------------------------------------------------------------------

def set_arc_cap( arc, c ):
    """
    Set capacity of arc "arc" to the value c
    """
    __FCTP_Lib.FCTPsetacap( __int(arc), __int( c ) )

#---------------------------------------------------------------------

def set_capacities( c ):
    """
    Set all arc capacities to the numpy array cap of int.
    """
    c32 = c.astype( __int )
    pc32 = c32.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPsetCaps( pc32 )

#---------------------------------------------------------------------

def set_fcost( i, j, f ):
    """
    Set fixed cost on arc from supplier i to customer j to float f
    """
    __FCTP_Lib.FCTPsetfcost( __int(i), __int(j), __double(f))

#---------------------------------------------------------------------

def set_arc_fcost( arc, f ):
    """
    Set fixed cost on arc "arc" to float value f
    """
    __FCTP_Lib.FCTPsetafcost( __int(arc),  __double(f) )

#---------------------------------------------------------------------

def set_fcosts( f ):
    """
    Set arc fixed cost to values given by the numpy array f of float.
    """
    pf = f.ctypes.data_as( ctypes.POINTER( __double ) )
    __FCTP_Lib.FCTPsetFcosts( pf )

#---------------------------------------------------------------------

def set_tcost( i, j, tc ):
    """
    Set unit cost on arc from supplier i to customer j to float tc
    """
    __FCTP_Lib.FCTPsettcost( __int(i), __int(j), __double(tc))

#---------------------------------------------------------------------

def set_arc_tcost( arc, tc ):
    """
    Set unit cost on arc "arc" to float value tc
    """
    __FCTP_Lib.FCTPsetatcost( __int(arc),  __double(tc) )

#---------------------------------------------------------------------

def set_tcosts( tc ):
    """
    Set arc unit costs to values given by the numpy array tc of float.
    """
    ptc = tc.ctypes.data_as( ctypes.POINTER( __double ) )
    __FCTP_Lib.FCTPsetTcosts( ptc )

#---------------------------------------------------------------------

def set_basis ( objval, flows, arc_stat, tree_p, tree_t ):
    """
    Overwrites the internally stored basic solution with the values
    given by the parameters above.
    
    Parameters:
        objval : float
            objective value of the basic solution
        flows : numpy array of int
            the flows on the arcs in the basic solution
        arc_stat : numpy array of int
            the array of status of each arc (basic or not)
        tree_p : numpy array of int
            the predecessors of each node in the basis tree
        tree_t : numpy arra of int
            the depth of the subtree rooted at each node
    """
    flo32 = flows.astype( __int )
    stat32 = arc_stat.astype( __int )
    pred32 = tree_p.astype( __int )
    tsize32 = tree_t.astype( __int )
    
    objv = __double( objval )
    p_flo = flo32.ctypes.data_as( ctypes.POINTER( __int ) )
    p_stat = stat32.ctypes.data_as( ctypes.POINTER( __int ) )
    p_pred = pred32.ctypes.data_as( ctypes.POINTER( __int ) )
    p_tsize = tsize32.ctypes.data_as( ctypes.POINTER( __int ) )
    
    __FCTP_Lib.FCTPflashbasis( objv, p_flo, p_stat, p_pred, p_tsize )

#---------------------------------------------------------------------

def FCTPclose( ):
    """
    Free the memory allocated by the library. 
    """
    __FCTP_Lib.FCTPclose()

#---------------------------------------------------------------------

def greedy( what_meas ):
    """
    Greedily constructs a solution for FCTP.
  
    Parameters:
        what_meas: int 
            specifies how to evaluate arcs:
            0 : costs per unit with fixed cost linearized by arc capacity
            1 : costs per unit with fixed cost linearized by remaining arc capacity
            2 : total cost of supplying the remaining quantity on an arc
    """            
    alpha = 0.0
    __FCTP_Lib.FCTPgreedy( __int(what_meas), __double(alpha) )

#---------------------------------------------------------------------

def local_search( createBasTree=False ):
    """
    Invokes the library's local search procedure to be applied on the
    internally stored current basic feasible solution. 
    If createBasTree is true, then the basis tree is first setup 
    Otherwise, it is assumed that this has already been done (the default)!
    If the procedure fails to "createBasTree" then a positive error code
    is returned; otherwise 0.
    """
    __FCTP_Lib.FCTPls.restype = __int
    return __FCTP_Lib.FCTPls( __bool(createBasTree) )
 
#---------------------------------------------------------------------

def grasp( what_meas, max_iter, alpha ):
    """
    Applies a Grasp procedure to the FCTP. See FCTPgreedy regarding the meaning
    of the parameters "what_meas" and "alpha". The GRASP procedure stops if
    after max_iter subsequent iterations the best solution found so far is not
    improved. If an error occurs (when trying to set a solution as basis),
    a positive error code is returned; otherwise 0.  
    """
    __FCTP_Lib.FCTPgrasp.restype = __int
    return __FCTP_Lib.FCTPgrasp( __int(what_meas), __int(max_iter), __double(alpha))
    
#---------------------------------------------------------------------

def abhc( ):
    """
    Applies an "attribute based hill climber heuristic to the current
    basic feasible solution. An positive error code is returned if
    getting the basis tree of this solution fails; otherwisee 0 is 
    returned.
    """
    __FCTP_Lib.FCTPabhc.restype = __int
    return __FCTP_Lib.FCTPabhc() 

#---------------------------------------------------------------------

def get_status( i, j ):
    """
    Obtain the status of the arc i->j in the current basic solution
    """
    __FCTP_Lib.FCTPgetarcstat.restype = __int
    return __FCTP_Lib.FCTPgetarcstat( __int(i), __int(j) ) 

#---------------------------------------------------------------------

def get_arc_status( arc ):
    """
    Obtain the status of arc "arc" in the current basic solution
    """
    __FCTP_Lib.FCTPgetarcstat.restype = __int
    return __FCTP_Lib.FCTPgetarcastat( __int(arc) ) 

#---------------------------------------------------------------------

def get_arc_stats( ):
    """
    Returns an numpy int array specifying the status of each arc in
    the current basic solution
    """
    narcs = get_num_customers()*get_num_suppliers()
    stats  = np.zeros( narcs, dtype=__int)
    pstats = stats.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPgetArcstats(pstats)
    return stats.astype(int)

#---------------------------------------------------------------------

def set_flow( i, j, flow ):
    """
    Set the flow on arc i->j to the int value flow
    """    
    __FCTP_Lib.FCTPsetflow( __int(i), __int(j), __int(flow) )

#---------------------------------------------------------------------

def set_arc_flow( arc, flow ):
    """
    Set the flow on arc "arc" to the int value flow
    """    
    __FCTP_Lib.FCTPsetaflow( __int(arc), __int(flow) )

#---------------------------------------------------------------------

def set_flows( flows ):
    """
    Set the flows on the arcs to the values specified in the numpy
    int array "flows"
    """
    flo32 = flows.astype(__int )
    pflo = flo32.ctypes.data_as( ctypes.POINTER( __int ) )
    __FCTP_Lib.FCTPsetFlows( pflo )
    
#---------------------------------------------------------------------

def comp_cost():
    """
    (Re-)compute the cost of the current basic feasible solution
    """
    __FCTP_Lib.FCTPcompCost.restype = __double
    return __FCTP_Lib.FCTPcompCost( )

#---------------------------------------------------------------------

def set_base():
    """
    Build the basis tree corresponding to the current solution.
    Return 0 if successful; otherwise a positive error code
    """
    __FCTP_Lib.FCTPsetbase.restype = __int
    return __FCTP_Lib.FCTPsetbase()

#---------------------------------------------------------------------

def get_cost_sav( i, j):
    """
    Compute and return the cost savings that can be achieved by
    introducing the arc i->j into the basis. 
    """
    
    __moveData["arc_stat"] = get_status( i, j);
    if __moveData["arc_stat"] == __BASIC: return( 0.0 )
    
    in_arc = __int( i*get_num_customers() + j )
    apex = __int(0)
    out_arc = __int( 0 )
    flow_chg = __int( 0 )
    i_path = __int(0)
    to_upper = __int(0)
    ref = ctypes.byref
    
    __FCTP_Lib.FCTPgetcstsav.restype = __double
    cstsav = __FCTP_Lib.FCTPgetcstsav( in_arc, ref(apex), ref(out_arc),\
                ref(flow_chg), ref(i_path), ref(to_upper ) )
    
    __moveData["in_arc"] = in_arc 
    __moveData["apex"] = apex
    __moveData["out_arc"] = out_arc
    __moveData["flow_chg"] = flow_chg
    __moveData["i_path"] = i_path
    __moveData["to_upper"] = to_upper
    __moveData["saving"] = __double(cstsav)
                                       
    return cstsav
 
#---------------------------------------------------------------------

def get_arc_cost_sav( arc ):
    """
    Compute and return the cost savings that can be achieved by
    introducing the arc "arc" into the basis. 
    """
    __moveData["arc_stat"] = get_arc_status( arc );
    if __moveData["arc_stat"] == __BASIC: return( 0.0 )
    
    in_arc = __int( arc )
    apex = __int(0)
    out_arc = __int( 0 )
    flow_chg = __int( 0 )
    i_path = __int(0)
    to_upper = __int(0)
    ref = ctypes.byref
    
    __FCTP_Lib.FCTPgetcstsav.restype = __double
    cstsav = __FCTP_Lib.FCTPgetcstsav( in_arc, ref(apex), ref(out_arc),\
                ref(flow_chg), ref(i_path), ref(to_upper ) )
    
    __moveData["in_arc"] = in_arc
    __moveData["apex"] = apex
    __moveData["out_arc"] = out_arc
    __moveData["flow_chg"] = flow_chg
    __moveData["i_path"] = i_path
    __moveData["to_upper"] = to_upper
    __moveData["saving"] = __double(cstsav)
                                       
    return cstsav

#---------------------------------------------------------------------

def get_leaving_arcN1( ):
    """
    Return the index of the supplier belonging to the arc leaving the basis 
    """
    out_arc = __moveData["out_arc"].value
    arc =  __moveData["in_arc"].value if out_arc < 0 else out_arc
    return ( int(arc) // get_num_customers() ) 
 
#---------------------------------------------------------------------

def get_leaving_arcN2( ):
    """
    Return the index of the customer belonging to the arc leaving the basis 
    """
    out_arc = __moveData["out_arc"].value
    arc =  __moveData["in_arc"].value if out_arc < 0 else out_arc
    return ( int(arc) % get_num_customers() ) 

#---------------------------------------------------------------------

def get_leaving_arc( ):
    """
    Return the index of the arc leaving the basis 
    """
    out_arc = __moveData["out_arc"].value
    arc =  __moveData["in_arc"].value if out_arc < 0 else out_arc
    return ( int(arc) ) 

#---------------------------------------------------------------------

def remember_move():
    """
    Store the data of a basic exchange (move) just investigated
    in order to later possibly apply that move.
    """
    global __stored_move
    __stored_move = __moveData.copy()

#---------------------------------------------------------------------

def is_degenerated():
    """
    Returns true if a tentative basic exchange that just was
    evaluated using function get_cost_sav() is degenerate.
    Note tht this function may only be called after get_cost_sav
    was called before.
    """
    return __moveData["flow_chg"].value == 0

#---------------------------------------------------------------------

def do_move():
    """
    Apply a move remembered before
    """
    if __stored_move["arc_stat"] != __BASIC:
        in_arc = __stored_move["in_arc"]
        out_arc = __stored_move["out_arc"]
        to_upper = __stored_move["to_upper"]
        apex = __stored_move["apex"]
        flow_chg = __stored_move["flow_chg"]
        i_path = __stored_move["i_path"]
        saving = __stored_move["saving"]
        __FCTP_Lib.FCTPdomove( in_arc, out_arc, to_upper, apex, flow_chg, i_path, saving )

#---------------------------------------------------------------------

def set_freq_pen( penalty, additive ):
    """
    Set the frequence penalty
    """
    __FCTP_Lib.FCTPSetFreqPen( __double(penalty), __int(additive) )

#---------------------------------------------------------------------

def reset_freq_pen( penalty, additive ):
    """
    Set the frequence penalty
    """
    __FCTP_Lib.FCTPResetFreqPen( __double(penalty), __int(additive) )

#---------------------------------------------------------------------

def clear_freq_pen():
    """
    Clear frequncy penalty
    """
    __FCTP_Lib.FCTPClearFreqPen()

#---------------------------------------------------------------------

is_ready = __loadLib()

# Check if Cplex package can be imported
try:
    import cplex
except ImportError:
    CplexOk=False    