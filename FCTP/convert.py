"""
  Converts data files given in the format used by Amini, Roberti, Agarwal
  to the format required by the library "libFCTPy". 

  author:
      Andreas Klose
  version:
      1.0 (05/03/20)    
"""
import os.path 
import sys
import numpy as np

#------------------------------------------------------------------------------

def __writeDim( data, f_out ):
    """
    Retrieve number of suppliers (m) and customers (n) from "data" and
    write to output file.
    """
    m, n = int(data[0]), int(data[1])
    print(m,file=f_out,end=' ')
    print(n,file=f_out)
    return m,n
    
#------------------------------------------------------------------------------

def __write_node_data( data, f_out):
    """
    Write supply or demand vector to output file.
    """
    for num in data[:-1]: print(num, file=f_out, end=' ')
    print(data[-1], file=f_out )
    
#------------------------------------------------------------------------------

def __writeCosts( m, tcost, fcost, f_out ):
    """
    Write unit and fixed arc costs to output file f_out.
    """
    # Output unit costs
    for i in range(m):
        for t in tcost[i][0:-1]:  print( t, file=f_out, end=' ')
        print( tcost[i][-1],file=f_out )
    
    # Output fixed costs
    for i in range(m):
        for f in fcost[i][0:-1]:  print( f, file=f_out, end=' ')
        print( fcost[i][-1],file=f_out ) 

#------------------------------------------------------------------------------

def __convertAgarwal ( f_in, f_out ):
    """
    Converts data file "in_file" of Agarwal's format to libFCTPy format
    """
    data = f_in.read().split()
    m, n = __writeDim( data, f_out )
    __write_node_data( data[2:m+2], f_out ) 
    __write_node_data( data[m+2:m+n+2], f_out )
    fcost = np.array(data[m+n+2:m+n+2+m*n]).reshape(m,n)
    tcost = np.array(data[m+n+2+m*n:]).reshape(m,n)
    __writeCosts(m, tcost, fcost, f_out )
      
#------------------------------------------------------------------------------

def __convertGlover( f_in, f_out ):
    """
    Convert data file in Glover's format to libFCTPy's format
    """
    
    data = f_in.read().split('\n')
    line = data[1].split()
    m, n = __writeDim( [line[2],line[5]], f_out )
    
    # Read section S about the sources and output supplies
    for count, s in enumerate(data): 
        if s[0] == 'S': break
    for line in data[count+1:count+m]:
        s=line.replace('.','').split()[1]
        print(s,file=f_out,end=' ')
    s=data[count+m].replace('.','').split()[1]
    print(s,file=f_out)
    
    # Read section D about demands and output demands
    for count, d in enumerate(data): 
        if d[0] == 'D': break
    for line in data[count+1:count+n]:
        d=line.replace('.','').split()[1]
        print(d,file=f_out,end=' ')
    d=data[count+n].replace('.','').split()[1]
    print(d,file=f_out)
    
    # Retrieve fixed and unit arc costs
    tcost = np.zeros(m*n,dtype=float).reshape(m,n)
    fcost = np.zeros(m*n,dtype=float).reshape(m,n)
    for count, l in enumerate(data):
        if l=='ARCS': break
    narcs=m*n    
    for l in data[count+1:count+narcs+1]:
        ls = l.split()
        i,j = int(ls[0])-1,int(ls[1])-1-m
        tcost[i][j] = float( ls[2] )            
        fcost[i][j] = float( ls[3] )
    __writeCosts(m, tcost, fcost, f_out )    
    
#------------------------------------------------------------------------------

def __convertRoberti( f_in, f_out ):
    """
    Converts data file of Roberti et al.'s format to LibFCTPy's format.
    """
    data = [l for l in f_in.read().split('\n') if len(l)>0]
    m, n = __writeDim( data, f_out )
    __write_node_data( data[2:m+2], f_out )
    __write_node_data( data[m+2:m+n+2], f_out )
    # Retrieve fixed costs und unit cost
    tcost = np.zeros(m*n,dtype=float).reshape(m,n)
    fcost = np.zeros(m*n,dtype=float).reshape(m,n)    
    for l in data[m+n+2:]:
        ls = l.split()
        i,j = int(ls[0]),int(ls[1])
        tcost[i][j] = float( ls[2] )            
        fcost[i][j] = float( ls[3] )
    __writeCosts(m, tcost, fcost, f_out )    
        
#------------------------------------------------------------------------------        

def convert( in_file, in_form, out_file=None ):
    """
    Converts the data file "inf_file" to the format required by libFCTPy
    and writes the result either to the screen or to file "out_file".
    
    `Parameters:`  
        in_file : str 
            name (and path) of the input data file
        out_file: str
            if None (the default), output is send to the screen,
            otherwise to the file "out_file".
        in_form : str
            Format of the input data file. Format must be one of the
            following.
                Agarwal - input data format used by Agarwal and Aneja [1]
                Glover  - input data format used by Glover et al [2]
                Roberti - input data format used by Roberti et al [3]
                
            [1] Agarwal Y, Aneja Y (2012) Fixed-charge transportation problem:
                Facets of the projection polyhedron. Oper. Res. 60(3):638–654.
                
            [2] Glover F, Amini M, Kochenberger G (2005) Parametric ghost image
                processes for fixed-charge problems: A study of transportation 
                networks. J. Heuristics 11(4):307–336.
            [3] Roberti R, Bartolini E, Mingozzi A (2015) The Fixed Charge 
                Transportation Problem: An Exact Algorithm Based on a New
                Integer Programming Formulation. Mgmt Sci 61(5):1275–1291
                
    `Returns:`
         err : int
             err = 0 if everything went fine
             err = 1 if the input data file cannot be read
             err = 2 if the output data file cannot be written
             err = 3 if format is not one of the above                     
    """
    if not os.path.isfile(in_file): return( 1 )
    if not os.access(in_file,os.R_OK ): return( 1 )
    
    if not out_file is None:
        (folder,fname) = os.path.split(out_file)
        folderOk = len(folder)==0 or os.path.exists(folder)
        if not folderOk: return( 2 )
        if os.path.isfile(out_file):
            if not os.access(out_file, os.W_OK): return( 2 )
        if not os.access(folder,os.W_OK): return( 2 )     
            
    in_form = str.upper(in_form)
    if not in_form[0] in ['A','G','R']: return( 3 )
    
    f_in  = open(in_file,'r')
    f_out = sys.stdout if out_file is None else open(out_file,'w')
    
    if in_form == 'A': __convertAgarwal( f_in, f_out )
    if in_form == 'G': __convertGlover( f_in, f_out )
    if in_form == 'R': __convertRoberti( f_in, f_out )
     
    f_in.close( )
    if not out_file is None: f_out.close()

    return 0


# Check if works
#if __name__ == "__main__":
    #convert('/home/au220629/p/problems/fctp/Agarwal/Theta0.2/1.txt','A')
    #convert('/home/au220629/p/problems/fctp/Roberti/B20Theta0.0/30x30/1.dat','R')
    #convert('/home/au220629/Downloads/N104.DAT','G')
    
    