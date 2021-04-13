# HW7 Tony Yuan

import globals
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
from scipy.linalg import lu
from numpy import array
import numpy as np
from math import *
#import geodesic
#import geodesicDense
import matplotlib.pyplot as plt
from time import time
import timeit
from scipy.sparse.csc import csc_matrix
from objg import objg
from woods import woods
from indef import indef
from cragglvy import cragglvy
#__________________________________________________________________

def StepSize(fun, x, d, alfa=1, params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}):
    global status
                                        
    alpha_L=0
    alpha_R=float('inf') 
    iter=0
    while True:
        iter+=1
        func_value_old=fun(x['p'],mode=1)[0]
        grad_value_old=array(fun(x['p'],mode=2))
        globals.numf+=1
        globals.numg+=1
        point_new=x['p']+alfa*d
        func_value_new=fun(point_new,mode=1)[0]
        grad_value_new=array(fun(point_new,mode=2))
        if func_value_new >func_value_old+params['ftol'] *alfa *np.dot(grad_value_old,d):
            alpha_R=alfa
        elif np.dot(grad_value_new,d) >=params['gtol']*np.dot(grad_value_old,d):
            alpha_star=alfa
            break
        else:
            alpha_L=alfa
        if alpha_R==float('inf') :
            alfa=2*alpha_L
        else:
            alfa=(alpha_L+alpha_R)/2
        if abs(alpha_L-alpha_R)<params['xtol']:
            alpha_star=alpha_L
            status=0
            print('invalid step size,tolerence for alpha reached')
            break
        elif iter>params['maxit']:
            alpha_star=alpha_L
            status=0
            print('invalid step size,maximum ierations for alpha reached')
            break
    status=1
    x['p']=point_new
    x['f']=func_value_new
    x['g']=grad_value_new[0]
    x['alpha']=alpha_star
    step_size={'iter':iter, 'status':status, 'x':x}
    return step_size
#_______________________________________________________________________________
def SteepDescent(fun,x,sdparams = {'maxit': 1000,'toler': 1.0e-4}):
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    d=-x['g']
    alfa=1
    iteration=0
    grad_seq=[x['g']]
    while np.linalg.norm(x['g'])>=sdparams['toler'] and iteration<sdparams['maxit'] :
        iteration +=1
        x0=x.copy()
        params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-8,'maxit': 100}
        step_size=StepSize(fun, x, d, alfa, params)
        x=step_size['x']
        d=-x['g']
        alfa=x.pop('alpha')
        alfa = max(10 * params['xtol'], alfa* np.linalg.norm(x0['g'])**2/np.linalg.norm(x['g'])**2)
        grad_seq.append(x['g'])
        if step_size['status']==0:
            break
    x['grad_seq']=grad_seq
    solution=x
    if np.linalg.norm(x['g'])<sdparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result


#_________________________________________________________________________________-


def Newton(fun,x,nparams = {'maxit': 1000,'toler': 1.0e-4, 'method': 'direct'}):
    m=len(x['p'])
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    x['H']=fun(x['p'],mode=4)[0]
    globals.numf+=1
    globals.numg+=1
    g=x['g']
    H=x['H']
    alfa=1
    iteration=0
    grad_seq=[x['g']]
    if nparams ['method']== 'direct':
        d= -np.linalg.solve(H,g)  
        if np.dot(-g,d)<0:
            Diagonal=np.zeros(m)
            for i in range(m):
                Diagonal[i]=1/max (abs(H[i][i]),0.01)
                D=np.diag(Diagonal)
                d=-np.dot(D,g)
        while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit']:
            iteration +=1
            x0=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 100}
            
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            x['H']=fun(x['p'],mode=4)[0]
            globals.numH+=1
            d= -np.linalg.solve(H,g)  
            globals.numFact+=1
            if np.dot(-g,d)<0:
                Diagonal=np.zeros(m)
                for i in range(m):
                    Diagonal[i]=1/max (abs(H[i][i]),0.01)
                    D=np.diag(Diagonal)
                    d=-np.dot(D,g)
            alfa=x.pop('alpha')
            alfa = 1
            grad_seq.append(x['g'])
            if step_size['status']==0:
                break
        globals.numFact+=1
    #___________________________________________________________________________________
    elif nparams ['method']== 'spdirect':
        LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
        d=-LU.solve(g) 
        if np.dot(-g,d)<0:
            Diagonal=np.zeros(m)
            for i in range(m):
                Diagonal[i]=1/max (abs(H[i][i]),0.01)
                D=np.diag(Diagonal)
                d=-np.dot(D,g)
        while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit']:
            iteration +=1
            x0=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 100}
            
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            x['H']=fun(x['p'],mode=4)[0]
            globals.numH+=1
            LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
            d=-LU.solve(g)  
            globals.numFact+=1
            if np.dot(-g,d)<0:
                D=np.diag(1/np.maximum(abs(H.diagonal()),0.01*np.ones(m)))
                d=-np.dot(D,g)
            alfa=x.pop('alpha')
            alfa = 1
            grad_seq.append(x['g'])
            if step_size['status']==0:
                break
        globals.numFact+=1

    #________________________________________________________________________
    elif nparams ['method']== 'pert':
        d= -np.linalg.solve(H,g)
        
        multiple=0
        while np.dot(-g,d)<0:
            globals.numFact+=1
            H+=2**multiple*np.identity(m)
            d= -np.linalg.solve(H,g)
            multiple+=1
        while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit'] :
            iteration +=1
            x0=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 100}
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            x['H']=fun(x['p'],mode=4)[0]
            globals.numH+=1
            H=x['H']
            d= -np.linalg.solve(H,g)
            multiple=0
            while np.dot(-g,d)<0:
                globals.numFact+=1
                H+=2**multiple*np.identity(m)
                d= -np.linalg.solve(H,g)
                multiple+=1
            alfa=x.pop('alpha')
            alfa =1
            grad_seq.append(x['g'])
            if step_size['status']==0:
                break   
        globals.numFact+=1 #The first decomposition
        
    #_________________________________________________________________
    elif nparams ['method']== 'sppert':
        LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
        d=-LU.solve(g)
        
        multiple=0
        while np.dot(-g,d)<0:
            globals.numFact+=1
            H+=2**multiple*np.identity(m)
            LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
            d=-LU.solve(g)
            multiple+=1
        while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit'] :
            iteration +=1
            x0=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 100}
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            x['H']=fun(x['p'],mode=4)[0]
            H=x['H']
            globals.numH+=1
            LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
            d=-LU.solve(g)
            multiple=0
            while np.dot(-g,d)<0:
                globals.numFact+=1
                H+=2**multiple*np.identity(m)
                LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
                d=-LU.solve(g)
                multiple+=1
            alfa=x.pop('alpha')
            alfa =1
            grad_seq.append(x['g'])
            
            if step_size['status']==0:
                break   
        globals.numFact+=1 #The first decomposition
        
    #_________________________________________________________    
    x['grad_seq']=grad_seq
    solution=x
    if np.linalg.norm(x['g'])<nparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result

#______________________________________________________________________

def BFGS(fun,x,nparams = {'maxit': 1000,'toler': 1.0e-4}):
    m=len(x['p'])
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    H=np.eye(m) #H is not x['H']!
    g=x['g']
    alfa=1
    iteration=0
    grad_seq=[x['g']]
    while ((np.linalg.norm(x['g'], inf))/ ( min(1000, 1+abs(x['f'])))) >=10**(-4) and iteration<nparams['maxit']:
        iteration +=1
        xold=x.copy()
        params = {'ftol': 0.1,'gtol': 0.9,'xtol': 1e-12,'maxit': 1000}
        #params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-12,'maxit': 1000}
        d= -H@g
            
        step_size=StepSize(fun, x, d, alfa, params)
        x=step_size['x']
        g=x['g']
        s=x['p']-xold['p']
        y=x['g']-xold['g']
        rho=1/(y@s)
        if iteration ==1:
                gam=(s@y)/(y@y)
                
                H=((np.eye(m)-rho*np.outer(s,y))@ (gam*H) @ (np.eye(m)-rho*np.outer(s,y)))+rho*np.outer(s,s)
        else:
                H=((np.eye(m)-rho*np.outer(s,y))@ H @ (np.eye(m)-rho*np.outer(s,y)))+rho*np.outer(s,s)
        alfa=x.pop('alpha')
        alfa = 1
        grad_seq.append(x['g'])
        if step_size['status']==0:
            break
        #print( x['f'])
    x['grad_seq']=grad_seq
    solution=x
    if np.linalg.norm(x['g'])<nparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result

#________________________________________________________________________

def LBFGS(fun,x,nparams = {'maxit': 1000,'toler': 1.0e-4, 'm':3}):
    step_m=nparams['m']
    k=0
    m=len(x['p'])
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    x['H']=np.eye(m)
    g=x['g']
    H=x['H']
    
    alfa=1
    iteration=0
    grad_seq=[x['g']]
    s_seq=[]
    y_seq=[]
    
    rho_seq=[]
    
    while np.linalg.norm(x['g'])>=nparams['toler'] and k<nparams['maxit']:
        iteration+=1
        #First iteration k=0
        if k==0:
            xold=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 1000}
            d=-H@g
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            s=x['p']-xold['p']
            y=x['g']-xold['g']
            rho=1/np.dot(y,s)
            gamma=np.dot(s,y)/np.dot(y,y)
            H=gamma*np.eye(m)
            x['H']=H
            alfa=x.pop('alpha')
            alfa = 1
            grad_seq.append(x['g'])
            if step_size['status']==0:
                break
            s_seq.append(s)
            y_seq.append(y)
            rho_seq.append(rho)
            
        elif 1<=k<=step_m:
            
            g=x['g']
            q=x['g']
            eta_seq=np.zeros(k)
            for i in range(k-1,-1,-1):
                eta=rho_seq[i]*np.dot(s_seq[i],q)
                eta_seq[i]= eta
                q=q-eta_seq[i]*y_seq[i] 
            r=H@q
            for i in range(k):
                beta=rho_seq[i]*np.dot(y_seq[i],r)
                r=r+s_seq[i]*(eta_seq[i]-beta)
                
            d=-r
            xold=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 1000}
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            s=x['p']-xold['p']
            y=x['g']-xold['g']
            rho=1/np.dot(y,s)
            gamma=np.dot(s,y)/np.dot(y,y)
            H=gamma*np.eye(m)
            x['H']=H
            alfa=x.pop('alpha')
            alfa = 1
            grad_seq.append(x['g'])
            if step_size['status']==0:
                break
            s_seq.append(s)
            y_seq.append(y)
            rho_seq.append(rho)
            if k==step_m:
                s_seq.pop(0)
                y_seq.pop(0)
                rho_seq.pop(0)
                
        else:
            g=x['g']
            q=x['g']
            eta_seq=np.zeros(step_m)
            for i in range(step_m-1,-1,-1):
                
                eta=rho_seq[i]*np.dot(s_seq[i],q)
                eta_seq[i]=eta
                
                q=q-eta_seq[i]*y_seq[i] 
                
            r=H@q
            
            for i in range(step_m):
                beta=rho_seq[i]*np.dot(y_seq[i],r)
                r=r+s_seq[i]*(eta_seq[i]-beta)
                
            d=-r
            xold=x.copy()
            params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 1000}
            step_size=StepSize(fun, x, d, alfa, params)
            x=step_size['x']
            g=x['g']
            s=x['p']-xold['p']
            y=x['g']-xold['g']
            rho=1/np.dot(y,s)
            gamma=np.dot(s,y)/np.dot(y,y)
            H=gamma*np.eye(m)
            x['H']=H
            alfa=x.pop('alpha')
            alfa = 1
            grad_seq.append(x['g'])
            if step_size['status']==0:
                break
            s_seq.append(s)
            y_seq.append(y)
            rho_seq.append(rho)
            s_seq.pop(0)
            y_seq.pop(0)
            rho_seq.pop(0)
            
        k+= 1
        
    solution=x
    if np.linalg.norm(x['g'])<nparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result
        
        
#_______________________________________________________________________________        
        
def TNewton(fun,x,nparams = {'maxit': 1000,'toler': 1.0e-4}):
    m=len(x['p'])
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    x['H']=fun(x['p'],mode=4)[0]
    g=x['g']
    H=x['H']
    alfa=1
    iteration=0
    
    
    while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit']:
        x_cg=np.zeros(m)
        b_cg=-x['g']
        
        Q=x['H']
        g_cg= Q@x_cg-b_cg
        d_cg= -g_cg
        ite_cg=0
        while np.linalg.norm(x['H']@x_cg+x['g'])> min(0.5, np.linalg.norm(x['g'])**0.5)* np.linalg.norm(x['g']):
            globals.cgits+=1
            if d_cg@Q@d_cg<=0 and ite_cg==0:
                x_cg=d_cg
                break
            elif d_cg@Q@d_cg<=0 and ite_cg>0:
                break
            
            alfa_cg=-np.dot(g_cg,d_cg)/(d_cg@Q@d_cg )
            x_cg= x_cg+ alfa_cg *d_cg
            g_cg=Q@x_cg-b_cg
            beta= g_cg@Q@d_cg/(d_cg@Q@d_cg )
            d_cg=-g_cg+beta*d_cg
        
        iteration +=1
        x0=x.copy()
        params = {'ftol': 10**-4,'gtol': 0.9,'xtol': 1e-8,'maxit': 100}
        d=x_cg   
        step_size=StepSize(fun, x, d, alfa, params)
        x=step_size['x']
        g=x['g']
        x['H']=fun(x['p'],mode=4)[0]
        alfa=x.pop('alpha')
        alfa = 1
        
        if step_size['status']==0:
            break
    solution=x
    if np.linalg.norm(x['g'])<nparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result
        
#____________________________________________________________________________________        
        
def DogLeg(fun,x,nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}):
    m=len(x['p'])
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    x['H']=fun(x['p'],mode=4)[0]
    globals.numf+=1
    globals.numg+=1
    globals.numH+=1
    g=x['g']
    H=x['H']
    delta= nparams['initdel']
    iteration=0
    grad_seq=[x['g']]
    
    while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit']:
        iteration +=1
        if g@H@g <=0 or (np.linalg.norm(g)**3) / (g@H@g) >delta:
            p_C = -(delta/np.linalg.norm(g))* g
            p=p_C
        else:
            p_C = -((np.linalg.norm(g)**2)/ (g@H@g) ) *g
            if nparams.get('fail') == 'cauchy':
                    try:
                        globals.numFact+=1
                        LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
                        p_N=-LU.solve(g) 
                        assert np.dot(-g,p_N)>=0
                        
                    except:
                        p=p_C
            else:
                while True:  
                    multiple=0  
                    try:
                        globals.numFact+=1
                        LU=splinalg.splu(H,diag_pivot_thresh=0,permc_spec='MMD_AT_PLUS_A')
                        p_N=-LU.solve(g) 
                        assert np.dot(-g,p_N)>=0
                        break
                    except:
                        H+=2**multiple*np.identity(m)
                        #x['H']=H
                        multiple+=1
                theta= (np.dot(p_C, p_N-p_C))**2- (np.linalg.norm(p_N-p_C))**2 *(np.linalg.norm(p_C)**2-delta**2)

                if np.linalg.norm(p_N)<=delta:
                    p=p_N
                else:
                    r= (-np.dot(p_C, p_N-p_C)+ theta**0.5 )/ (np.linalg.norm(p_N-p_C)**2)
                    p=p_C+r*(p_N-p_C)
            
            
        
        rho= (x['f']-fun(x['p']+p,mode=1)[0])/ (- np.dot(x['g'],p)-1/2*p@x['H']@p)
        
        if rho<1/4:
            delta= delta/4
        elif rho>3/4 and np.linalg.norm(p)==delta:
            delta= min(2*delta, nparams ['delbar'])
        if rho>nparams ['eta']:
            x['p']= x['p']+ p
            x['f']=fun(x['p'],mode=1)[0]
            x['g']=fun(x['p'],mode=2)[0]
            x['H']=fun(x['p'],mode=4)[0]
            globals.numf+=1
            globals.numg+=1
            globals.numH+=1
            g=x['g']
            H=x['H']
            grad_seq.append(x['g'])
            
    x['grad_seq']=grad_seq
    solution=x
    if np.linalg.norm(x['g'])<nparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result    
        
        
        
#____________________________________________________________________________________        
        
def cgTrust(fun,x,nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}):
    m=len(x['p'])
    x['f']=fun(x['p'],mode=1)[0]
    x['g']=fun(x['p'],mode=2)[0]
    x['H']=fun(x['p'],mode=4)[0]
    g=x['g']
    H=x['H']
    delta= nparams['initdel']
    iteration=0
    grad_seq=[x['g']]
    
    while np.linalg.norm(x['g'])>=nparams['toler'] and iteration<nparams['maxit']:
        iteration+=1
        epis=min(0.5,np.linalg.norm(g)**0.5) * np.linalg.norm(g)
        s=np.zeros(m); r=g; d=-r
        while True:
            globals.cgits+=1
            if d@H@d<=0:
                tau=(-s@d+( (s@d)**2-(d@d)*(s@s-delta**2) )**0.5)/ (d@d)
                p=s+tau*d
                break
            alpha=(r@r)/(d@H@d)
            if (np.linalg.norm(s+alpha*d)) >=delta:
                tau=(-s@d+( (s@d)**2-(d@d)*(s@s-delta**2) )**0.5)/ (d@d)
                p=s+tau*d
                break
            s=s+alpha*d
            r_new=r+alpha*(H@d)
            if np.linalg.norm(r_new) <=epis:
                p=s
                break
            beta=(r_new@r_new)/(r@r)
            d=-r_new+beta*d
            r=r_new
        rho= (x['f']-fun(x['p']+p,mode=1)[0])/ (- np.dot(x['g'],p)-1/2*p@x['H']@p)
        
        if rho<1/4:
            delta= delta/4
        elif rho>3/4 and np.linalg.norm(p)==delta:
            delta= min(2*delta, nparams ['delbar'])
        if rho>nparams ['eta']:
            x['p']= x['p']+ p
            x['f']=fun(x['p'],mode=1)[0]
            x['g']=fun(x['p'],mode=2)[0]
            x['H']=fun(x['p'],mode=4)[0]
            globals.numf+=1
            globals.numg+=1
            globals.numH+=1
            g=x['g']
            H=x['H']
            grad_seq.append(x['g'])
        
    x['grad_seq']=grad_seq
    solution=x
    if np.linalg.norm(x['g'])<nparams['toler']:
        status=1
    else:
        status=0
    inform={'status':status, 'iter':iteration}
    result=[inform, solution]
    return result            
        
   

#Tests_________________________________________________________________________________

def probres(inform,x,params):
  np.set_printoptions(formatter={'float': '{:8.4f}'.format})
  if inform['status'] == 0:
    print('CONVERGENCE FAILURE:')
    print('{0} steps were taken without gradient size decreasing below {1:.4g}.\n'.format(inform['iter'],params['toler']))
  else:
    print('Success: {0} steps taken\n'.format(inform['iter']))

  #print('Ending point: {0}'.format(x['p']))
  print('Ending value: {0:.4g}'.format(x['f']))
  print('No. function evaluations: {0}'.format(globals.numf))
  #print('Ending gradient: {0}'.format(x['g']))
  print('No. gradient evaluations {0}'.format(globals.numg))
  print('Norm of ending gradient: {0:8.4g}'.format(np.linalg.norm(x['g'])))
  print('No. Hessian evaluations {0}'.format(globals.numH))
  print('No. Factorizations {0}'.format(globals.numFact))
  print('Cg iterations {0}\n'.format(globals.cgits))

#_____________________________________________________________________


#Q1

n_list=[200,400,1000,2000,4000]
for item in n_list:
    x = {'p': np.arange(1.0,item+1)}
    print('Newton')
    globals.initialize()
    nparams = {'maxit': 100,'toler': 1.0e-4,'method': 'sppert'}
    [inform,path] = Newton(woods,x,nparams)
    probres(inform,path,nparams)

#BFGS is very slow!
x = {'p': np.tile([-3.0,-1.0],200)} 
print('BFGS')
globals.initialize()
nparams = {'maxit': 5000,'toler': 1.0e-4}
[inform,path] = BFGS(woods,x,nparams)
probres(inform,path,nparams)


#Q2
print('LBFGS')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 17}
[inform,path] = LBFGS(woods,x,nparams)
probres(inform,path,sdparams)

#Q4
print('DogLeg')
#x = {'p': np.array([-1.2, 1.0])}
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(objg,x,nparams)
probres(inform,path,nparams)
 
print('DogLeg')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(woods,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

#'cauchy'
print('DogLeg')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 5000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1,'fail':'cauchy'}
[inform,path] = DogLeg(cragglvy,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

#This is slow! It fails in 10^4 steps.
print('DogLeg')
x = {'p': np.arange(1.0,2001.0)}
globals.initialize()
nparams = {'maxit': 10000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(indef,x,nparams)
probres(inform,path,nparams)

#This is slow! 
print('DogLeg')
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(cragglvy,x,nparams)
probres(inform,path,nparams)

#Q5
print('cgTrust')
x# = {'p': np.array([-1.2, 1.0])}
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(objg,x,nparams)
probres(inform,path,nparams)


#Q6
#'woods'
print('woods:')

print('Newton')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'method': 'sppert'}
[inform,path] = Newton(woods,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('LBFGS')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 17}
[inform,path] = LBFGS(woods,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('TNewton')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = TNewton(woods,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('DogLeg')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(woods,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('cgTrust')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 2000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(woods,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)




#'indef':
print('indef:')
print('Newton')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'method': 'sppert'}
#[inform,path] = Newton(indef,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('LBFGS')
x = {'p': 2.0*np.ones((10000,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-5,'m': 17}
[inform,path] = LBFGS(indef,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('TNewton')
x = {'p': 2.0*np.ones((10000,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-5}
[inform,path] = TNewton(indef,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

#This is slow! 
print('DogLeg')
x = {'p': 2.0*np.ones((10000,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 10000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 10}
[inform,path] = DogLeg(indef,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

#This is slow! 
print('cgTrust')
x = {'p': 2.0*np.ones((10000,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 10}
[inform,path] = cgTrust(indef,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)


# 'cragglvy'

print('cragglvy:')
print('LBFGS')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 17}
[inform,path] = Opt726.LBFGS(cragglvy,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('TNewton')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = TNewton(cragglvy,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('DogLeg')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0 
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(cragglvy,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('cgTrust')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0 
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(cragglvy,x,nparams)
probres(inform,path,nparams)
print("The time difference is :", timeit.default_timer() - starttime)

