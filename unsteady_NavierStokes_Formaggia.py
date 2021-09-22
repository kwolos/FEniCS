# %%
from dolfin import * 
from petsc4py import PETSc
import numpy as np 
from tqdm import trange

set_log_level(40)
# pylint: disable=unbalanced-tuple-unpacking
RKSolver

# basic data connected with the simulation 
T           = 1.0       # time of the simulation
dt          = 0.001      # time step 
nu_         = 1         # dynamic viscosity [kg/m/s] 
num_steps   = T/dt      # number of steps 
mRK4        = 2         # number of steps in RK4 scheme in OIFS procedure
h           = dt/mRK4   # step in RK4 scheme 

# flow through the inputs/outputs (X - geometry, numbering from 0)
Q_1 = 0 
Q_2 = 0 
Q_3 = 0

# backward method - coefficients 
def BDF(J):
    alpha = np.zeros(5) 
    if J == 1: 
        alpha[0:2] = 1, 1 
        
    elif J == 2:
        alpha[0:3] = 1.5, 2, -0.5
    
    elif J == 3:
        alpha[0:4] = 11/6, 3, -1.5, 1/3

    else:
        alpha[:] = 25/12, 4, -3, 4/3, -0.25 
    
    return alpha

f_ODE = lambda u:  - dot(u, nabla_grad(u))

def RK4_model(un_, un0, mRK4):

    for i in range(mRK4):       
        if i == 0:
            un_.vector()[:] = un0.vector()

        k1 = project(f_ODE(un_),           ME0.sub(0).collapse())
        k2 = project(f_ODE(un_ + h/2*k1),  ME0.sub(0).collapse())       
        k3 = project(f_ODE(un_ + h/2*k2),  ME0.sub(0).collapse())
        k4 = project(f_ODE(un_ + h*k3),    ME0.sub(0).collapse())
        
        un_ = project(un_ + h/6 * (k1 + 2*k2 + 2*k3 + k4), ME0.sub(0).collapse())
        
        print(un_.vector()[:])
    
    return(un_)


alpha = BDF(1)


# loading mesh 
mesh = Mesh('siatki/siatki_xml/geometria_X_gestsza/X.xml')
cd   = MeshFunction('size_t', mesh, 'siatki/siatki_xml/geometria_X_gestsza/X_physical_region.xml')      # inside
fd   = MeshFunction('size_t', mesh, 'siatki/siatki_xml/geometria_X_gestsza/X_facet_region.xml')         # boundaries 

# defining function spaces for subproblems 

# subproblem I and inflows/outflows 
V0_elem  = VectorElement('P', mesh.ufl_cell(), 2)
Q0_elem  = FiniteElement('P', mesh.ufl_cell(), 1)
ME0_elem = MixedElement([V0_elem, Q0_elem])
ME0      = FunctionSpace(mesh, ME0_elem)

V2 = VectorFunctionSpace(mesh, 'P', 2)
Q2 = FunctionSpace(mesh, 'P', 1)

V3 = VectorFunctionSpace(mesh, 'P', 2)
Q3 = FunctionSpace(mesh, 'P', 1)

# test functions for zero subproblem 
v0, q0 = TestFunctions(ME0)

# test functions for other subproblem (to lagrange multipliers)
v1, q1 = TestFunctions(ME0)
v2, q2 = TestFunctions(ME0)
v3, q3 = TestFunctions(ME0)

## defining solutions 

# zero subproblem
_w0    = Function(ME0)
w0, p0 = split(_w0)

# other subproblems
_w1     = Function(ME0) 
w1, p1  = split(_w1) 

_w2     = Function(ME0) 
w2, p2  = split(_w2) 

_w3     = Function(ME0) 
w3, p3  = split(_w3)

# previous time step
_un     = Function(ME0)
un, pn  = split(_un)

# initializing previous time step by zeros 
var_un  = Expression(('0', '0'), degree = 2)
var_pn  = Expression('0', degree = 1)

un      = project(var_un, ME0.sub(0).collapse())
pn      = project(var_pn, ME0.sub(1).collapse())

un_1     = project(var_un, ME0.sub(0).collapse())
pn_1     = project(var_pn, ME0.sub(1).collapse())
un_1prev = project(var_un, ME0.sub(0).collapse())

un_2     = project(var_un, ME0.sub(0).collapse())
pn_2     = project(var_pn, ME0.sub(1).collapse())
un_2prev = project(var_un, ME0.sub(0).collapse())

un_3     = project(var_un, ME0.sub(0).collapse())
pn_3     = project(var_pn, ME0.sub(1).collapse())
un_3prev = project(var_un, ME0.sub(0).collapse())


# imposing on the walls no-slip BC == zero Dirichlet BC 
bcu_walls = DirichletBC(ME0.sub(0), Constant((0.0, 0.0)), fd, 5)

# defining measures of the integrals 
ds = Measure('ds', domain=mesh, subdomain_data=fd)
dx = Measure('dx', domain=mesh, subdomain_data=cd)

# function f, describing external influence on the model 
f  = Expression(('0', '0'), degree=2)

# defining constant values which occurs in weak formulations 
al0      = Constant(alpha[0])
k        = Constant(dt)
nu       = Constant(nu_)

# normal vector 
n       = FacetNormal(mesh)

## three Stokes subproblems
# I 
F1 = al0/k * dot(w1, v1) * dx \
    + nu * inner(grad(w1), grad(v1)) * dx \
        - p1 * div(v1) * dx \
            + dot(v1, n) * ds(1) \
                + q1 * div(w1) * dx 

solve(F1 == 0, _w1, [bcu_walls])
u1, p1 = _w1.split() 

# count integrals for the matrix 
F11 = assemble(dot(u1, n) * ds(1))
F21 = assemble(dot(u1, n) * ds(2))
F31 = assemble(dot(u1, n) * ds(3))

# II
F2 = al0/k * dot(w2, v2) * dx \
    + nu * inner(grad(w2), grad(v2)) * dx \
        - p2 * div(v2) * dx \
            + dot(v2, n) * ds(2) \
                + q2 * div(w2) * dx 

solve(F2 == 0, _w2, [bcu_walls])
u2, p2 = _w2.split() 

# count integrals for the matrix 
F12 = assemble(dot(u2, n) * ds(1))
F22 = assemble(dot(u2, n) * ds(2))
F32 = assemble(dot(u2, n) * ds(3))

# III 
F3 = al0/k * dot(w3, v3) * dx \
    + nu * inner(grad(w3), grad(v3)) * dx \
        - p3 * div(v3) * dx \
            + dot(v3, n) * ds(3) \
                + q3 * div(w3) * dx 

solve(F3 == 0, _w3, [bcu_walls])
u3, p3 = _w3.split() 

# count integrals for the matrix 
F13 = assemble(dot(u3, n) * ds(1))
F23 = assemble(dot(u3, n) * ds(2))
F33 = assemble(dot(u3, n) * ds(3))

A1 = PETSc.Mat().create()
A1.setSizes([3,3])
A1.setType('aij')
A1.setUp()

A = np.array([[F11, F12, F13], 
              [F21, F22, F23], 
              [F31, F32, F33]])


A1.setValues([0,1,2], [0,1,2], A)
A1.assemble() 

M = PETScMatrix(A1)
xdmf_file0 = XDMFFile('NavierStokesFormaggia/solution_u.xdmf')
xdmf_file1 = XDMFFile('NavierStokesFormaggia/solution_p.xdmf')

comm = MPI.comm_world

# flow rates 
flowrate1 = assemble(dot(u1, n) * ds(1) + dot(u1, n) * ds(2) + dot(u1, n) * ds(3) + dot(u1,n) * ds(4))
flowrate2 = assemble(dot(u2, n) * ds(1) + dot(u2, n) * ds(2) + dot(u2, n) * ds(3) + dot(u2,n) * ds(4))
flowrate3 = assemble(dot(u3, n) * ds(1) + dot(u3, n) * ds(2) + dot(u3, n) * ds(3) + dot(u3,n) * ds(4))

# ---- step 0 ------ 
t = 0.0 
al1 = Constant(alpha[1])
al2 = Constant(alpha[2])
al3 = Constant(alpha[3])

F0 = al0/k * dot(w0, v0) * dx \
    + nu * inner(grad(w0), grad(v0)) * dx \
        - p0 * div(v0) * dx \
            - dot(f, v0) * dx \
                - al1/k * dot(un_1, v0) * dx \
                    + q0 * div(w0) * dx
                    # - al2/k * dot(un_2, v0) * dx \
                        # - al3/k * dot(un_3, v0) * dx \
                             
# %%
for qqq in trange(int(num_steps)):
    
    solve(F0 == 0, _w0, [bcu_walls])

    # we suppose that we have knowledge about inflows/outflows (only)
    Q_1 = 10
    Q_2 = 10
    Q_3 = 10

    u0, p0 = _w0.split()

    F01 = assemble(dot(u0, n) * ds(1))
    F02 = assemble(dot(u0, n) * ds(2))
    F03 = assemble(dot(u0, n) * ds(3))

    b = np.zeros(3)
    b[0] = Q_1 - F01
    b[1] = Q_2 - F02 
    b[2] = Q_3 - F03
    
    z    = Vector(comm, 3)
    z.add_local(b)

    beta = Vector(comm, 3)

    # solving equation M * beta = z 
    solve(M, beta, z)

    # parameters for the linear combination 
    bt = beta.get_local()
    b1, b2, b3 = float(bt[0]), float(bt[1]), float(bt[2])

    # assemble of the solution as an combination of Stokes subproblems 
    un_temp = project(u0 + b1*u1 + b2*u2 + b3*u3, ME0.sub(0).collapse())
    pn_temp = project(p0 + b1*p1 + b2*p2 + b3*p3, ME0.sub(1).collapse())

    
    # in BDF_3 we need to store last 3 solution in memory 
    # un   -- given solution 
    # un_1 -- previosu solutions (-1) un_2 -- prev sol. (-2) and so on... 


    # here we count solution of OIFS step, which is used in F0 variational form 
 
    # if qqq > 2: 
    #     un_3prev.vector()[:] = un_2.vector()
    #     _un_3 = RK4_model(un_3, un_3prev, 3*h)
    #     un_3.assign(_un_3)
    
    # if qqq > 1: 
    #     un_2prev.vector()[:] = un_1prev.vector() 
    #     _un_2 = RK4_model(un_2, un_2prev, 2*mRK4)
    #     un_2.assign(_un_2)
    
    if qqq > 0:
        un_1prev.vector()[:] = un.vector()
        _un_1 = RK4_model(un_1, un_1prev, mRK4)
        un_1.assign(_un_1)

    # these (whole) solution will be saved 
    un.assign(un_temp)
    pn.assign(pn_temp)

    # print solution
    print("solution", un.vector()[:])
    

    xdmf_file0.write(un, t)
    xdmf_file1.write(pn, t)

    t += dt 

# %%