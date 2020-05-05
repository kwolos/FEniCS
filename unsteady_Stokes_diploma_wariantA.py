#============================================================================================
#============================================================================================
#=============================  KAMIL WOŁOS - PRACA DYPLOMOWA ===============================
#============================== WARIANT A WARUNKU BRZEGOWEGO ================================
#============================================================================================
#============================================================================================

# zagadnienie z trzema wlotami/ wylotami


# import biblioteki dolfin i potrzebnych bibliotek do realizacji kodu 
from dolfin import * 
import numpy as np 
from petsc4py import PETSc
from tqdm import trange 

set_log_level(40)

# podstawowe dane dotyczące symulacji 
T           = 5.0           # czas symulacji 
num_steps   = 5000          # liczba kroków czasowych
lll         = 1000          # żeby okres się zgadzał
dt          = T/num_steps   # długość kroku czasowego 
nu          = 0.001         # lepkość (dynamiczna) 
rho         = 1             # gęstość


# parametry wlotowo - wylotowe 
# zadaj funkcję S_1 przy wlocie (1)
S_1 = 100
# zadaj wydatek przy wylocie (2) 
Q_2 = 0
# zadaj wydatek przy wylocie (3) 
Q_3 = 50
# zadaj wydatej przy wylocie (4)
Q_4 = 50

# współczynniki alpha do metody backward 

# J == 1 
alpha_0 = 1 
alpha_1 = 1 

# współczynniki lambda i gamma (podane w formie tablicy) 
_lambda = np.array([0.001, 0.001, 0.001, 0.001]) 
_gamma  = np.array([0.001, 0.001, 0.001 ,0.001]) 
_eta    = _lambda + alpha_0/dt * _gamma

# wczytywanie siatki 
mesh = Mesh('siatki/siatki_xml/geometria_X/X.xml')                                              # wczytanie siatki
cd   = MeshFunction('size_t', mesh, 'siatki/siatki_xml/geometria_X/X_physical_region.xml')      # wnętrze 
fd   = MeshFunction('size_t', mesh, 'siatki/siatki_xml/geometria_X/X_facet_region.xml')         # brzegi 

# zdefiniowanie przestrzeni funkcyjnych dla podzagadnień 

# podzagadnienie I oraz wloty i wyloty
V0_elem  = VectorElement('P', mesh.ufl_cell(), 2)
Q0_elem  = FiniteElement('P', mesh.ufl_cell(), 1)
ME0_elem = MixedElement([V0_elem, Q0_elem])
ME0      = FunctionSpace(mesh, ME0_elem)

V2 = VectorFunctionSpace(mesh, 'P', 2) 
Q2 = FunctionSpace(mesh, 'P', 1) 

V3 = VectorFunctionSpace(mesh, 'P', 2) 
Q3 = FunctionSpace(mesh, 'P', 1) 

V4 = VectorFunctionSpace(mesh, 'P', 2) 
Q4 = FunctionSpace(mesh, 'P', 1)

# nie będziemy zadawać innego warunku brzegowego, niż Dirichleta narzucony na ściany 

## zdefiniowanie warunków brzegowych 

bcu_walls    = DirichletBC(ME0.sub(0), Constant((0,0)), fd, 5) 


# zdefiniowanie miar całek 
ds = Measure('ds', domain =mesh, subdomain_data =fd)
dx = Measure('dx', domain =mesh, subdomain_data =cd)


# zerowe zagadnienie 
v0, q0  = TestFunctions(ME0) 

# podzgagadnienia dla wlotów/ wylotów 
v1, q1  = TestFunctions(ME0) 
v2, q2  = TestFunctions(ME0) 
v3, q3  = TestFunctions(ME0) 
v4, q4  = TestFunctions(ME0)

## zdefiniowanie rozwiązań 

# ---- zerowe zagadnienie 
_w0     = Function(ME0) 
w0, p0  = split(_w0) 

# ---- podzagadnienia dla wlotów i wylotów 
_w1     = Function(ME0) 
w1, p1  = split(_w1) 

_w2     = Function(ME0) 
w2, p2  = split(_w2) 

_w3     = Function(ME0) 
w3, p3  = split(_w3)

_w4     = Function(ME0) 
w4, p4  = split(_w4)

# ----- poprzedni krok czasowy 
_un     = Function(ME0)
un, pn  = split(_un)

# ----- warunek początkowy na poprzedni krok czasowy
war_01  = Expression(('0', '0'), degree =2)
war_02  = Expression('0', degree =1) 

un      = project(war_01, ME0.sub(0).collapse())
pn      = project(war_02, ME0.sub(1).collapse())

# ----- funkcja g ze sformułowania 
# (można rzecz jasna zaproponować funkcję zależną od czasu/ współrzędnych, 
#  wtedy trzeba użyć Expression i w każdym korku aktualizować funkcję)
g       = Expression(('0', '0'), degree =2)                   


# zdefniowanie stałych wartości wystepujących słabych w postaciach 
alpha_0 = Constant(alpha_0)
k       = Constant(dt)
nu      = Constant(nu)
gamma1  = Constant(_gamma[0])
gamma2  = Constant(_gamma[1])
gamma3  = Constant(_gamma[2])
gamma4  = Constant(_gamma[3])
eta1    = Constant(_eta[0])
eta2    = Constant(_eta[1])
eta3    = Constant(_eta[2])
eta4    = Constant(_eta[3])
n       = FacetNormal(mesh)         # wektor normalny


F1 = alpha_0/k * dot(w1, v1) * dx \
    + nu * inner(grad(w1), grad(v1)) * dx \
        - p1 * div(v1)  * dx \
            + dot(v1, n) * ds(1) \
                + q1 * div(w1) * dx \
                    + eta1 * dot(w1, n) * dot(v1, n) * ds(1) \
                        + eta2 * dot(w1, n) * dot(v1, n) * ds(2) \
                            + eta3 * dot(w1, n) * dot(v1, n) * ds(3) \
                                + eta4 * dot(w1, n) * dot(v1, n) * ds(4)

solve(F1 == 0, _w1, [bcu_walls])
u1, p1 = _w1.split()

# policz całki do macierzy 
F11 = assemble(dot(u1, n) * ds(1))
F12 = assemble(dot(u1, n) * ds(2))
F13 = assemble(dot(u1, n) * ds(3))
F14 = assemble(dot(u1, n) * ds(4))


F2 = alpha_0/k * dot(w2, v2) * dx \
    + nu * inner(grad(w2), grad(v2)) * dx \
        - p2 * div(v2)  * dx \
            + dot(v2, n) * ds(2) \
                + q2 * div(w2) * dx \
                    + eta1 * dot(w2, n) * dot(v2, n) * ds(1) \
                        + eta2 * dot(w2, n) * dot(v2, n) * ds(2) \
                            + eta3 * dot(w2, n) * dot(v2, n) * ds(3) \
                                + eta4 * dot(w2, n) * dot(v2, n) * ds(4)

solve(F2 == 0, _w2, [bcu_walls])
u2, p2 = _w2.split()

F21 = assemble(dot(u2, n) * ds(1))
F22 = assemble(dot(u2, n) * ds(2))
F23 = assemble(dot(u2, n) * ds(3))
F24 = assemble(dot(u2, n) * ds(4))


F3 = alpha_0/k * dot(w3, v3) * dx \
    + nu * inner(grad(w3), grad(v3)) * dx \
        - p3 * div(v3)  * dx \
            + dot(v3, n) * ds(3) \
                + q3 * div(w3) * dx \
                    + eta1 * dot(w3, n) * dot(v3, n) * ds(1) \
                        + eta2 * dot(w3, n) * dot(v3, n) * ds(2) \
                            + eta3 * dot(w3, n) * dot(v3, n) * ds(3) \
                                +eta4 * dot(w3, n) * dot(v3, n) * ds(4)

 
solve(F3 == 0, _w3, [bcu_walls])
u3, p3 = _w3.split()

F31 = assemble(dot(u3, n) * ds(1))
F32 = assemble(dot(u3, n) * ds(2))
F33 = assemble(dot(u3, n) * ds(3))
F34 = assemble(dot(u3, n) * ds(4))


F4 = alpha_0/k * dot(w4, v4) * dx \
    + nu * inner(grad(w4), grad(v4)) * dx \
        - p4 * div(v4)  * dx \
            + dot(v4, n) * ds(4) \
                + q4 * div(w4) * dx \
                    + eta1 * dot(w4, n) * dot(v4, n) * ds(1) \
                        + eta2 * dot(w4, n) * dot(v4, n) * ds(2) \
                            + eta3 * dot(w4, n) * dot(v4, n) * ds(3) \
                                + eta4 * dot(w4, n) * dot(v4, n) * ds(4)
 
solve(F4 == 0, _w4, [bcu_walls])
u4, p4 = _w4.split()

F41 = assemble(dot(u4, n) * ds(1))
F42 = assemble(dot(u4, n) * ds(2))
F43 = assemble(dot(u4, n) * ds(3))
F44 = assemble(dot(u4, n) * ds(4))


A1 = PETSc.Mat().create() 
A1.setSizes([4,4])
A1.setType('aij') # zaimplementować inną macierz, bo to są rzadkie
A1.setUp()

A = np.zeros([4,4])
A[0,0] = 1 
A[0,1] = 0 
A[0,2] = 0
A[0,3] = 0 

A[1,0] = 0
A[1,1] = F22 
A[1,2] = F32 
A[1,3] = F42

A[2,0] = 0 
A[2,1] = F23 
A[2,2] = F33
A[2,3] = F43

A[3,0] = 0
A[3,1] = F24
A[3,2] = F34
A[3,3] = F44

A1.setValues([0,1,2,3], [0,1,2,3], A)
A1.assemble()

M = PETScMatrix(A1)

xdmf_file0 = XDMFFile('mgr_x2_t_' + str(num_steps) +'/solution_u.xdmf')
xdmf_file1 = XDMFFile('mgr_x2_t_' + str(num_steps) +'/solution_p.xdmf')

comm = MPI.comm_world

wydatek1 = assemble(dot(u1, n) * ds(1) + dot(u1, n) * ds(2) + dot(u1, n) * ds(3) + dot(u1,n) * ds(4))
wydatek2 = assemble(dot(u2, n) * ds(1) + dot(u2, n) * ds(2) + dot(u2, n) * ds(3) + dot(u2,n) * ds(4))
wydatek3 = assemble(dot(u3, n) * ds(1) + dot(u3, n) * ds(2) + dot(u3, n) * ds(3) + dot(u3,n) * ds(4))
wydatek4 = assemble(dot(u4, n) * ds(1) + dot(u4, n) * ds(2) + dot(u4, n) * ds(3) + dot(u4,n) * ds(4))


##-------------- KROK ZEROWY --------------------------------------------------------------

t = 0.0

# problem Stokesa
f1 = dot(un, n) * ds(1)
f1 = Constant(assemble(f1))

f2 = dot(un, n) * ds(2) 
f2 = Constant(assemble(f2))

f3 = dot(un, n) * ds(3) 
f3 = Constant(assemble(f3))

f4 = dot(un, n) * ds(4) 
f4 = Constant(assemble(f4))
            


F0 = alpha_0/k * dot(w0, v0) * dx \
    + nu * inner(grad(w0), grad(v0)) * dx \
        - p0 * div(v0)  * dx \
            + q0 * div(w0) * dx \
                - dot(g , v0) * dx \
                    - alpha_1/k * dot(un, v0) * dx \
                        - alpha_1/k * gamma1 * dot(un, n) * dot(v0, n) * ds(1) \
                            - alpha_1/k * gamma2 * dot(un, n) * dot(v0, n) * ds(2) \
                                - alpha_1/k * gamma3 * dot(un, n) * dot(v0, n) * ds(3) \
                                    - alpha_1/k * gamma4 * dot(un, n) * dot(v0, n) * ds(4) \
                                        + _eta[0] * dot(w0, n) * dot(v0, n) * ds(1) \
                                            + _eta[1] * dot(w0, n) * dot(v0, n) * ds(2) \
                                                + _eta[2] * dot(w0, n) * dot(v0, n) * ds(3) \
                                                    + _eta[3] * dot(w0, n) * dot(v0, n) * ds(4) 


# ----- MACIERZ DO ZAPISYWANIA WYNIKÓW -------
flux = np.zeros([num_steps, 9])
zzz  = 0

for qqq in trange(num_steps):
            

    solve(F0 == 0, _w0, [bcu_walls])
    
    Q_2 = 50 * np.sin(qqq/lll * np.pi)
    Q_3 = 50 - Q_2
    Q_4 = 50 + Q_2
    
    
    u0, p0 = _w0.split() 


    F01 = assemble(dot(u0, n) * ds(1)) 
    F02 = assemble(dot(u0, n) * ds(2)) 
    F03 = assemble(dot(u0, n) * ds(3))
    F04 = assemble(dot(u0, n) * ds(4))

    b = np.zeros(4) 
    b[0] = S_1
    b[1] = Q_2 - F02 - S_1 * F12
    b[2] = Q_3 - F03 - S_1 * F13
    b[3] = Q_4 - F04 - S_1 * F14

    beta = Vector(comm, 4)

    z = Vector(comm, 4)
    z.add_local(b)

    solve(M, beta, z)

    bt = beta.get_local()

    b1 = float(bt[0])
    b2 = float(bt[1])
    b3 = float(bt[2])
    b4 = float(bt[3])


    # poprzednie rozwiązanie 
    un_temp = project(u0 + b1*u1 + b2*u2 + b3*u3 + b4*u4, ME0.sub(0).collapse()) 
    pn_temp = project(p0 + b1*p1 + b2*p2 + b3*p3 + b4*p4, ME0.sub(1).collapse())

    un.vector()[:] = un_temp.vector()
    pn.vector()[:] = pn_temp.vector()
    
    # problem Stokesa
    f1  = dot(un, n) * ds(1)
    f11 = assemble(f1)
    f1  = Constant(assemble(f1))
    
    f2  = dot(un, n) * ds(2) 
    f22 = assemble(f2) 
    f2  = Constant(assemble(f2))

    f3  = dot(un, n) * ds(3) 
    f33 = assemble(f3)
    f3  = Constant(assemble(f3))
    
    f4  = dot(un, n) * ds(4) 
    f44 = assemble(f4)
    f4  = Constant(assemble(f4))

    xdmf_file0.write(un, t)
    xdmf_file1.write(pn, t)
    
    t += dt 
    
    #if qqq%100 == 0:
    flux[zzz, 0:4] = [f11+b1*wydatek1, f22+b2*wydatek2, f33+b3*wydatek3, f44+b4*wydatek4]
    flux[zzz, 4:8] = [b1*wydatek1, b2*wydatek2, b3*wydatek3, b4*wydatek4] 
    flux[zzz, 8]   = f11 + f22 + f33 + f44 + b1*wydatek1 + b2*wydatek2 + b3*wydatek3 + b4*wydatek4
    np.savetxt('results_x_t_' + str(num_steps)+ '.csv', flux, delimiter =',')
    zzz += 1 
    print('zapisano dla t = ' + str(round(t, 1)) + ' sekundy')
    print('wydatek: ', f11 + f22 + f33 + f44 + b1*wydatek1 + b2*wydatek2 + b3*wydatek3 + b4*wydatek4)
    
    
