# soving a system of 2 quadratic equations of two variables
# interseciton points of planar conics 
# common tangetnts to planar conics
    
import numpy as np
import math

# technical module functions, detials
def homogenize(x):
    return np.array([x[0], x[1], 1])

def cos_sin(angle_deg):
    return math.cos(angle_deg*math.pi/180), math.sin(angle_deg*math.pi/180)

def rotation(cs_sn):
    return np.array([[cs_sn[0], -cs_sn[1]], 
                     [cs_sn[1],  cs_sn[0]]])

# defeining the isometry (the rotation plus translation) transformations that
# between coordinate system aligned with the conic and a general (world) 
# coordinate system
def isom_inverse(angle, translation):
    '''
    isometry from conic-aligned coordinate system (conic attached)
    to global coordinate system (world system) 
    '''
    cos_, sin_ = cos_sin(angle)
    return np.array([[cos_, -sin_, translation[0]], 
                     [sin_,  cos_, translation[1]], 
                     [   0,     0,             1]])
    
def isom(angle, translation):
    '''
    isometry from global coordinate system (world system) 
    to conic-aligned coordinate system (conic attached) 
    '''
    cos_, sin_ = cos_sin(-angle)
    tr = - rotation((cos_, sin_)).dot(translation)
    return np.array([[ cos_, -sin_, tr[0]], 
                     [ sin_,  cos_, tr[1]], 
                     [    0,     0,    1 ]])

# calculating the coinc defined by a pair of axes' lengts, 
# axes rotation angle and center of the conic  
def Conic(major, minor, angle, center):
    D = np.array([[minor**2,        0,                 0],
                  [       0, major**2,                 0], 
                  [       0,        0, -(minor*major)**2]])
    U = isom(angle, center)
    return (U.T).dot(D.dot(U))     

# calculating the coinc dual to the conic defined by a pair of axes' lengths, 
# axes rotation angle and center of the conic  
def dual_Conic(major, minor, angle, center):
    D_1 = np.array([[major**2,        0,  0], 
                    [       0, minor**2,  0], 
                    [       0,        0, -1]])
    U_1 =  isom_inverse(angle, center)
    return (U_1).dot(D_1.dot(U_1.T)) 

# transfromaing the matrix of a conic into a vector of six coeficients
# of a quadratic equation with two variables
def conic_to_equation(C):
    '''
    c[0]*x**2 + c[1]*x*y + c[2]*y**2 + c[3]*x + c[4]*y + c[5] = 0
    '''
    return np.array([C[0,0], 2*C[0,1], C[1,1], 2*C[0,2], 2*C[1,2], C[2,2]])    

# transforming the vector of six coeficients
# of a quadratic equation with two variables into a matrix of 
# the corresponid conic 
def equation_to_conic(eq):
    '''
    eq[0]*x**2 + eq[1]*x*y + eq[2]*y**2 + eq[3]*x + eq[4]*y + eq[5] = 0
    '''
    return np.array([[2*eq[0],   eq[1],   eq[3]],
                     [  eq[1], 2*eq[2],   eq[4]],
                     [  eq[3],   eq[4], 2*eq[5]]]) / 2

# given a point (x,y) define the vector (x^2, xy, y^2, x, y, 1)    
def argument(x):
    return np.array([x[0]**2, x[0]*x[1], x[1]**2, x[0], x[1], 1])

# given x = (x[0],x[1]) calculate the value of the quadratic equation with
# six coefficients coeff
def quadratic_equation(x, coeff):
    '''
    coeff[0]*x**2 + coeff[1]*x*y + coeff[2]*y**2 + coeff[3]*x + coeff[4]*y + coeff[5] = 0
    '''
    return coeff.dot( argument(x) )    

# given a pair of conics, as a pair of symmetric matrices, 
# calculate the vector k = (k[0], k[1], k[2]) of values for each of which 
# the conic c1 - k[i]*c2 from the pencil of conics c1 - t*c2 
# is a degenerate conic (the anti-symmetric product of a pair of linear forms) 
# and also find the matrix U 
# of the projective transfrormation that simplifies the geometry of 
# the pair of conics, the geometry of the pencil c1 - t*c2 in general, 
# as well as the geoemtry of the three degenerate conics in particular    
def transform(c1, c2):
    '''
    c1 and c2 are 3 by 3 symmetric matrices of the two conics
    '''
    c21 = np.linalg.inv(c2).dot(c1)
    k, U = np.linalg.eig(c21)
    return k, U

# the same as before, but for a pair of equations instead of matrices of conics
def eq_transform(eq1, eq2):
    '''
    eq1 and eq2 = np.array([eq[0], eq[1], eq[2], eq[3], eq[4], eq[5]])
    '''
    C1 = equation_to_conic(eq1)
    C2 = equation_to_conic(eq2)
    return transform(C1, C2)

# realizing the matrix U as a projective transformation
def proj(U, x):
    if len(x) == 2:
        x = homogenize(x)
    y = U.dot(x)
    y = y / y[2]
    return y[0:2]

# find the common points, i.e. points of intersection of a pair of conics
# represented by a pair of symmetric matrices
def find_common_points(c1, c2):
    k, U = transform(c1, c2)
    L1 = (U.T).dot((c1 - k[0]*c2).dot(U))
    L2 = (U.T).dot((c1 - k[1]*c2).dot(U))
    sol = np.empty((4,3), dtype=float)
    for i in range(2):
        for j in range(2):
            sol[i+2*j,0:2] = np.array([math.sqrt(abs(L2[2,2] / L2[0,0]))*(-1)**i, math.sqrt(abs(L1[2,2] / L1[1,1]))*(-1)**j])
            sol[i+2*j,0:2] = proj(U, sol[i+2*j,0:2])
    sol[:,2] = np.ones(4)
    return sol

# find the solutions, i.e. the points x=(x[0],x[1]) saisfying the pair 
# of quadratic equations 
# represented by a pair of vectors eq1 and eq2 of 6 coefficients
def solve_eq(eq1, eq2):
    conic1 = equation_to_conic(eq1)
    conic2 = equation_to_conic(eq2)
    return find_common_points(conic1, conic2)


'''
Esample of finding the common tangents of a pair of conics:
conic 1: major axis = 2, minor axis = 1, angle = 45, center = (0,0)
conic 2: major axis = 3, minor axis = 1, angle = 120, center = (15,0)
'''

a = 2
b = 1
cntr = np.array([0,0])
w = 45

Q1 = Conic(a, b, w, cntr)
dQ1 = dual_Conic(a, b, w, cntr)

a = 3
b = 1
cntr = np.array([15,0])
w = 120

Q2 = Conic(a, b, w, cntr)
dQ2 = dual_Conic(a, b, w, cntr)

#dQ12 = np.linalg.inv(dQ1).dot(dQ2)
#m, W = np.linalg.eig(dQ12)
    
#k, U = transform(dQ1, dQ2)  
#L1 = (U.T).dot((dQ1 - k[0]*dQ2).dot(U))
#L2 = (U.T).dot((dQ1 - k[1]*dQ2).dot(U))
#L3 = (U.T).dot((dQ1 - k[2]*dQ2).dot(U))

R = find_common_points(dQ1, dQ2)

print('')
print(R)
print('')
print('checking that the output forms common tangent lines: ')
print('')
print('conic 1: ')
print(np.diagonal(R.dot(dQ1.dot(R.T))) )
print('')
print('conic 2: ')
print(np.diagonal(R.dot(dQ2.dot(R.T))) )
