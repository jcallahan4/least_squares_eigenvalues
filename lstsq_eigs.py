# lstsq_eigs.py
"""Least Squares and Computing Eigenvalues
Jake Callahan

Because of its numerical stability and convenient structure, the QR decomposition
is the basis of many important and practical algorithms. In this program, I model
linear least squares problems, tools in Python for computing least squares solutions,
and two fundamental algorithms for computing eigenvalue. The QR decomposition makes
solving several of these problems quick and numerically stable
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath

def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    #Get QR decomposition
    Q,R = la.qr(A, mode = 'economic')
    #Create right side of equation
    solve = Q.T.dot(b)

    #Solve system
    solution = la.solve_triangular(R, solve)

    return solution

def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #Set up initial data
    housing = np.load("Housing.npy")
    y = housing[:,1]
    ones = np.ones(len(y))

    #Create A matrix
    A = np.column_stack((housing[:,0], ones))
    #Get linear regression
    line_fit = least_squares(A, y)
    #Domain to graph over
    x = np.linspace(0,16,100)
    #Graph linear regression vs raw data
    plt.plot(housing.T[0], housing.T[1], 'go')
    plt.plot(x, line_fit[0]*x + line_fit[1], 'g-')
    plt.show()

def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #Set up initial data
    housing = np.load("Housing.npy")
    xset = housing.T[0]
    y = housing.T[1]
    #Create vandermonde matrices of varying degree
    thrd_deg = np.vander(xset, 4)
    sxth_deg = np.vander(xset, 7)
    nnth_deg = np.vander(xset, 10)
    twlv_deg = np.vander(xset, 13)

    #Get least squares solutions from the vandermonde matrices
    thrd_fit = la.lstsq(thrd_deg, y)[0]
    sxth_fit = la.lstsq(sxth_deg, y)[0]
    nnth_fit = la.lstsq(nnth_deg, y)[0]
    twlv_fit = la.lstsq(twlv_deg, y)[0]

    #Domain to graph over
    x = np.linspace(0, 16, 100)

    #Plot 3rd degree polynomial vs raw data
    ax1 = plt.subplot(221)
    ax1.set_title("Third degree fit")
    ax1.plot(x, thrd_fit[0]*(x**3) + thrd_fit[1]*(x**2) + thrd_fit[2]*(x) + thrd_fit[3], 'g-')
    ax1.plot(housing.T[0], housing.T[1], 'g*')

    #Plot 6th degree polynomial vs raw data
    ax2 = plt.subplot(222)
    ax2.set_title("Sixth degree fit")
    ax2.plot(x, sxth_fit[0]*(x**6) + sxth_fit[1]*(x**5) + sxth_fit[2]*(x**4) + sxth_fit[3]*(x**3) + sxth_fit[4]*(x**2) + sxth_fit[5]*(x) + sxth_fit[6], 'g-')
    ax2.plot(housing.T[0], housing.T[1], 'g*')

    #Plot 9th degree polynomial vs raw data
    ax3 = plt.subplot(223)
    ax3.set_title("Ninth degree fit")
    ax3.plot(x, nnth_fit[0]*(x**9) + nnth_fit[1]*(x**8) + nnth_fit[2]*(x**7) + nnth_fit[3]*(x**6) + nnth_fit[4]*(x**5) + nnth_fit[5]*(x**4) + nnth_fit[6]*(x**3) + nnth_fit[7]*(x**2) + nnth_fit[8]*(x) + nnth_fit[9], 'g-')
    ax3.plot(housing.T[0], housing.T[1], 'g*')

    #Plot 12th degree polynomial vs raw data
    ax3 = plt.subplot(224)
    ax3.set_title("Twelfth degree fit")
    ax3.plot(x, twlv_fit[0]*(x**12) + twlv_fit[1]*(x**11) + twlv_fit[2]*(x**10) + twlv_fit[3]*(x**9) + twlv_fit[4]*(x**8) + twlv_fit[5]*(x**7) + twlv_fit[6]*(x**6) + twlv_fit[7]*(x**5) + twlv_fit[8]*(x**4) + twlv_fit[9]*(x**3) + twlv_fit[10]*(x**2) + twlv_fit[11]*(x) + twlv_fit[12], 'g-')
    ax3.plot(housing.T[0], housing.T[1], 'g*')

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    #Load data from ellipse, get x and y vals
    ellipse = np.load("ellipse.npy")
    x = ellipse[:,0]
    y = ellipse[:,1]

    #Create columns of A matrix
    x_square = x * x
    x_y = x * y
    y_square = y * y

    #Stack columns into full matrix
    A = np.column_stack((x_square, x, x_y, y, y_square))
    b = np.ones(len(x))

    #Get least squares solution
    solution = la.lstsq(A, b)[0]

    #Plot regression vs raw data
    plot_ellipse(solution[0], solution[1], solution[2], solution[3], solution[4])
    plt.plot(x, y, 'g*')
    plt.show()

def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #Get data from A and create initial two vectors
    m,n = np.shape(A)
    x_naught = np.random.random(n)
    x_next = np.zeros(n)

    #Start loop for creating convergent series
    for k in range(N):
        #Get x_k+1, normalize it
        x_next = A @ x_naught
        x_next = x_next / la.norm(x_next)
        #break if less than tolerance
        if la.norm(x_next - x_naught) < tol:
            break
        x_naught = x_next

    return x_next.T @ A @ x_next, x_next

def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    #Get data from A, put in hessenberg form
    m,n = np.shape(A)
    S = la.hessenberg(A)

    #Get the QR decomposition of Ak.
    #Recombine Rk and Qk into Ak+1.
    for k in range(N):
        Q, R = la.qr(S)
        S = R @ Q

    #Initialize empty vector of eigenvals
    eigs = []

    i = 0
    while i < n:
        #For S a 1x1 matrix, append s_i
        if i == n -1:
            eigs.append(S[i,i])
        elif abs(S[i+1,i]) < tol:
            eigs.append(S[i,i])

        #For S a 2x2 matrix, find eigenvals and append
        else:
            #Use quadratic formula
            a = 1
            b = S[i,i] + S[i+1,i+1]
            c = S[i,i] * S[i+1,i+1] - S[i+1,i] * S[i,i+1]
            eig_one = (b + cmath.sqrt(b**2 - 4*a*c)) / (2*a)
            eig_two = (b - cmath.sqrt(b**2 - 4*a*c)) / (2*a)
            eigs.append(eig_one)
            eigs.append(eig_two)
            i += 1
        i += 1

    return eigs
