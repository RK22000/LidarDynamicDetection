import numpy as np

def polar_to_cart(p):
    a,d = p
    return d*np.array([np.cos(a),np.sin(a)])
def cart_to_polar(p):
    x,y=p
    return np.array([np.arctan2(y,x), np.linalg.norm(p)])
def make_polar_points(n=150):
    r = np.random.randn(n)
    d0 = r*1+6
    d0 = np.convolve(d0, [1/(int(0.03*n))]*int(0.03*n), mode='same')
    a0 = np.linspace(0, 2*np.pi, n)

    points = np.stack([a0,d0], axis=1)
    points = np.array([polar_to_cart(p) for p in points])
    points *= (np.random.randn(2)+1)
    points = np.array([cart_to_polar(p) for p in points])

    return points

def heuristic_pos_theta_theta(points0, points1):
    '''Takes 2d list of cartesian points
    return position, theta, -theta%2pi
    '''
    pos0 = np.sum(points0, 0)/len(points0)
    cov = np.cov(points0.T)
    eval, evec = np.linalg.eig(cov)
    p0 = evec[:, np.argmax(eval)]
    t0 = np.arctan2(*p0[::-1]) + 3*np.pi/4

    pos1 = np.sum(points1, 0)/len(points1)
    cov = np.cov(points1.T)
    eval, evec = np.linalg.eig(cov)
    p1 = evec[:, np.argmax(eval)]
    t1 = np.arctan2(*p1[::-1]) + 3*np.pi/4

    pos = pos1-pos0
    delta_thetas = np.array([t1-t0, t1-t0+np.pi, t1-t0-np.pi])
    theta = delta_thetas[np.argmin(np.abs(delta_thetas))]
    return (pos, theta, -theta%(2*np.pi))
