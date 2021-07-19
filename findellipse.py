import numpy as np
import cv2
import math

# The only function in here meant to be called externally is:
#
# find_ellipse(C, img, debug=False)
#
# See the comments below for what it does.
#author Chris Monico



################################################################
def filter_contour_points(C, cx, cy, max_deviation_from_orthogonal):
    # Given a contour which should be approximately
    # a circle centered near (cx, cy), filter out some
    # points which are obviously noise; specifically,
    # points on the contour where the path is not locally
    # close to being orthogonal to the vector
    # from the center to that portion of the path.
    # This is crude, but reasonably fast.
    # max_deviation_from_orthogonal is the largest deviation from
    # orthogonal (in radians) that we will accept.
    N = len(C)
    chunk_size = 10
    SS = []
    j=0
    cutoff = math.cos(max_deviation_from_orthogonal - 3.1415926535/2)
    
    while j<N-chunk_size:
        P1 = C[j]
        P2 = C[j+chunk_size-1]
        # Find the angle between:
        # (i) the vector from <cx,cy> to P1 and
        # (ii) the vector from P1 to P2:
        v1 = (P1[0][0] - cx, P1[0][1] - cy)
        v2 = (P2[0][0] - P1[0][0], P2[0][1] - P1[0][1])
        n1 = (v1[0]*v1[0] + v1[1]*v1[1])
        n2 = (v2[0]*v2[0] + v2[1]*v2[1])
        if abs(n1*n2)>1e-10:
            costheta = (v1[0]*v2[0] + v1[1]*v2[1])/( (n1*n2)**(1/2))
            if abs(costheta) < cutoff:
                # Close enough to keep these.
                #SS += C[j:j+chunk_size]
                for k in range(chunk_size):
                    SS.append(C[j+k])
        j += chunk_size

    N1 = len(SS)
    NewC = np.array(SS)
    NewC = NewC.reshape((N1,1,2))
    return NewC
        
    
##################################################
def residuals(ellipse, samples):
    # Given an ellipse and a collection of points, find the squared distance
    # from each point to the ellipse.
    # Return: mean_err, E
    # where mean_err is the average squared distance from a point to the ellipse
    # and E is a dictionary: E[i] = distance(samples[i], ellipse)^2.
    center = ellipse[0]
    dims = ellipse[1]
    angle = ellipse[2]
    cx = center[0]
    cy = center[1]
    width = dims[0]
    height = dims[1]
    res = 0
    E = {}
    N = len(samples)
    for i in range(N):
        # Borrowed from StackExchange, because I didn't feel like working it out myself:
        px = (samples[i][0][0] - cx) * math.cos(-angle) - (samples[i][0][1]- cy) * math.sin(-angle)
        py = (samples[i][0][0] - cx) * math.sin(-angle) + (samples[i][0][1] - cy) * math.cos(-angle)
        err = abs((px/width)**2 + (py/height)**2 - 0.25)
        res += err
        E[i] = err
    return res/N, E

####################################################################
def try_ellipse_it(C, img=None):
    # Given a contour Contour, try to fit an ellipse to it.
    # We will do the following:
    # (1) Fit an ellipse to the contour,
    # (2) Remove a percentage of the points which are farthest from the ellipse.
    # (3) Repeat (1) and (2), until we hit a lower threshold for the number of points remaining.

    SS = C.copy()
    j=1
    # Here, 200 is arbitrary. One should probably calculate something
    # reasonable based on the dimensions of the image instead.
    while len(SS)>200:
        N = len(SS)
        ell = cv2.fitEllipse(SS)

        if not(img is None):
            tmp_img = img.copy()
            cv2.ellipse(tmp_img, ell, (0,255,0), 2)
            cv2.drawContours(tmp_img, SS, -1, (255,0,0), 2)
            cv2.imshow('Pass '+str(j)+', len(SS)='+str(len(SS)), tmp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            j += 1


        res, E = residuals(ell, SS)
        E_sorted = [k for k,v in sorted(E.items(), key=lambda item:item[1])]
        # Now discard the 5% of points with largest error - 0.95 CAN BE CHANGED
        N2 = int(N*0.93) 
        subsamples = [SS[E_sorted[j]] for j in range(N2)]

        npsamp = np.array(subsamples)
        SS = npsamp.reshape((N2,1,2))


    ell = cv2.fitEllipse(SS)

    res, E = residuals(ell, subsamples)
    #print("residual = {0}".format(res))
    return ell

################################################################
def find_ellipse(C, img, debug=False):
    # Given a contour which is assumed to be an ellipse plus some noise,
    # attempt to determine the ellipse.
    # Return value: the ellipse, on success, or None on error.

    #SS = C.copy()
    dims = img.shape
    h = dims[0]
    w = dims[1]
    if len(C) < 600:
        #400 - CAN BE CHANGED
        # There's very little chance this will work.
        return None
    SS = filter_contour_points(C, w//2, h//2, 3.0*2*math.pi/360)
    #CAN BE CHANGED
    
    N = len(SS)
    if N < 500:
        #300 - CAN BE CHANGED
        return None
    if debug:
        tmp_img = img.copy()
        cv2.drawContours(tmp_img, SS, -1, (255,0,0), 2)
        cv2.imshow('After filter_contour_points', tmp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ell = try_ellipse_it(SS, img)
    else:
        ell = try_ellipse_it(SS)
    return ell

    






