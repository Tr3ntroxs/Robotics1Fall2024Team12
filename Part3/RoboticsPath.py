import time #import the time module. Used for adding pauses during operation
from Arm_Lib import Arm_Device #import the module associated with the arm
from scipy.spatial.transform import Rotation as R
#from Part1_FK import fk_Dofbot


import general_robotics_toolbox as rox
import math
import numpy as np
# from general_robotics_toolbox_invkin import iterative_invkin as inv

from qpsolvers import solve_qp


Arm = Arm_Device() # Get DOFBOT object
time.sleep(.2) #this pauses execution for the given number of seconds


def main():
    speedtime = 100 #time in milliseconds to reach desired joint position
    
    l0 = 0.061 # base to servo 1
    l1 = 0.0435 # servo 1 to servo 2
    l2 = 0.08285 # servo 2 to servo 3
    l3 = 0.08285 # servo 3 to servo 4
    l4 = 0.07385 # servo 4 to servo 5
    l5 = 0.05457 # servo 5 to gripper
    
    joint_type = [0, 0, 0, 0, 0]
    
    H = np.zeros((3, 5))  # 3 rows (x, y, z directions) and 5 columns for 5 joints


    # Define the axes of rotation for each joint
    H[:, 0] = np.array([0, 0, 1])  # Joint 1 rotates about the z-axis (R01 is a rotz rotation)
    H[:, 1] = np.array([0, 1, 0])  # Joint 2 rotates about the y-axis (R12 is a roty rotation)
    H[:, 2] = np.array([0, 1, 0])  # Joint 3 rotates about the y-axis (R23 is a roty rotation)
    H[:, 3] = np.array([0, 1, 0])  # Joint 4 rotates about the y-axis (R34 is a roty rotation)
    H[:, 4] = np.array([1, 0, 0])  # Joint 5 rotates about the x-axis (R45 is a rotx rotation)
    
    P = np.zeros((3, 6))  # 3 rows (x, y, z) and 6 columns (5 joints + end-effector)


    # Define the position vectors for each joint
    P[:, 0] = (l0 + l1) * np.array([0, 0, 1])  # P01: base to frame 1 (along z-axis)
    P[:, 1] = np.array([0, 0, 0])              # P12: frame 1 to frame 2 (no translation)
    P[:, 2] = l2 * np.array([1, 0, 0])         # P23: frame 2 to frame 3 (along x-axis)
    P[:, 3] = -l3 * np.array([0, 0, 1])        # P34: frame 3 to frame 4 (along -z-axis)
    P[:, 4] = np.array([0, 0, 0])              # P45: frame 4 to frame 5 (no translation)
    P[:, 5] = -(l4 + l5) * np.array([1, 0, 0]) # P5T: frame 5 to tool (along -x-axis)
    
    #print(f"H Matrix: {H}, P Matrix: {P}")
    
    joint_lower_limit = np.array([0, 0, 0, 0, 0]) # Joints 0-5 lower limit is 0 radians
    joint_upper_limit = np.array([180, 180, 180, 180, 270]) 
    
    dofbot = rox.Robot(H, P, joint_type, joint_lower_limit, joint_upper_limit, joint_vel_limit = None, joint_acc_limit = None, M = None, \
                 R_tool=None, p_tool=None, joint_names = None, root_link_name = None, tip_link_name = None, T_flange = None, T_base = None) 
    
    conv = math.pi/180
    q0 = np.array([0,45*conv,135*conv,45*conv,135*conv])
    
    R0,P0 = fk_Dofbot(q0)
    R0Td = R0
    P0Td = P0-np.array([0,0,0.05])
    N = 100
    epsilon_r = 0.1
    epsilon_p = 0.1
    q_prime_min = np.full(5,-np.inf)
    q_prime_max = np.full(5,np.inf)
    q_lambda,lamb,P0T_lambda,R0T_lambda = path_plan(dofbot,q0,P0Td,R0Td,epsilon_r,epsilon_p,q_prime_min,q_prime_max,N)
    
    # Run reference solution
    q,l,P,R = path_plan(dofbot,q0,P0Td,R0Td,epsilon_r,epsilon_p,q_prime_min,q_prime_max,N)
    
    assert(norm(q_lambda[:,end]-q[:,end])<1e-8)


def path_plan(robot,q0,P0Td,R0Td,epsilon_r,epsilon_p,q_prime_min,q_prime_max,N):
    
    # Set-Up
    n = len(q0)
    lmbd = np.arange(0, 1, 1/N)
    
    R0T0, P0T0 = fk_Dofbot(q0)
    
    # Compute path in Task Space
    ER0 = R0T0.as_matrix()*np.transpose(R0Td.as_matrix())
    temp_k, temp_theta = rox.R2rot(ER0)
    k_hat = np.array([temp_k]).T  # Solution provided by copilot, check here in case
    theta0 = temp_theta
    Euldes_lmbd = np.zeros((3, len(lmbd)))
    Pdes_lmbd = np.zeros((3, len(lmbd)))
    dP0T_dlmbd = (P0Td - P0T0)
    der_dlmbd = np.zeros((1, len(lmbd)))


    for k in range(len(lmbd)):
        theta = (1-lmbd[k])*theta0
        Robj = R.from_rotvec(k_hat.flatten() * theta)
        Euldes_lmbd[:,k] = np.flip(Robj.as_euler("zxy", True)) # Solution provided by copilot, check here in case
        Pdes_lmbd[:,k] = (1-lmbd[k])*P0T0 + lmbd[k]*P0Td
    
    
    # Solve QP Problem, generate joint space path
    q_prime = np.zeros((n,len(lmbd)))
    q_lmbd = np.zeros((n,len(lmbd)))
    q_lmbd[:,0] = q0
    exitflag = np.zeros((1,len(lmbd)))
    P0T_lmbd = np.zeros((3,len(lmbd)))
    R0T_lmbd = np.zeros((3,3,len(lmbd)))
    P0T_lmbd[:,0] = P0T0
    R0T_lmbd[:,:,0] = R0T0.as_matrix()
    Eul_lmbd = np.zeros((3,len(lmbd)))
    Eul_lmbd[:,0] = np.flip(R.from_matrix(R0T_lmbd[:,:,0]).as_euler("zxy", True))
    qprev = q0
    
    for k in range(len(lmbd)):
        qlimit = np.array([robot.joint_lower_limit, robot.joint_upper_limit]).T
        lb, ub = qprimelimits_full(qlimit,qprev,N,q_prime_max,q_prime_min)
        J = rox.robotjacobian(robot, qprev)
        
        vt = dP0T_dlmbd
        vr = der_dlmbd[k]*k_hat
        H = getqp_H(qprev, J, vr, vt, epsilon_r, epsilon_p)
        f = getqp_f(qprev, epsilon_r, epsilon_p )
        
        q_prime_temp = solve_qp(H,f,[],[],[],[],lb,ub,[])
#         q_prime_temp, DONTCARE, exitflag(k) = solve_qp(H,f,[],[],[],[],lb,ub,[])
        q_prime_temp = q_prime_temp[1:n]
        # check exit flag - all elements should be 1
        if exitflag(k) != 1:
            print('Generation Error')
            return
        q_prime[0:,k] = q_prime_temp;


        qprev = qprev + (1/N)*q_prime_temp


        q_lmbd[0:,k+1] = qprev
        Rtemp, Ptemp = rox.fwdkin(robot,qprev)
        P0T_lmbd[0:,k+1] = Ptemp
        R0T_lmbd[0:,:,k+1] = Rtemp
        Eul_lmbd[0:,k+1] = np.flip((Rtemp).as_euler("zxy", True))
        
    # Chop off excess
    q_lmbd[0:,len(q_lmbd)] = []
    P0T_lmbd[0:,len(P0T_lmbd)] = []
    R0T_lmbd[0:,:,len(R0T_lmbd)] = []


def fk_Dofbot (q):
    # FWDKIN_DOFBOT Computes the end effector position and orientation relative to the base frame for Yahboom's Dofbot manipulator 
    #     using the product of exponentials approach
    # Input :
    # q: 5x1 vector of joint angles in degrees
    #
    # Output :
    # Rot: The 3x3 rotation matrix describing the relative orientation of the end effector frame to the base frame (R_ {0T})
    # Pot: The 3x1 vector describing the position of the end effector relative to the base, 
    #      where the first element is the position along the base frame x-axis,
    #      the second element is the position along the base frame y-axis,
    #      and the third element is the position along the base frame z- axis (P_ {0T})
    print(f"Q IN FK FUNCTION: {q}. SHAPE: {q.shape}")
    #set up the basis unit vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])


    # define the link lengths in meters
    l0 = 0.061 # base to servo 1
    l1 = 0.0435 # servo 1 to servo 2
    l2 = 0.08285 # servo 2 to servo 3
    l3 = 0.08285 # servo 3 to servo 4
    l4 = 0.07385 # servo 4 to servo 5
    l5 = 0.05457 # servo 5 to gripper


    #set up the rotation matrices between subsequent frames
    R01 = rotz(q[0]) # rotation between base frame and 1 frame
    R12 = roty(-q[1]) # rotation between 1 and 2 frames
    R23 = roty(-q[2]) # rotation between 2 and 3 frames
    R34 = roty(-q[3]) # rotation between 3 and 4 frames
    R45 = rotx(-q[4]) # rotation between 4 and 5 frames
    R5T = roty(0) #the tool frame is defined to be the same as frame 5


    #set up the position vectors between subsequent frames
    P01 = (l0+l1)*ez # translation between base frame and 1 frame in base frame
    P12 = np.zeros(3,) # translation between 1 and 2 frame in 1 frame
    P23 = l2*ex # translation between 2 and 3 frame in 2 frame
    P34 = -l3*ez # translation between 3 and 4 frame in 3 frame
    P45 = np.zeros(3,) # translation between 4 and 5 frame in 4 frame
    P5T = -(l4+l5)*ex # translation between 5 and tool frame in 5 frame


    # calculate Rot and Pot
    #Rot is a sequence of rotations
    Rot = R01*R12*R23*R34*R45*R5T
    #Pot is a combination of the position vectors. 
    #    Each vector must be represented in the base frame before addition. 
    #    This is achieved using the rotation matrices.
    Pot = P01 + R01.apply(P12 + R12.apply(P23 + R23.apply(P34 + R34.apply(P45 + R45.apply(P5T)))))


    return Rot, Pot












def rotx(theta):
    # return the principal axis rotation matrix for a rotation about the Xaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rx = R.from_euler('x',theta , degrees = True )
    return Rx


def roty(theta):
    # return the principal axis rotation matrix for a rotation about the Yaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Ry = R.from_euler('y',theta , degrees = True )
    return Ry


def rotz(theta):
    # return the principal axis rotation matrix for a rotation about the Zaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rz = R.from_euler('z',theta , degrees = True )
    return Rz


def qprimelimits_full(qlimit,qprev,N,qpmax,qpmin):
    n = len(qlimit)
    print(len(qlimit))
    # Compute limits due to joint stops
    lb_js = N*(qlimit[0:,0] - qprev)
    ub_js = N*(qlimit[0:,1] - qprev) # 1 and 2 replaced; assumed they might be matlab index error
    # Compare and find most restrictive bound
    lb = np.zeros(n+2) # (n+2,1) corrected
    ub = np.zeros(n+2)
    ub[n-1] = 1
    ub[n] = 1
    for k in range(n):
        if lb_js[k] > qpmin[k]:
            lb[k] = lb_js[k]
        else:
            lb[k] = qpmin[k]
           
        if ub_js[k] < qpmax[k]:
            ub[k] = ub_js[k]
        else:
            ub[k] = qpmax[k]
    return lb, ub


def getqp_H(dq, J, vr, vp, er, ep):
    n = len(dq)
    H1 = np.dot(np.transpose(np.hstack((J,np.zeros((6,2))))), np.array(np.hstack((J,np.zeros((6,2))))))
    H2 = np.dot(np.transpose(np.array([[np.zeros((3,n)),vr,np.zeros((3,1))], [np.zeros((3,n)),np.zeros((3,1)),vp]])), np.array([[np.zeros((3,n)),vr,np.zeros((3,1))], [np.zeros((3,n)),np.zeros((3,1)),vp]]))
    H3 = -2*[J,np.zeros(6,2)].T*[np.zeros(3,n),vr,np.zeros(3,1), np.zeros(3,n),np.zeros(3,1),vp]
    H3 = (H3+np.transpose(H3))/2
    H4 = np.transpose(np.array([[np.zeros(1,n),math.sqrt(er),0], [np.zeros(1,n),0,math.sqrt(ep)]])) * np.array([[np.zeros(1,n),math.sqrt(er),0], [np.zeros(1,n),0,math.sqrt(ep)]])
    H = 2*(H1+H2+H3+H4)
    return H

def getqp_f( dq, er, ep ):
    f = -2*[np.zeros(1,len(dq)),er,ep].T
    return f


#execute the main loop unless the stop button is pressed to stop the kernel 
try:
    main()
except KeyboardInterrupt:
    print("Program closed!")
    pass


del Arm # release the arm object


