import numpy as np
import general_robotics_toolbox as rox
import general_robotics_toolbox_invkin as rox_ik

# execute the main function
if __name__ == "__main__" :
    #Choose qd for FK
    #Jacobian:
    #   generate initial guess: qguess = qdesired + error (small)
    #   Run qguess through IK to get qcomputed
    #   Run qcomputed through FK to get Pcomputed and Rcomputed (compare with Pdesired and Rdesired)
    #   While qdesired != qmeasured
    #       Send to arm and measure: qcomputed = qmeasured
    #       if not, run qm through FK to get Pm and Rm, compare again

    # Setup
    robot = rox.DofBot()
    qc = np.zeros(5)  # Computed angles
    qd = [30, 60, 90, 120, 30] # Desired angles
    
    for i in range(5):
        qc[i] = input("Enter angle for joint " + str(i+1) + ": ")

    # Guessed/desired angle's rotations/positions
    mc = rox.fwdkin(robot, qc)  # Holds 4x3 matrix of Rc and Pc
    md = rox.fwdkin(robot, qd)
    Pc = mc[0:3, 3]
    Rc = mc[0:3, 0:3]
    Pd = md[0:3, 3]
    Rd = md[0:3, 0:3]

    # Compute errors between Pd and Pc, and Rd and Rc
    ep = 0.5*np.linalg.norm(Pc - Pd)**2 # Position error
    Er = np.dot(Rc, np.transpose(Rd)) # Rotation error matrix

    # Compute theta and k from Er
    theta, k = np.rotation_matrix_to_axis_angle(Er)

    # Compute rotation error metrics
    er = [4*np.sin(theta/2)**2, 8*np.sin(theta/4)**2, 
        theta*theta, 0.5*np.linalg.norm(Er)**2]
    
    J = rox.robotjacobian(robot, qc) # Jacobian matrix
    b = 0.001 # Step size

    # Procedure
    while(er[0] > 0.01 and ep > 0.01):
        s = 4*np.sin(theta/2)**2
        U = np.dot(J, np.array([[b * np.transpose(s)], [Pc - Pd]]))
        qc = qc - U
        J = rox.robotjacobian(robot, qc)
        mc = rox.fwdkin(robot, qc)
        md = rox.fwdkin(robot, qd)
        Pc = mc[0:3, 3]
        Rc = mc[0:3, 0:3]
        ep = 0.5*np.linalg.norm(Pc - Pd)**2
        Er = np.dot(Rc, np.transpose(Rd))
        theta, k = np.rotation_matrix_to_axis_angle(Er) # --------------
        er = [4*np.sin(theta/2)**2, 8*np.sin(theta/4)**2, 
            theta*theta, 0.5*np.linalg.norm(Er)**2]
    
    return qc