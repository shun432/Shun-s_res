from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math
import matplotlib.pyplot as plt

# Estimation parameter of EKF
Q = np.diag([0.1, 0.1, 0.1, math.radians(1.0), math.radians(1.0), 1.0])**2 #~~~~~~~~~~~~~~~~~~~~~~~~
R = np.diag([1.0, 1.0, 1.0])**2

# #  Simulation parameter
Qsim = np.diag([1.0, 1.0, 1.0])**2                  #noise GPS
Rsim = np.diag([1.0, 1.0, 1.0, math.radians(10.0), math.radians(10.0), math.radians(10.0)])**2        #noise IMU

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True

vmat = np.matrix(np.zeros((6, 1)))


#   シミュレーションのパラメータをここに入力
def calc_input():

    a_x, a_y, a_z = 0, 0, 0

    v_x, v_y, v_z = 1.0, 1.0, 0
    rolrate, pitrate, yawrate = 1.0, 0, 0


    if (rolrate >= 0 and pitrate >= 0) or (rolrate < 0 and pitrate < 0):
        thetarate = math.sqrt(rolrate**2 + pitrate**2)
    else:
        thetarate = -math.sqrt(rolrate ** 2 + pitrate ** 2)

    u = np.matrix([v_x, v_y, v_z, thetarate, yawrate]).T
    return u, rolrate, pitrate

def a_to_t(a, v):
    t = a / 2 * (DT ** 2) + v * DT
    return t

def wrot_to_rot(wrot):
    rot = wrot * DT
    return rot


#   真値にガウスノイズを加えて観測値を模擬的に作る
def observation(xTrue, xd, u, w_rol, w_pit, R_pos, R_pos_ob):

    xTrue = motion_model(xTrue, u, R_pos)

    # 真値にノイズを加えて観測されたGPS位置を作る   add noise to gps x-y (SLAM)
    zx = xTrue[0, 0] + np.random.randn() * Qsim[0, 0]
    zy = xTrue[1, 0] + np.random.randn() * Qsim[1, 1]
    zz = xTrue[2, 0] + np.random.randn() * Qsim[2, 2]
    z = np.matrix([zx, zy, zz])

    # IMUの速度ベクトルにノイズを加える    add noise to input (IMU)
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]  #verocity x
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]  #verocity y
    ud3 = u[2, 0] + np.random.randn() * Rsim[2, 2]  #verocity z

    ud5 = u[4, 0] + np.random.randn() * Rsim[3, 3]  #yaw
    w_rol = w_rol + np.random.randn() * Rsim[4, 4]  #rol
    w_pit = w_pit + np.random.randn() * Rsim[5, 5]  #pit


    if (w_rol >= 0 and w_pit >= 0) or (w_rol < 0 and w_pit < 0):   #theta (include rol pit)
        ud4 = math.sqrt(w_rol ** 2 + w_pit ** 2)
    else:
        ud4 = -math.sqrt(w_rol ** 2 + w_pit ** 2)

    ud = np.matrix([ud1, ud2, ud3, ud4, ud5]).T

    xd = motion_model(xd, ud, R_pos_ob)

    return xTrue, z, xd, ud, w_rol, w_pit


#   三次元用の「現在の位置　＝　直前の位置　＋　回転＊微小変化」とした運動モデル
def motion_model(x, u, R_pos):

    vmat[5, 0] = math.sqrt(u[0, 0]**2 + u[1, 0]**2 + u[2, 0]**2)

    for i in range(3):
        u[i, 0] = a_to_t(0, u[i, 0])

    F = np.matrix([[1.0, 0, 0, 0, 0, 0],
                   [0, 1.0, 0, 0, 0, 0],
                   [0, 0, 1.0, 0, 0, 0],
                   [0, 0, 0, 1.0, 0, 0],
                   [0, 0, 0, 0, 1.0, 0],
                   [0, 0, 0, 0, 0, 0]])

    B = np.matrix([[R_pos[0, 0], R_pos[0, 1], R_pos[0, 2], 0, 0, 0],
                   [R_pos[1, 0], R_pos[1, 1], R_pos[1, 2], 0, 0, 0],
                   [R_pos[2, 0], R_pos[2, 1], R_pos[2, 2], 0, 0, 0],
                   [0, 0, 0, DT, 0, 0],
                   [0, 0, 0, 0, DT, 0]])    #??????????

    x = F * x + (u.T * B).T + vmat

    return x


#   行列の大きさを変える（位置情報のみにする）
def observation_model(x):
    #  Observation Model
    H = np.matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    z = H * x

    return z


#   三次元ヤコビ行列
def jacobF(x, vmat):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    theta = x[3, 0]
    yaw = x[4, 0]
    v = vmat[5, 0]
    jF = np.matrix([[1.0, 0, 0, v*DT*math.cos(theta)*math.cos(yaw), -v*DT*math.sin(theta)*math.sin(yaw), math.sin(theta)*math.cos(yaw)],
                   [0, 1.0, 0, v*DT*math.sin(theta)*math.cos(yaw), v*DT*math.cos(theta)*math.sin(yaw), math.sin(theta)*math.sin(yaw)],
                   [0, 0, 1.0, 0.0, math.cos(theta), -v*DT*math.sin(theta)],
                   [0, 0, 0, 1.0, 0, 0],
                   [0, 0, 0, 0, 1.0, 0],
                   [0, 0, 0, 0, 0, 1.0]])

    return jF


#   行列の大きさを変える（位置情報のみにする）
def jacobH(x):
    # Jacobian of Observation Model
    jH = np.matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    return jH


#   位置推定モデル
def ekf_estimation(xEst, PEst, z, u, R_pos):

    #  Predict
    xPred = motion_model(xEst, u, R_pos)


    #######################
    jF = jacobF(xPred, vmat)
    PPred = jF * PEst * jF.T + Q

    #  Update

    ######################
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH * PPred * jH.T + R
    K = PPred * jH.T * np.linalg.inv(S)
    xEst = xPred + K * y
    PEst = (np.eye(len(xEst)) - K * jH) * PPred

    return xEst, PEst


#   メインループ、座標表示
def main():

    fig = plt.figure()
    ax = Axes3D(fig)

    R_pos = np.matrix([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    R_pos_ob = np.matrix([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y z theta yaw v]'
    xEst = np.matrix(np.zeros((6, 1)))
    xTrue = np.matrix(np.zeros((6, 1)))
    PEst = np.eye(6)

    xDR = np.matrix(np.zeros((6, 1)))  # Dead reckoning

    # # history
    # hxEst = xEst
    # hxTrue = xTrue
    # hxDR = xTrue
    # hz = np.zeros((1, 3))


    #   座標表示用の配列
    hxEst_x = []
    hxEst_y = []
    hxEst_z = []
    hxTrue_x = []
    hxTrue_y = []
    hxTrue_z = []
    hxDR_x = []
    hxDR_y = []
    hxDR_z = []
    hz_x = []
    hz_y = []
    hz_z = []

    hxEst_x.append(xEst[0, 0])
    hxEst_y.append(xEst[1, 0])
    hxEst_z.append(xEst[2, 0])
    hxDR_x.append(xDR[0, 0])
    hxDR_y.append(xDR[1, 0])
    hxDR_z.append(xDR[2, 0])
    hxTrue_x.append(xTrue[0, 0])
    hxTrue_y.append(xTrue[1, 0])
    hxTrue_z.append(xTrue[2, 0])
    hz_x.append(0)
    hz_y.append(0)
    hz_z.append(0)

    while SIM_TIME >= time:
        time += DT
        u, w_rol, w_pit = calc_input()

        xTrue, z, xDR, ud, w_rol_ob, w_pit_ob= observation(xTrue, xDR, u, w_rol, w_pit, R_pos, R_pos_ob)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud, R_pos_ob)

        rol = wrot_to_rot(w_rol)
        pit = wrot_to_rot(w_pit)
        w_yaw = u[4, 0]
        yaw = wrot_to_rot(w_yaw)

        rol_ob = wrot_to_rot(w_rol_ob)
        pit_ob = wrot_to_rot(w_pit_ob)
        w_yaw_ob = ud[4, 0]
        yaw_ob = wrot_to_rot(w_yaw_ob)


        #   ロール、ピッチ、ヨー角を回転行列に変える
        R = np.matrix([[math.cos(pit) * math.cos(yaw), -math.cos(pit) * math.sin(yaw), math.sin(pit)],
                       [math.cos(rol) * math.sin(yaw) + math.sin(rol) * math.sin(pit) * math.cos(yaw),
                        math.cos(rol) * math.cos(yaw) - math.sin(rol) * math.sin(pit) * math.sin(yaw),
                        - math.sin(rol) * math.cos(pit)],
                       [math.sin(rol) * math.sin(yaw) - math.cos(rol) * math.sin(pit) * math.cos(yaw),
                        math.sin(rol) * math.cos(yaw) + math.cos(rol) * math.sin(pit) * math.sin(yaw),
                        math.cos(rol) * math.cos(pit)]])

        R_ob = np.matrix([[math.cos(pit_ob) * math.cos(yaw_ob), -math.cos(pit_ob) * math.sin(yaw_ob), math.sin(pit_ob)],
                       [math.cos(rol_ob) * math.sin(yaw_ob) + math.sin(rol_ob) * math.sin(pit_ob) * math.cos(yaw_ob),
                        math.cos(rol_ob) * math.cos(yaw_ob) - math.sin(rol_ob) * math.sin(pit_ob) * math.sin(yaw_ob),
                        - math.sin(rol_ob) * math.cos(pit_ob)],
                       [math.sin(rol_ob) * math.sin(yaw_ob) - math.cos(rol_ob) * math.sin(pit_ob) * math.cos(yaw_ob),
                        math.sin(rol_ob) * math.cos(yaw_ob) + math.cos(rol_ob) * math.sin(pit_ob) * math.sin(yaw_ob),
                        math.cos(rol_ob) * math.cos(pit_ob)]])

        R_pos = R * R_pos

        R_pos_ob = R_ob * R_pos_ob


        hxEst_x.append(xEst[0, 0])
        hxEst_y.append(xEst[1, 0])
        hxEst_z.append(xEst[2, 0])
        hxDR_x.append(xDR[0, 0])
        hxDR_y.append(xDR[1, 0])
        hxDR_z.append(xDR[2, 0])
        hxTrue_x.append(xTrue[0, 0])
        hxTrue_y.append(xTrue[1, 0])
        hxTrue_z.append(xTrue[2, 0])
        hz_x.append(z[0, 0])
        hz_y.append(z[0, 1])
        hz_z.append(z[0, 2])

        if show_animation:

            plt.cla()

            plt.title("t = " + str(int(time)) + " / " + str(int(SIM_TIME)))

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.scatter(hz_x, hz_y, hz_z, color="gray", s=10, label="observed GPS(SLAM)")
            plt.plot(hxTrue_x, hxTrue_y, hxTrue_z, color="green", marker="+", label="Truth")
            plt.plot(hxDR_x, hxDR_y, hxDR_z, color="red", marker="", label="observed IMU")
            plt.plot(hxEst_x, hxEst_y, hxEst_z, color="blue", marker="", label="estimated")

            # plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.legend(loc='upper left', borderaxespad=0)
            plt.grid(True)
            plt.pause(0.001)


    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()