import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

field_height = 400
field_width = 600
field_length = 600
lmbda = np.arange(1, -1, -2/field_width)

'''Reused Open Source code'''
# refrence: https://github.com/ymalitsky/primal-dual-linesearch/blob/master/pd_algorithms.py
def pd(J, prox_g, prox_f_conj, K,  x0, y0, sigma, tau, numb_iter=100):
    """
    Primal-dual algorithm of Pock and Chambolle for the problem min_x
    max_y [<Kx,y> + g(x) - f*(y)]
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    begin = clock()  # process_time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    time_list = [process_time() - begin]
    for i in range(numb_iter):
        x1 = prox_g(x - tau * K.T.dot(y), tau)
        z = x1 + theta * (x1 - x)
        y = prox_f_conj(y + sigma * K.dot(z), sigma)
        x = x1
        values.append(J(x, y))
        # time_list.append(process_time() - begin)
        time_list.append(clock() - begin)
    end = clock()  # process_time()
    print("----- Primal-dual method -----")
    print("Time execution:", end - begin)
    return [time_list, values, x, y]

# reference: https://github.com/ymalitsky/primal-dual-linesearch/blob/master/opt_operators.py
def proj_ball(x, center, rad):
    """
    Compute projection of x onto the closed ball B(center, rad)
    """
    d = x - center
    dist = LA.norm(d)
    if dist <= rad:
        return x
    else:
        return rad * d / dist + center


def prox_norm_1(x, eps, u=0):
    """
    Find proximal operator of function eps*||x-u||_1
    """
    x1 = x + np.clip(u - x, -eps, eps)
    return x1


def prox_norm_2(x, eps, a=0):
    """
    Find proximal operator of function f = 0.5*eps*||x-a||**2,
    """
    return (x + eps * a) / (eps + 1)


def project_nd(x, r=1):
    '''perform a pixel-wise projection onto r-radius balls. Here r = 1'''
    norm_x = np.sqrt(
        (x * x).sum(-1))  # Calculate L2 norm over the last array dimension
    nx = np.maximum(1.0, norm_x / r)
    return x / nx[..., np.newaxis]


'''*************************************************************************************'''
'''ALL THE CODE BELOW THIS HAS BEEN WRITTEN COMPLETELY BY ME REFERENCING THE DOCUMENTAION ONLY. NO PART OF IT IS COPIED FROM SOMEONE ELSE'S WORK'''

class voxel:
    def __init__(self):
        self.h = np.zeros((1, len(lmbda)), dtype = np.uint16)
        # self.d = 0
        self.u = 0
        self.p = np.zeros((3, 1), dtype = np.float32)    

def read_data(index):
    path_to_data = "./data/"
    data_list = os.listdir(path_to_data)
    print(data_list)

    # 0, 1, 2, 3, 4, 5, 6 based on the specific dataset to work on
    data_index = index

    image_path = path_to_data + data_list[data_index] + "/images/"
    image_names = sorted(os.listdir(image_path))

    images = []
    for filename in image_names:
        images.append(cv2.imread(image_path + filename))
    
    param_file = glob.glob(path_to_data + data_list[data_index] + "/*_par.txt")

    if len(param_file) == 0:
        return images, [], []

    with open(param_file[0]) as f:
        lines = f.readlines()

    camera_intrinsic_matrices = []
    Transformation_matrices = []
    for i in range(len(lines)):
        if i == 0:
            continue
        lines[i] = lines[i].strip("\n")
        data_bundle = lines[i].split(" ")    
        intrinsic = [[], [], []]
        T = np.zeros((4, 4))
        R = [[], [], []]
        t = []
        for i, data in enumerate(data_bundle):
            if i == 0:
                continue
            elif i < 10:
                intrinsic[int((i - 1) / 3)].append(float(data))
            elif i < 19:
                R[int((i - 10) / 3)].append(float(data))
            else:
                t.append(float(data))
        intrinsic = np.array(intrinsic)
        camera_intrinsic_matrices.append(intrinsic)
        t = np.reshape(t, (3, 1))
        T[:3, :3] = R
        T[:3, 3] = t[:3, 0]
        T[3, 3] = 1
        Transformation_matrices.append(T)
    # print(len(images))
    # print(len(camera_intrinsic_matrices))
    # print(len(Transformation_matrices))

    return images, camera_intrinsic_matrices, Transformation_matrices



def drawEpilines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def detect_and_match_features(img1, img2, detector, matcher):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    matches = matcher.knnMatch(des1, des2, k = 2)
    
    return kp1, des1, kp2, des2, matches

def lowes_ratio_test(matches, kp1, kp2, ratio):
    pts1 = []
    pts2 = []

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            matchesMask[i]=[1,0]
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return matchesMask, pts1, pts2

def create_depth_from_disparity(disparity_map, K, T):
    h, w = disparity_map.shape[:2]
    depth = np.zeros((h, w))
    depth_normalized = np.zeros((h, w))
    f = K[0, 0]
    # print(f)
    # print("T: ", T)
    b = np.linalg.norm(T[:3, 3])
    # print(b)
    # scaling = 100
    for i in range(h):
        for j in range(w):
            if (disparity_map[i, j] > 0):
                depth[i, j] =  b * f / (disparity_map[i, j])

                # print(depth[i, j])
            else:
                depth[i, j] = 0
    # min_depth = np.min(depth)
    # max_depth = np.max(depth)
    # for i in range(h):
    #     for j in range(w):
    #         depth_normalized[i, j] = (255) * depth[i, j] / (max_depth - min_depth)
    
    # plt.imshow(depth_normalized), plt.show()
    return depth

def create_distance_field(points, cam_posn):
    print(len(points[0]))
    print(cam_posn)    
    
    volume_field = [[[voxel()for k in range(field_height)] for j in range(field_length)] for i in range(field_width)]
    volume_field[:][:][:].h = np.array((1, len(lmbda)))

    points[0, :] = 50 + points[0, :]
    points[1, :] = 50 + points[1, :]
    points[2, :] = 300 + points[2, :]
    cam_posn[0, 1] = 50 + cam_posn[0, 1]
    cam_posn[1, 1] = 50 + cam_posn[1, 1]
    
    c = cam_posn

    for i in range(len(points[0])):
        g = np.reshape(np.array([points[0, i], points[1, i], points[2, i]]), (3, 1))
        g = np.reshape(points[:, i], (3, 1))
        # print(g)
        # print("norm: ", np.linalg.norm(g))
        if np.linalg.norm(g) > 800:
            continue
        # g = np.vstack()
        # g = points[:, i]
        for k, l in enumerate(lmbda):
            pt = g + l * (g - c)
            # print(g)
            # print(c)
            # print(pt)
            if np.any(pt[:2] > field_length) or pt[2] > field_height:
                continue
            volume_field[int(pt[0])][int(pt[1])][int(pt[2])].u = l
            volume_field[int(pt[0])][int(pt[1])][int(pt[2])].h[k] += 1
    
    return volume_field

def plot_3D_points(x_pts, y_pts, z_pts):
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot3D(x_pts, y_pts, z_pts, 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()