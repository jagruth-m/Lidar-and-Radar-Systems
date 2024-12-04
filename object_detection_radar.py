import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # visualize point clouds
import cv2

record_number = 1  # Either record 1 or record 2

root = f"C:/Users/medav/OneDrive/Documents/LiDAR RaDAR/TASKS/Dataset2022/record{record_number}/"
path_blickfeld = f"{root}/blickfeld"
path_camera = f"{root}/camera"
path_groundtruth = f"{root}/groundtruth"
path_radar = f"{root}/radar"
path_velodyne = f"{root}/velodyne"

def readPC(path):
    """
       Will return a numpy array of
       shape:
       Nx4 for LiDAR Data (x,y,z,itensity)
       Nx5 for RADAR Data (x,y,z,velocity,itensity)

    """
    return np.loadtxt(path)


def readImage(path):
    """
       Will return an numpy array of
       shape height x width x 3.
    """
    return cv2.imread(path)  # [:,:,[2,1,0]]


def readLabels(path):
    """
       Reads the ground truth labels.
       In the labels are the following
       information stored:
       1. width in m
       2. length in m
       3. height in m
       4.-6. Coordinates of the center in m
       7. yaw rotation in degree
    """
    return np.loadtxt(path)

def dis(a, b):
    """
    finding distance between a(x,y,z) and b(x,y,z)
    """
    return round(math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2) + math.pow(a[2]-b[2], 2)), 4)


def get_movingpoints(pc):
    """
    This function returns moving points (dynamic) from a given frame.
    """
    i = 0
    movingpoints_array = np.empty((0, 5), float)
    while i < len(pc):
        if pc[i, 3] != 0:
            ###appending if velocity of points are non-zero
            movingpoints_array = np.append(movingpoints_array, [pc[i, :]], axis=0)
        i = i + 1
    return movingpoints_array


def refine_cluster(pc_movingpoints):

    """
    1. In each frame, all the points have some particular velocities (say 3 - 5). We are calling them as unique_vel
    Taking those unique velocities and their count (unique vel count means number of points having that unique velocity)

    2. Seperate all the moving points in each frame with respect to their velocity value.
    Arranging them in an array called 'b'. Finding mean for every set of velocity.

    3. As these velocity clusters are spread out all over the frame and there can be noise included in these clusters,
    eliminating far-away points are necessary. Far-away means having longer distances from the mean point of cluster.
        a. extra column with zeros is added.
        b. finding distance of that point from mean
        c. sorting rows from min  to max distance
        d. eliminating 20% longer distance points from 'b'
        e. finding mean for remaining 80%
        f. checking  whether the distance between new mean and prev mean is less than 0.5
        g. If it is more than 0.5, repeat the whole process
        h. Keeping all these means in Centre


    4. Using Unique vel count and centre, we find weighted mean.
    Returns distance from weighted mean to origin.

    """

    if pc_movingpoints.shape[0] != 0:  ### checking if there are moving points for sure
        ###finding out how many different values of velocities are there
        center = []
        unique_vel, unique_vel_count = np.unique(pc_movingpoints[:, 3], return_counts=True)
        #print(f'unique Vel count before processing: {unique_vel_count}')


        for i in range(len(unique_vel)):
            a = (pc_movingpoints[:, 3] == unique_vel[i])
            b = pc_movingpoints[a, :]
            mean = b.mean(0)[0:3]
            prev_mean = [0, 0, 0]
            b = np.c_[b, np.zeros(b.shape[0])]  ###adding extra column with zeros


            while dis(mean, prev_mean) > 0.5:
                b[:, 5] = np.sqrt(np.square(b[:, 0] - mean[0]) + np.square(b[:, 1] - mean[1]) + np.square(b[:, 2] - mean[2]))
                b = b[b[:, 5].argsort()] ###sorting from min to max distance
                lt = math.ceil(0.8 * b.shape[0])
                #print(f'unique vel:{i},num:{lt}')
                b = b[0:lt, :]
                prev_mean = mean
                mean = b.mean(axis=0)[0:3]

            unique_vel_count[i] = b.shape[0]
            #print(b.shape[0])
            #center = np.append(center, mean)
            center += mean,    ###all the means are centres of different unique velocities
            #print(center)


        ###finding average of all the centers
        x, y, z = 0, 0, 0
        for i in range(len(center)):
            center[i] = center[i] * unique_vel_count[i]
        for i in range(len(center)):
            x += center[i][0]
            y += center[i][1]
            z += center[i][2]

        sm = sum(unique_vel_count)
        x = x / sm
        y = y / sm
        z = z / sm

        distance = round((math.sqrt((x * x) + (y * y) + (z * z))), 4)
    else:
        max_unique_vel = 0 ###this line executes when there is no moving point (all are at rest)
        print(max_unique_vel)
        center = pc_radar.mean(0)[0:3] ###taking mean of all the points
        distance = round((math.sqrt((center[0] * center[0]) + (center[1] * center[1]) + (center[2] * center[2]))), 4)

    return distance

###main code from here###

#root = f"C:/Users/medav/OneDrive/Documents/LiDAR RaDAR/TASKS/Dataset2022/record1/"
root = f"C:/Users/medav/OneDrive/Documents/LiDAR RaDAR/TASKS/Dataset2022/record3/"
dist_array = np.empty((0, 1), float)
vel_array = np.empty((0, 1), float)
groundtruth_dist = np.empty((0, 1), float)
distance_diff = np.empty((0, 1), float)
for i in range(294):  # 240 for Record1 and 294 for Record3
    #pc_blick = readPC(f"{root}/blickfeld/{i:06d}.csv")
    #pc_velo = readPC(f"{root}/velodyne/{i:06d}.csv")
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    #label = readLabels(f"{root}/groundtruth/{i:06d}.csv")    ## no label data for record 3
    print("i=", i)
    #d = round((math.sqrt((label[3] * label[3]) + (label[4] * label[4]) + (label[5] * label[5]))), 4)  ##no label data for record 3
    ###Should we rotate the points here as we did in task 1? No. I guess
    movingpoints = get_movingpoints(pc_radar)
    dist = refine_cluster(movingpoints)
    #print(dist)
    dist_array = np.append(dist_array, dist)
    ###vel_array = np.append(vel_array, velocity)
    #groundtruth_dist = np.append(groundtruth_dist, d)
    #x = d - dist
    #distance_diff = np.append(distance_diff, x)

#mean_diff = sum(distance_diff)/len(distance_diff)
    #distance_diff += (d[-1] - dist[-1]),
#print(mean_diff)

x_data = list(range(294))
fig, ax = plt.subplots()
#ax.scatter(x_data, groundtruth_dist, color='red', label='True Distance')
ax.scatter(x_data, dist_array, color='orange', label='Algorithm Distance')
#ax.scatter(x_data, distance_diff, color='black', label='Difference')
#ax.scatter(x_data, mean_diff, color='green', label='Difference')
ax.set_xlabel('Record ID')
ax.set_ylabel('Distance (m)')
plt.legend(loc='upper right')

#ax2 = ax.twinx()
#ax2.plot(x_data, vel_array, color='black', label='max velocity')
#plt.legend(loc='upper right')
plt.show()

### visualisation for single frame
frame_id = 120
assert (record_number == 1 and frame_id < 240) or (record_number == 2 and frame_id < 100)

pc_blick = readPC(f"{path_blickfeld}/{frame_id:06d}.csv")
pc_velo = readPC(f"{path_velodyne}/{frame_id:06d}.csv")
pc_radar = readPC(f"{path_radar}/{frame_id:06d}.csv")
label = readLabels(f"{path_groundtruth}/{frame_id:06d}.csv")
img = readImage(f"{path_camera}/{frame_id:06d}.jpg")

print(f"Blickfeld point cloud shape: {pc_blick.shape}\n\
Velodyne point cloud shape: {pc_velo.shape}\n\
Radar point cloud shape: {pc_radar.shape}\n\
Label data shape: {label.shape}\n\
Image shape: {img.shape}")
movingpoints_f = get_movingpoints(pc_radar)

data1 = [go.Scatter3d(x=pc_radar[:, 0],
                     y=pc_radar[:, 1],
                     z=pc_radar[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(pc_radar.shape[0]),
                     marker={
                         'size': 2,
                         'color': "green",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=movingpoints_f[:, 0],
                     y=movingpoints_f[:, 1],
                     z=movingpoints_f[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(movingpoints.shape[0]),
                     marker={
                         'size': 2,
                         'color': "blue",
                         'colorscale': 'rainbow',
                     }),
        ]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [0, 40], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-20., 20.], 'rangemode': 'tozero'}
    }
)
fig = go.Figure(data=data1, layout=layout)
fig.show()
