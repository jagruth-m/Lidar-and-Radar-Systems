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

'''np.loadtxt(path_blickfeld, delimiter=',')
np.loadtxt(path_velodyne, delimiter=',')
np.loadtxt(path_groundtruth, delimiter=',')'''


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


def rt_matrix(roll=0, pitch=0, yaw=0):
    """
        Returns a 3x3 Rotation Matrix. Angels in degree!
    """
    yaw = yaw * np.pi / 180
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    # Rotationmatrix
    rot = np.dot(np.dot(np.array([[c_y, - s_y, 0],
                                  [s_y, c_y, 0],
                                  [0, 0, 1]]),
                        np.array([[c_p, 0, s_p],
                                  [0, 1, 0],
                                  [-s_p, 0, c_p]])),
                 np.array([[1, 0, 0],
                           [0, c_r, - s_r],
                           [0, s_r, c_r]]))
    return rot


def rotate_points(points, rot_t):
    """
        Input must be of shape N x 3
        Returns the rotated point cloud for a given roation matrix
        and point cloud.
    """
    points[0:3, :] = np.dot(rot_t, points[0:3, :])
    return points


def make_boundingbox(label):
    """
        Returns the corners of a bounding box from a label.
    """
    corner = np.array([
        [+ label[0] / 2, + label[1] / 2, + label[2] / 2],
        [+ label[0] / 2, + label[1] / 2, - label[2] / 2],
        [+ label[0] / 2, - label[1] / 2, + label[2] / 2],
        [+ label[0] / 2, - label[1] / 2, - label[2] / 2],
        [- label[0] / 2, + label[1] / 2, + label[2] / 2],
        [- label[0] / 2, - label[1] / 2, + label[2] / 2],
        [- label[0] / 2, + label[1] / 2, - label[2] / 2],
        [- label[0] / 2, - label[1] / 2, - label[2] / 2],
    ])
    corner = rotate_points(corner, rt_matrix(yaw=label[6]))
    corner = corner + label[3:6]
    return corner


def get_sub_points_of_object(obj, pc):
    """
       Determines those points of the point cloud (pc)
       that are inside of the object (obj).
    """
    """1. Get the bounding box corners."""

    bb_raw = make_boundingbox(obj)

    # 2. Rotate the data such that the edges of the
    #    bounding box is aligned with the axis.
    #    For this we rotate the point cloud and the
    #    corners into the opposite direction than
    #    stated by the yaw value from the groundtruth
    #    file. Calculate the inverse of the
    #    rotation matrix.

    inv = np.linalg.inv(rt_matrix(yaw=label[6]))
    pc_new = rotate_points(pc, inv)
    #print('new_len =', len(pc_new))
    bb = rotate_points(bb_raw, inv)
    #print(bb)
    # New bounding box for the visualization
    bb = np.array([bb[0], bb[1], bb[3], bb[2], bb[0], bb[4], bb[5],
                   bb[2], bb[3], bb[7], bb[5], bb[4], bb[6], bb[7], bb[6], bb[1]])
    #print(bb)

    # 3. We need for every dimension the minimal and
    #    maximal value of the corners.
    x_min = np.min(bb[:, 0])
    x_max = np.max(bb[:, 0])
    y_min = np.min(bb[:, 1])
    y_max = np.max(bb[:, 1])
    z_min = np.min(bb[:, 2])
    z_max = np.max(bb[:, 2])

    # 4. Only those points of the point cloud that are
    #    smaller than the maximum and larger than the
    #    minimum in every dimension are inside the box.
    i = 0
    subpoints_array = np.empty((0, 3), float)
    while i < len(pc_new):
        # row_array = np.array(pc_new[i,:])

        if (pc_new[i, 0] > x_min) and (pc_new[i, 0] < x_max):
            if (pc_new[i, 1] > y_min) and (pc_new[i, 1] < y_max):
                if (pc_new[i, 2] > z_min) and (pc_new[i, 2] < z_max):
                    # print(pc_new[i,:])
                    subpoints_array = np.append(subpoints_array, [pc_new[i, :]], axis=0)
        i = i + 1
    # print("i=",i)
    return subpoints_array

###main code from here###

root = f"C:/Users/medav/OneDrive/Documents/LiDAR RaDAR/TASKS/Dataset2022/record1/"
'''pts_blick_rec1 = []
pts_velo_rec1 = []
pts_radar_rec1 = []'''
dist_array = np.empty((0, 1), float)
no_of_blick_points = np.empty((0, 1), float)
no_of_velo_points = np.empty((0, 1), float)
for i in range(240):
    pc_blick = readPC(f"{root}/blickfeld/{i:06d}.csv")
    pc_velo = readPC(f"{root}/velodyne/{i:06d}.csv")
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    label = readLabels(f"{root}/groundtruth/{i:06d}.csv")
    print("i=", i)
    subpoints_blick = get_sub_points_of_object(label, pc_blick[:, 0:3])
    subpoints_velo = get_sub_points_of_object(label, pc_velo[:, 0:3])
    x_coord = label[3]
    y_coord = label[4]
    z_coord = label[5]
    dist = math.sqrt(x_coord ** 2 + y_coord ** 2 +z_coord ** 2)
    dist_array = np.append(dist_array, dist)
    no_of_blick_points = np.append(no_of_blick_points, len(subpoints_blick))
    no_of_velo_points = np.append(no_of_velo_points, len(subpoints_velo))

    '''pts_blick_rec1 += [get_sub_points_of_object(label, pc_blick[:, 0:3])[0].sum()]
    print(pts_blick_rec1)
    #a = pts_blick_rec1.shape()
    pts_velo_rec1 += [get_sub_points_of_object(label, pc_velo[:, 0:3])[0].sum()]
    pts_radar_rec1 += [get_sub_points_of_object(label, pc_radar[:, 0:3])[0].sum()]

    plt.plot(pts_blick_rec1)
    plt.plot(pts_velo_rec1)
    plt.plot(pts_radar_rec1)
    plt.legend(["Blick", "Velodyne", "Radar"])
    plt.show(block=False)'''

plt.figure(1)
plt.xlabel('distance from the sensor(in m)')
plt.ylabel('Number of points in boundingbox')
plt.title('Blickfield')
plt.scatter(dist_array, no_of_blick_points, color='red')

plt.figure(2)
plt.xlabel('distance from the sensor (in m)')
plt.ylabel('Number of points in boundingbox')
plt.title('Velodyne')
plt.scatter(dist_array, no_of_velo_points, color='blue')

frame_id = 120
assert (record_number == 1 and frame_id < 240) or (record_number == 2 and frame_id < 100), \
    "Record number 1 only has 240 frames and" \
    " record number 2 only has 100 frames."
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

subpoints_b = get_sub_points_of_object(label, pc_blick[:, 0:3])
subpoints_v = get_sub_points_of_object(label, pc_velo[:, 0:3])
subpoints_r = get_sub_points_of_object(label, pc_radar[:, 0:3])
#print(subpoints)
#print(len(subpoints))
bb_raw = make_boundingbox(label)
inv = np.linalg.inv(rt_matrix(yaw=label[6]))
pc_new_b = rotate_points(pc_blick[:, 0:3], inv)
pc_new_v = rotate_points(pc_velo[:, 0:3], inv)
pc_new_r = rotate_points(pc_radar[:, 0:3], inv)
bb = rotate_points(bb_raw, inv)
# New bounding box for the visualization
bb = np.array([bb[0], bb[1], bb[3], bb[2], bb[0], bb[4], bb[5],
               bb[2], bb[3], bb[7], bb[5], bb[4], bb[6], bb[7], bb[6], bb[1]])

data1 = [go.Scatter3d(x=pc_new_b[:, 0],
                     y=pc_new_b[:, 1],
                     z=pc_new_b[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(pc_blick.shape[0]),
                     marker={
                         'size': 2,
                         'color': "green",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=subpoints_b[:, 0],
                     y=subpoints_b[:, 1],
                     z=subpoints_b[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(subpoints_b.shape[0]),
                     marker={
                         'size': 2,
                         'color': "blue",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=bb[:, 0],
                     y=bb[:, 1],
                     z=bb[:, 2],
                     mode='lines', type='scatter3d',
                     line={
                         'width': 10,
                         'color': "red",
                         'colorscale': 'rainbow'
                     })
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
data2 = [go.Scatter3d(x=pc_new_v[:, 0],
                     y=pc_new_v[:, 1],
                     z=pc_new_v[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(pc_velo.shape[0]),
                     marker={
                         'size': 2,
                         'color': "green",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=subpoints_v[:, 0],
                     y=subpoints_v[:, 1],
                     z=subpoints_v[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(subpoints_v.shape[0]),
                     marker={
                         'size': 2,
                         'color': "blue",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=bb[:, 0],
                     y=bb[:, 1],
                     z=bb[:, 2],
                     mode='lines', type='scatter3d',
                     line={
                         'width': 10,
                         'color': "red",
                         'colorscale': 'rainbow'
                     })
        ]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [0, 40], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-20., 20.], 'rangemode': 'tozero'}
    }
)
fig = go.Figure(data=data2, layout=layout)
fig.show()
data3 = [go.Scatter3d(x=pc_new_r[:, 0],
                     y=pc_new_r[:, 1],
                     z=pc_new_r[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(pc_radar.shape[0]),
                     marker={
                         'size': 2,
                         'color': "green",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=subpoints_r[:, 0],
                     y=subpoints_r[:, 1],
                     z=subpoints_r[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(subpoints_r.shape[0]),
                     marker={
                         'size': 2,
                         'color': "blue",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=bb[:, 0],
                     y=bb[:, 1],
                     z=bb[:, 2],
                     mode='lines', type='scatter3d',
                     line={
                         'width': 10,
                         'color': "red",
                         'colorscale': 'rainbow'
                     })
        ]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [0, 40], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-20., 20.], 'rangemode': 'tozero'}
    }
)
fig = go.Figure(data=data3, layout=layout)
fig.show()
plt.show()
'''subpoints = get_sub_points_of_object(label, pc_velo[:, 0:3])
data = [go.Scatter3d(x=pc_velo[:, 0],
                     y=pc_velo[:, 1],
                     z=pc_velo[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(pc_velo.shape[0]),
                     marker={
                         'size': 2,
                         'color': "green",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=subpoints[1][:, 0],
                     y=subpoints[1][:, 1],
                     z=subpoints[1][:, 2],
                     mode='markers', type='scatter3d',
                     marker={
                         'size': 2,
                         'color': "blue",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=bb[:, 0],
                     y=bb[:, 1],
                     z=bb[:, 2],
                     mode='lines', type='scatter3d',
                     line={
                         'width': 10,
                         'color': "red",
                         'colorscale': 'rainbow'
                     })
        ]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [0, 40], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-20., 20.], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout=layout)

subpoints = get_sub_points_of_object(label, pc_radar[:, 0:3])
data = [go.Scatter3d(x=pc_radar[:, 0],
                     y=pc_radar[:, 1],
                     z=pc_radar[:, 2],
                     mode='markers', type='scatter3d',
                     text=np.arange(pc_radar.shape[0]),
                     marker={
                         'size': 2,
                         'color': "green",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=subpoints[1][:, 0],
                     y=subpoints[1][:, 1],
                     z=subpoints[1][:, 2],
                     mode='markers', type='scatter3d',
                     marker={
                         'size': 2,
                         'color': "blue",
                         'colorscale': 'rainbow',
                     }),
        go.Scatter3d(x=bb[:, 0],
                     y=bb[:, 1],
                     z=bb[:, 2],
                     mode='lines', type='scatter3d',
                     line={
                         'width': 10,
                         'color': "red",
                         'colorscale': 'rainbow'
                     })
        ]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [0, 40], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-20., 20.], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout=layout)

root = f"C:/Users/medav/OneDrive/Documents/LiDAR RaDAR/TASKS/Dataset2022/record1/"
pts_blick_rec1 = []
pts_velo_rec1 = []
pts_radar_rec1 = []
for i in range(240):
    pc_blick = readPC(f"{root}/blickfeld/{i:06d}.csv")
    pc_velo = readPC(f"{root}/velodyne/{i:06d}.csv")
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    label = readLabels(f"{root}/groundtruth/{i:06d}.csv")

    pts_blick_rec1 += [get_sub_points_of_object(label, pc_blick[:, 0:3])[0].sum()]
    pts_velo_rec1 += [get_sub_points_of_object(label, pc_velo[:, 0:3])[0].sum()]
    pts_radar_rec1 += [get_sub_points_of_object(label, pc_radar[:, 0:3])[0].sum()]

plt.plot(pts_blick_rec1)
plt.plot(pts_velo_rec1)
plt.plot(pts_radar_rec1)
plt.legend(["Blick", "Velodyne", "Radar"])

root = f"C:/Users/medav/OneDrive/Documents/LiDAR RaDAR/TASKS/Dataset2022/record2/"
pts_blick_rec2 = []
pts_velo_rec2 = []
pts_radar_rec2 = []
for i in range(100):
    pc_blick = readPC(f"{root}/blickfeld/{i:06d}.csv")
    pc_velo = readPC(f"{root}/velodyne/{i:06d}.csv")
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    label = readLabels(f"{root}/groundtruth/{i:06d}.csv")

    pts_blick_rec2 += [get_sub_points_of_object(label, pc_blick[:, 0:3])[0].sum()]
    pts_velo_rec2 += [get_sub_points_of_object(label, pc_velo[:, 0:3])[0].sum()]
    pts_radar_rec2 += [get_sub_points_of_object(label, pc_radar[:, 0:3])[0].sum()]

plt.plot(pts_blick_rec2)
plt.plot(pts_velo_rec2)
plt.plot(pts_radar_rec2)
plt.legend(["Blick", "Velodyne", "Radar"])'''
