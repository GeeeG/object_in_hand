import numpy as np
import os, sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # for cv2 package
import cv2
import open3d as o3d
from IPython import embed
from scipy import stats
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import transforms3d
import math
import random
import tensorflow as tf

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079
depth_scale = 1
img_path = "/data_c/HANDS2017/training/images/"

b_visual = False


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def FPS_w_cls(pts, K):
    pts_xyz = pts[:, 0:3]
    farthest_pts = np.zeros((K, 3))
    farthest_pts_w_cls = np.zeros((K, 5))
    idx_0 = np.random.randint(len(pts_xyz))
    farthest_pts[0] = pts_xyz[idx_0]
    farthest_pts_w_cls[0] = pts[idx_0]
    distances = calc_distances(farthest_pts[0], pts_xyz)
    for i in range(1, K):
        farthest_pts[i] = pts_xyz[np.argmax(distances)]
        farthest_pts_w_cls[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts_xyz))
    return farthest_pts_w_cls


def depth_to_cloud(depth, fx, fy, cx, cy, depth_scaling_factor):
    """
    translate data from pixel coordinate frame into image coordinate frame
    :param depth: a matrix with a shape (height, width)
    :param fx: f/dx
    :param fy: f/dy
    :param cx: u0
    :param cy: v0
    :param depth_scaling_factor: used to translate to meter
    :return:
    """
    depth_in_meter = depth / depth_scaling_factor
    dshape = depth.shape
    v_shape_height = dshape[0]
    u_shape_width = dshape[1]

    xv = np.arange(0, u_shape_width, 1)
    yu = np.arange(0, v_shape_height, 1)

    X, Y = np.meshgrid(xv, yu)
    x = ((X - cx) * depth_in_meter / fx)
    y = ((Y - cy) * depth_in_meter / fy)

    xyz = np.stack([x, y, depth_in_meter], axis=2) # (v_shape_height, u_shape_width, 3)

    return xyz


def pca_rotation(points):
    hand_points = points.copy()
    hand_points_mean = hand_points.mean(axis=0)
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(hand_points)
    hand_points_pca = pca.transform(hand_points) + hand_points_mean
    return hand_points_pca, pca.components_


def get_human_points(line):
    # 1 read the groundtruth and the image
    frame = line.split(' ')[0].replace("\t", "")
    print(frame)

    # image path depends on the location of your training dataset
    try:
        img = cv2.imread(img_path + str(frame), cv2.IMREAD_ANYDEPTH)
    except:
        print("no Image", frame)
        return

    label_source = line.split('\t')[1:]
    label = [float(l.replace(" ", "")) for l in label_source[0:63]]
    keypoints = np.array(label).reshape(21, 3)

    # 2 get hand points
    padding = 80
    points_raw = depth_to_cloud(img, focalLengthX, focalLengthY, centerX, centerY, depth_scale)
    points_raw_position = points_raw.reshape((-1, 3))

    x_min_max = [np.min(keypoints[:, 0] - padding / 2), np.max(keypoints[:, 0]) + padding / 2]
    y_min_max = [np.min(keypoints[:, 1] - padding / 2), np.max(keypoints[:, 1]) + padding / 2]
    z_min_max = [np.min(keypoints[:, 2] - padding / 2), np.max(keypoints[:, 2]) + padding / 2]

    points = points_raw[np.where((points_raw[:, :, 0] > x_min_max[0]) & (points_raw[:, :, 0] < x_min_max[1])
                                 & (points_raw[:, :, 1] > y_min_max[0]) & (points_raw[:, :, 1] < y_min_max[1])
                                 & (points_raw[:, :, 2] > z_min_max[0]) & (points_raw[:, :, 2] < z_min_max[1]))]

    hand_ok = True
    if len(points) < 300:
        print("%s hand points is %d, which is less than 300. Maybe it's a broken image" % (frame, len(points)))
        hand_ok = False
        return points, hand_ok
    else:
        farthest_pts = FPS(points, 512)
        farthest_pts = farthest_pts / 1000
        hand_mean = np.mean(farthest_pts, axis=0)
        farthest_pts = farthest_pts - np.expand_dims(hand_mean, 0)

    if False:

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(farthest_pts)
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0, 0, 0]))
        o3d.visualization.draw_geometries([pcd, world])

    return farthest_pts, hand_ok


def get_object_model(obj_name):
    ply_path = "/data_c/ycb_data/ycb_video_obj_coor_ply/" + obj_name + ".ply"
    pcd = o3d.io.read_point_cloud(ply_path)

    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    #pts = np.concatenate((xyz, rgb), axis=1)
    farthest_pts = FPS(xyz, 512)

    max_bounds = pcd.get_max_bound()
    min_bounds = pcd.get_min_bound()

    return farthest_pts, max_bounds-min_bounds


def sphericalFlip(points, center, param):
    n = len(points)  # total n points
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    normPoints = np.linalg.norm(points, axis=1)  # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis=0)  # Radius of Sphere
    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]),
                                                           axis=1))  # Apply Equation to get Flipped Points
    flippedPoints += points

    return flippedPoints


def convexHull(points):
    points = np.append(points, [[0, 0, 0]], axis=0)  # All points plus origin
    hull = ConvexHull(points)  # Visibal points plus possible origin. Use its vertices property.

    return hull


def hidden_point_removal(points):
    points_xyz = points[:, 0:3]
    flag = np.zeros(len(points_xyz), int)
    C = np.array([[0,0,0]])
    flippedPoints = sphericalFlip(points_xyz, C, 0.9 * math.pi)
    myHull = convexHull(flippedPoints)
    visibleVertex = myHull.vertices[:-1]  # indexes of visible points
    flag[visibleVertex] = 1
    visibleId = np.where(flag == 1)[0]  # indexes of the invisible points
    return points[visibleId, :]


def sample_rot(min_rad=0, max_rad=np.pi):
    # http: // mathworld.wolfram.com / SpherePointPicking.html
    theta = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(-1, 1)
    x = np.sqrt(1 - u * u) * np.cos(theta)
    y = np.sqrt(1 - u * u) * np.sin(theta)
    z = u
    axis = np.array([x, y, z])
    return axis, np.random.uniform(min_rad, max_rad)


def main(writer, data_size):

    table_points = np.fromfile("cloud.npy", dtype='float64')
    table_points = table_points.reshape(-1, 3)
    table_points = table_points[table_points[:, 2] < 5.0]
    table_cloud = o3d.geometry.PointCloud()
    table_cloud.points = o3d.utility.Vector3dVector(np.asarray(table_points))

    poses = np.loadtxt("tams_ur5_workspace.txt")
    translations = poses[:, 0:3]
    trans_tr_kde = stats.gaussian_kde(translations.T)
    trans_gen = np.transpose(trans_tr_kde.resample(data_size))

    datafile = open("/data_c/HANDS2017/training/Training_Annotation.txt", "r")
    lines_test = datafile.read().splitlines()
    lines_test.sort()
    object_points, bounds = get_object_model("035_power_drill")
    object_cloud = o3d.geometry.PointCloud()
    object_cloud.points = o3d.utility.Vector3dVector(object_points)
    object_cloud.paint_uniform_color([0.9, 0.1, 0.1])

    hand_cloud = o3d.geometry.PointCloud()

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0, 0, 0]))

    visible_cloud = o3d.geometry.PointCloud()

    if b_visual:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters("view.json")

        vis.add_geometry(table_cloud)
        vis.add_geometry(camera_frame)
        vis.add_geometry(visible_cloud)
        # vis.add_geometry(object_cloud)
        # vis.add_geometry(hand_cloud)
        ctr.convert_from_pinhole_camera_parameters(param)

    axag = []

    for i in np.arange(data_size):
        hand_points, hand_ok = get_human_points(lines_test[i])
        if hand_ok:
            hand_points_rotate, _ = pca_rotation(hand_points)
            z_factor_pool = [1, -1]
            z_factor = random.choice(z_factor_pool)

            x_factor = random.uniform(-1, 1)
            y_factor = random.uniform(0.5, 1)

            # use drill 0
            if False:
                hand_cloud.points = o3d.utility.Vector3dVector(hand_points_rotate + np.array([bounds[0]/4, -0.02, z_factor*bounds[2]/2]))

            # hand-over top 1
            if False:
                rot_mat = transforms3d.axangles.axangle2mat([0, 0, 1], np.pi*0.5)
                trans = np.array([bounds[0]/2*x_factor, bounds[1]/2*y_factor, z_factor*bounds[2]/2])
                hand_transform = transforms3d.affines.compose(trans, rot_mat, np.ones(3))
                hand_cloud.points = o3d.utility.Vector3dVector(hand_points_rotate)
                hand_cloud.transform(hand_transform)

            # hand-over tail 2
            if True:
                rot_mat = transforms3d.axangles.axangle2mat([0, 0, 1], np.pi * 1.5)
                trans = np.array([bounds[0] / 3 * x_factor, - bounds[1] / 2 * y_factor, z_factor * bounds[2] / 2])
                hand_transform = transforms3d.affines.compose(trans, rot_mat, np.ones(3))
                hand_cloud.points = o3d.utility.Vector3dVector(hand_points_rotate)
                hand_cloud.transform(hand_transform)

            hand_cloud.paint_uniform_color([0.1, 0.1, 0.9])

            axis, angle = sample_rot(0, np.pi)
            axag.append(angle * axis)
            rot_mat_current = transforms3d.axangles.axangle2mat(axis, angle)

            object_pose_transform = transforms3d.affines.compose(trans_gen[i, :], rot_mat_current, np.ones(3))

            hand_cloud.transform(object_pose_transform)
            object_cloud.transform(object_pose_transform)

            # hidden point removal
            hand_points_transformed = np.asarray(hand_cloud.points)
            # add hand class info: 22
            hand_cls_gt_onehot = np.identity(2)[0]
            hand_cls_gt_onehot_expand = np.expand_dims(hand_cls_gt_onehot, axis=0)
            hand_cls_gt_onehot_tile = np.tile(hand_cls_gt_onehot_expand, (512, 1))

            object_points_transformed = np.asarray(object_cloud.points)
            # add object class info: 14
            object_cls_gt_onehot = np.identity(2)[1]
            object_cls_gt_onehot_expand = np.expand_dims(object_cls_gt_onehot, axis=0)
            object_cls_gt_onehot_tile = np.tile(object_cls_gt_onehot_expand, (512, 1))

            all_points = np.vstack([np.hstack([hand_points_transformed, hand_cls_gt_onehot_tile]),
                                    np.hstack([object_points_transformed, object_cls_gt_onehot_tile])])

            visible_points = hidden_point_removal(all_points)
            visible_points_fps = FPS_w_cls(visible_points, 256)

            if b_visual:
                visible_cloud.points = o3d.utility.Vector3dVector(visible_points_fps[:, 0:3])
                visible_cloud.paint_uniform_color([0.9, 0.1, 0.9])
                vis.update_geometry(visible_cloud)
                vis.poll_events()
                vis.update_renderer()

            object_cloud.transform(np.linalg.inv(object_pose_transform))

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _int64_feature([14]),
                    'xyz': _float_feature(visible_points_fps.reshape(-1)),
                    'translation': _float_feature(trans_gen[i, :].reshape(-1)),
                    'rotation': _float_feature((angle * axis).reshape(-1))
                }
            ))

            writer.write(example.SerializeToString())

        else:
            continue

    writer.close()

    datafile.close()

if __name__ == '__main__':

    output_path = "/data_c/PointNet/pointnet/object_in_hand/tfRecords/train_14_notready.tfrecords"

    writer = tf.python_io.TFRecordWriter(output_path)

    main(writer, data_size=10000)
