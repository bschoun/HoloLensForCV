# Script to compute 3D point clouds from depth images donwloaded with recorder_console.py.
#

import argparse
import cv2
from glob import glob
import numpy as np
import os

from recorder_console import read_sensor_poses


# Depth range for short throw and long throw, in meters (approximate)
SHORT_THROW_RANGE = [0.02, 3.]
LONG_THROW_RANGE = [1., 4.]


def save_obj(output_path, points):
    with open(output_path, 'w') as f:
        f.write("# OBJ file\n")
        for v in points:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))


def parse_projection_bin(path, w, h):
    # See repo issue #63
    # Read binary file
    projection = np.fromfile(path, dtype=np.float32)
    x_list = [ projection[i] for i in range(0, len(projection), 2) ]
    y_list = [ projection[i] for i in range(1, len(projection), 2) ]
    
    u = np.asarray(x_list).reshape(w, h).T
    v = np.asarray(y_list).reshape(w, h).T

    return [u, v]


def pgm2distance(img, encoded=False):
    # See repo issue #19
    img.byteswap(inplace=True)
    return img.astype(np.float)/1000.0


def get_points(img, us, vs, cam2world, depth_range, no_inf=True):
    distance_img = pgm2distance(img, encoded=False)

    if cam2world is not None:
        R = cam2world[:3, :3]
        t = cam2world[:3, 3]
    else:
        R, t = np.eye(3), np.zeros(3)

    # Create a mask to remember which points to disregard in the end
    mask = np.ones(distance_img.shape, dtype=bool)
    mask[us==np.inf] = False
    mask[vs==np.inf] = False
    mask[distance_img > depth_range[1]] = False
    mask[distance_img < depth_range[0]] = False

    # Now fill in any np.inf values with 0's for easier processing
    us[us == np.inf] = 0.0
    vs[vs == np.inf] = 0.0
    distance_img[distance_img > depth_range[1]] = 0.0
    distance_img[distance_img < depth_range[0]] = 0.0

    # Negate us and vs
    us *= -1
    vs *= -1

    # Create a 3rd value with -1's
    ones = np.ones(us.shape)*-1
    
    # Stack us (x), vs (y), and ones (z) depth-wise
    points = np.dstack((us,vs,ones))

    # Calculate the norm of each point (axis 2)
    norm = np.linalg.norm(points,axis=2)

    # Reshape in order to divide
    points /= norm.reshape(norm.shape[0], norm.shape[1], 1)

    # Apply rotation matrix to each element of array
    shape = points.shape
    points = points.reshape((-1, 3, 1))
    points = R.dot(points.T).T
    points = points.reshape(shape)

    # Multiply by the distance image
    points *= distance_img.reshape((distance_img.shape[0], distance_img.shape[1], 1))

    # Apply translation
    points += t

    # Return a list of points with no infinity values
    if no_inf:
        return points[mask]

    # Return the array in the original image shape, including np.inf values
    else:
        # Set bad points back to np.inf
        points[mask==False] = np.inf
        return points


def get_cam2world(path, sensor_poses):
    time_stamp = int(os.path.splitext(os.path.basename(path))[0])
    world2cam = sensor_poses[time_stamp]    
    cam2world = np.linalg.inv(world2cam)
    return cam2world


def process_folder(args, cam):
    # Input folder
    folder = args.workspace_path
    cam_folder = os.path.join(folder, cam)
    assert(os.path.exists(cam_folder))
    # Output folder
    output_folder = os.path.join(args.output_path, cam)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get camera projection info
    bin_path = os.path.join(args.workspace_path, "%s_camera_space_projection.bin" % cam)
    
    # From frame to world coordinate system
    sensor_poses = None
    if not args.ignore_sensor_poses:
        sensor_poses = read_sensor_poses(os.path.join(folder, cam + ".csv"), identity_camera_to_image=True)        

    # Get appropriate depth thresholds
    depth_range = LONG_THROW_RANGE if 'long' in cam else SHORT_THROW_RANGE

    # Get depth paths
    depth_paths = sorted(glob(os.path.join(cam_folder, "*pgm")))
    if args.max_num_frames == -1:
        args.max_num_frames = len(depth_paths)
    depth_paths = depth_paths[args.start_frame:(args.start_frame + args.max_num_frames)]    

    us = vs = None
    # Process paths
    for i_path, path in enumerate(depth_paths):
        if (i_path % 10) == 0:
            print("Progress: %d/%d" % (i_path+1, len(depth_paths)))
        output_suffix = "_%s" % args.output_suffix if len(args.output_suffix) else ""
        pcloud_output_path = os.path.join(output_folder, os.path.basename(path).replace(".pgm", "%s.obj" % output_suffix))
        if os.path.exists(pcloud_output_path):
            continue
        img = cv2.imread(path, -1)
        if us is None or vs is None:
            us, vs = parse_projection_bin(bin_path, img.shape[1], img.shape[0])
        cam2world = get_cam2world(path, sensor_poses) if sensor_poses is not None else None
        points = get_points(img, us, vs, cam2world, depth_range)        
        save_obj(pcloud_output_path, points)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_path", required=True, help="Path to workspace folder used for downloading")
    parser.add_argument("--output_path", required=False, help="Path to output folder where to save the point clouds. By default, equal to output_path")
    parser.add_argument("--output_suffix", required=False, default="", help="If a suffix is specified, point clouds will be saved as [tstamp]_[suffix].obj")
    parser.add_argument("--short_throw", action='store_true', help="Extract point clouds from short throw frames")
    parser.add_argument("--long_throw", action='store_true', help="Extract point clouds from long throw frames")
    parser.add_argument("--ignore_sensor_poses", action='store_true', help="Drop HL pose information (point clouds will not be aligned to a common ref space)")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_num_frames", type=int, default=-1)

    args = parser.parse_args()

    if (not args.short_throw) and (not args.long_throw):
        print("At least one between short_throw and long_throw must be set to true.\
                Please pass \"--short_throw\" and/or \"--long_throw\" as parameter.")
        exit()
    
    assert(os.path.exists(args.workspace_path))    
    if args.output_path is None:
        args.output_path = args.workspace_path

    return args


def main():
    args = parse_args()

    if args.short_throw:
        print('Processing short throw depth folder...')
        process_folder(args, 'short_throw_depth')
        print('done.')

    if args.long_throw:
        print('Processing long throw depth folder...')
        process_folder(args, 'long_throw_depth')
        print('done.')


if __name__ == "__main__":    
    main()    
