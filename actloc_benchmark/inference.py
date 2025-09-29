import argparse
import logging
import os
import sys
from scipy.spatial.transform import Rotation as R

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

try:
    from utils.io import *
except ImportError as e:
    logging.error(f"Failed to import required local modules: {e}")
    logging.error("ensure utils/io.py is accessible")
    sys.exit(1)


from method.your_method import generate_smooth_and_aware_orientations


def predict_best_angles_per_pose(input: dict):
    """
    Compute the best viewing direction per waypoint.
    ...
    """
    waypoints = input["waypoints"]
    points3D = input["points3D"]


    # -- Predictive Smoothing Controls --
    # How many waypoints ahead the camera should look to anticipate turns.
    # A larger number creates smoother, longer, more sweeping turns.
    smoothing_window = 5

    # How much influence future path segments have.
    # Value is between 0 and 1. Closer to 1 gives more weight to the immediate
    # path. Closer to 0 gives more weight to upcoming turns, making it smoother.
    smoothing_strength = 0.6

    # -- Obstacle Detection Controls (from before) --
    # The width of the corridor for detecting obstacles.
    path_clearance = 0.4

    # The maximum angle the camera will tilt down when an obstacle is detected.
    max_tilt_angle = 30.0
    
    # The vertical tolerance to ignore points on the floor or ceiling.
    vertical_tolerance = 0.75



    best_angles = generate_smooth_and_aware_orientations(
        waypoints,
        points3D,
        path_clearance=path_clearance,
        vertical_tolerance=vertical_tolerance,
        max_tilt_angle=max_tilt_angle,
        smoothing_window=smoothing_window,
        smoothing_strength=smoothing_strength,
    )
    return best_angles

def main():
    parser = argparse.ArgumentParser(
        description="run inference on sfm scene with waypoints"
    )

    # required arguments
    parser.add_argument(
        "--sfm-dir",
        type=str,
        default="./example_data/00010-DBjEcHFg4oq/scene_reconstruction",
        help="path to colmap sfm reconstruction folder",
    )
    parser.add_argument(
        "--waypoints-file",
        type=str,
        default="./example_data/00010-DBjEcHFg4oq/sampled_waypoints.txt",
        help="path to text file containing waypoint coordinates [required]",
        required=True,
    )
    parser.add_argument(
        "--output-estimate",
        type=str,
        default="./example_data/estimate/pose_estimate.txt",
        help="output file to save best viewing angles for each waypoint [required]",
        required=True,
    )
    args = parser.parse_args()

    try:
        # load data
        logging.info("loading sfm model and waypoints...")
        cameras, images, points3D = load_sfm_model(args.sfm_dir)
        waypoints = load_waypoints(args.waypoints_file)

        input = {
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "waypoints": waypoints,
        }

        best_angles = predict_best_angles_per_pose(input)
        # save best angles to text file
        if best_angles:
            assert len(best_angles.keys()) == len(waypoints.keys())
            output_dir = os.path.dirname(args.output_estimate)
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"writing results in: {output_dir}")

            write_colmap_pose_file(waypoints, best_angles, args.output_estimate)
            logging.info(f"saved best viewing angles to {args.output_estimate}")

        logging.info("inference completed successfully!")

    except Exception as e:
        logging.error(f"inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
