import argparse
import logging
import sys
import os
import numpy as np
import torch

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actloc_core.io import load_sfm_model
from actloc_core.processing import filter_points_by_error, calculate_extrinsic_matrix
from actloc_core.torch_utils import load_model
from utils.helpers import run_inference_for_one_waypoint
from utils.io import load_waypoints


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run inference on SfM scene with waypoints")
    
    parser.add_argument('--sfm-dir', type=str, default='./example_data/00005_reference_sfm',
                       help='Path to COLMAP SfM reconstruction folder')
    parser.add_argument('--waypoints-file', type=str, default='./example_data/waypoints.txt',
                       help='Path to text file containing waypoint coordinates')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/trained_actloc.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-file', type=str, default='./example_data/predicted_poses.npz',
                       help='Output file for predicted poses (.npz format)')
    
    
    args = parser.parse_args()

    # Setup device and logging
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This model requires a NVIDIA GPU with CUDA support to run.")
        sys.exit(1)

    device = torch.device('cuda')
    logging.info(f"Using device: {device}")
    
    # Setup AMP
    amp_enabled = True
    use_bf16 = torch.cuda.is_bf16_supported()
    logging.info(f"AMP is enabled. Using {'bfloat16' if use_bf16 else 'float16'}.")
    
    try:
        # Load data
        logging.info("Loading SfM model and waypoints...")
        cameras, images, points3D = load_sfm_model(args.sfm_dir)
        waypoints = load_waypoints(args.waypoints_file)
        
        # Filter points by error
        filtered_points, filtered_colors = filter_points_by_error(points3D)
        
        # Load model
        model = load_model(args.checkpoint, device)
        
        # Set model dtype for AMP
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
        model.to(dtype=model_dtype)
        logging.info(f"Model set to dtype: {model_dtype}")
        
        # Process each waypoint
        print("\n--- Inference Results ---")
        results_to_save = []
        
        for waypoint_idx, waypoint in enumerate(waypoints):
            result = run_inference_for_one_waypoint(
                waypoint, waypoint_idx, filtered_points, filtered_colors,
                images, model, device, amp_enabled, use_bf16
            )
            
            if result is not None:
                prediction_grid, _, best_direction = result

                extrinsic = calculate_extrinsic_matrix(
                    position=waypoint,
                    x_angle=best_direction['best_x_angle'],
                    y_angle=best_direction['best_y_angle']
                )

                results_to_save.append({
                    'waypoint': waypoint,
                    'angles': np.array([best_direction['best_x_angle'], best_direction['best_y_angle']]),
                    'extrinsic': extrinsic,
                    'probability': best_direction['class_0_probability']
                })
                
                print(f"\nWaypoint {waypoint_idx + 1} Coordinates: {waypoint}")
                print("Predicted Grid (argmax classes):")
                print(prediction_grid)
                print("\n=== Best Viewing Direction for Class 0 (Good Accuracy) ===")
                print(f"Best cell position: Row {best_direction['best_cell'][0]}, Col {best_direction['best_cell'][1]}")
                print(f"Best viewing angles: X={best_direction['best_x_angle']}°, Y={best_direction['best_y_angle']}°")
                print(f"Class 0 probability: {best_direction['class_0_probability']:.4f}")
                print("-" * 50)
            else:
                # If processing failed, store NaN values
                results_to_save.append({
                    'waypoint': waypoint,
                    'angles': np.array([float('nan'), float('nan')]),
                    'extrinsic': np.eye(4),
                    'probability': float('nan')
                })
                print(f"\nWaypoint {waypoint_idx + 1}: Failed to process")
                print("-" * 50)
        
        if results_to_save:
            np.savez(
                args.output_file,
                waypoints=np.array([r['waypoint'] for r in results_to_save]),
                angles=np.array([r['angles'] for r in results_to_save]),
                extrinsics=np.array([r['extrinsic'] for r in results_to_save]),
                probabilities=np.array([r['probability'] for r in results_to_save])
            )
            logging.info(f"Saved predicted poses to {args.output_file}")
        
        logging.info("Inference completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
