#!/usr/bin/env python3

import json
import numpy as np
from pyquaternion import Quaternion


def apply_coordinate_transformation(position):
    """
    Apply coordinate transformation:
    x_new = -z_old
    y_new = x_old
    z_new = -y_old
    """
    x_old, y_old, z_old = position
    return np.array([-z_old, x_old, -y_old])


def apply_rotation_transformation(quaternion):
    """
    Apply rotation transformation to match the coordinate system change
    """
    # Convert to rotation matrix
    rotation_matrix = quaternion.rotation_matrix
    
    # Transformation matrix
    T = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ])
    
    # Apply transformation: R_new = T * R_old
    transformed_rotation_matrix = T @ rotation_matrix
    
    # Convert back to quaternion
    transformed_quaternion = Quaternion(matrix=transformed_rotation_matrix)
    
    return transformed_quaternion


def quaternion_to_transform_matrix(quaternion, translation):
    """
    Convert quaternion and translation to 4x4 transformation matrix
    """
    rotation_matrix = quaternion.rotation_matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix


def rotation_matrix_to_euler_robust(R, cam_name=""):
    """
    Convert rotation matrix to Euler angles using the specified mathematical formulation.
    
    Implementation of:
    if (R31 != ±1)
        theta_1 = -asin(R31)
        theta_2 = pi - theta_1
        psi_1 = atan2(R32/cos theta_1, R33/cos theta_1)
        psi_2 = atan2(R32/cos theta_2, R33/cos theta_2)
        phi_1 = atan2(R21/cos theta_1, R11/cos theta_1)
        phi_2 = atan2(R21/cos theta_2, R11/cos theta_2)
    else
        phi = 0  # can set to anything
        if (R31 = -1)
            theta = pi/2
            psi = phi + atan2(R12, R13)
        else
            theta = -pi/2
            psi = -phi + atan2(-R12, -R13)
    
    Args:
        R: 3x3 rotation matrix
        cam_name: Camera name for debug prints
    
    Returns:
        tuple: (psi, theta, phi) in radians (yaw, pitch, roll)
    """
    print(f"\n=== {cam_name} - Euler Angle Computation ===")
    
    # Ensure numerical precision
    R = np.array(R, dtype=np.float64)
    print(f"Rotation matrix R:\n{R}")
    print(f"R[2,0] (R31): {R[2, 0]}")
    
    # Check if R31 != ±1 (not at singularity)
    if abs(R[2, 0]) != 1.0 and abs(R[2, 0]) != -1.0:
        print("Using non-singularity case")
        
        # Two possible solutions
        theta_1 = -np.arcsin(R[2, 0])  # asin(R31)
        theta_2 = np.pi - theta_1
        print(f"theta_1 = -arcsin({R[2, 0]}) = {theta_1} rad = {np.degrees(theta_1)} deg")
        print(f"theta_2 = pi - theta_1 = {theta_2} rad = {np.degrees(theta_2)} deg")
        
        # Compute corresponding psi and phi values
        cos_theta_1 = np.cos(theta_1)
        cos_theta_2 = np.cos(theta_2)
        print(f"cos(theta_1) = {cos_theta_1}")
        print(f"cos(theta_2) = {cos_theta_2}")
        
        psi_1 = np.arctan2(R[2, 1] / cos_theta_1, R[2, 2] / cos_theta_1)  # atan2(R32/cos theta_1, R33/cos theta_1)
        psi_2 = np.arctan2(R[2, 1] / cos_theta_2, R[2, 2] / cos_theta_2)  # atan2(R32/cos theta_2, R33/cos theta_2)
        print(f"psi_1 = atan2({R[2, 1]}/{cos_theta_1}, {R[2, 2]}/{cos_theta_1}) = {psi_1} rad = {np.degrees(psi_1)} deg")
        print(f"psi_2 = atan2({R[2, 1]}/{cos_theta_2}, {R[2, 2]}/{cos_theta_2}) = {psi_2} rad = {np.degrees(psi_2)} deg")
        
        phi_1 = np.arctan2(R[1, 0] / cos_theta_1, R[0, 0] / cos_theta_1)  # atan2(R21/cos theta_1, R11/cos theta_1)
        phi_2 = np.arctan2(R[1, 0] / cos_theta_2, R[0, 0] / cos_theta_2)  # atan2(R21/cos theta_2, R11/cos theta_2)
        print(f"phi_1 = atan2({R[1, 0]}/{cos_theta_1}, {R[0, 0]}/{cos_theta_1}) = {phi_1} rad = {np.degrees(phi_1)} deg")
        print(f"phi_2 = atan2({R[1, 0]}/{cos_theta_2}, {R[0, 0]}/{cos_theta_2}) = {phi_2} rad = {np.degrees(phi_2)} deg")
        
        # Choose the second solution (as per your modification)
        phi = phi_2
        theta = theta_2
        psi = psi_2
        print(f"Selected solution 2: phi={phi} ({np.degrees(phi)} deg), theta={theta} ({np.degrees(theta)} deg), psi={psi} ({np.degrees(psi)} deg)")
        
    else:
        print("Using singularity case")
        # Singularity case: R31 = ±1
        phi = 0.0  # Set to 0 as suggested
        print(f"phi set to 0.0")
        
        if R[2, 0] == -1.0:  # R31 = -1
            print("R31 = -1 case")
            theta = np.pi / 2
            psi = phi + np.arctan2(R[0, 1], R[0, 2])  # atan2(R12, R13)
            print(f"theta = pi/2 = {theta} rad = {np.degrees(theta)} deg")
            print(f"psi = {phi} + atan2({R[0, 1]}, {R[0, 2]}) = {psi} rad = {np.degrees(psi)} deg")
        else:  # R31 = 1
            print("R31 = 1 case")
            theta = -np.pi / 2
            psi = -phi + np.arctan2(-R[0, 1], -R[0, 2])  # atan2(-R12, -R13)
            print(f"theta = -pi/2 = {theta} rad = {np.degrees(theta)} deg")
            print(f"psi = -{phi} + atan2({-R[0, 1]}, {-R[0, 2]}) = {psi} rad = {np.degrees(psi)} deg")
    
    print(f"Final Euler angles: psi={psi} ({np.degrees(psi)} deg), theta={theta} ({np.degrees(theta)} deg), phi={phi} ({np.degrees(phi)} deg)")
    return psi, theta, phi


def quaternion_to_euler_robust(quaternion, cam_name=""):
    """
    Convert quaternion to Euler angles using robust rotation matrix method.
    
    Args:
        quaternion: Quaternion object or [w, x, y, z] list
        cam_name: Camera name for debug prints
    
    Returns:
        tuple: (yaw, pitch, roll) in radians
    """
    # Convert to rotation matrix first
    if isinstance(quaternion, Quaternion):
        rotation_matrix = quaternion.rotation_matrix
    else:
        # Assume [w, x, y, z] format
        q = Quaternion(w=quaternion[0], x=quaternion[1], y=quaternion[2], z=quaternion[3])
        rotation_matrix = q.rotation_matrix
    
    # Use robust matrix to Euler conversion
    return rotation_matrix_to_euler_robust(rotation_matrix, cam_name)


def process_camera_data(cam_data, cam_name):
    """Process a single camera's transformation data with detailed debug output."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {cam_name}")
    print(f"{'='*60}")
    
    # Extract original data
    original_translation = np.array(cam_data['translation'])
    original_rotation = cam_data['rotation']  # [w, x, y, z]
    camera_intrinsic = np.array(cam_data['camera_intrinsic'])
    
    print(f"\nOriginal data:")
    print(f"Translation: {original_translation}")
    print(f"Rotation (quaternion w,x,y,z): {original_rotation}")
    print(f"Camera intrinsic:\n{camera_intrinsic}")
    
    # Convert to quaternion
    original_quaternion = Quaternion(w=original_rotation[0], x=original_rotation[1], 
                                   y=original_rotation[2], z=original_rotation[3])
    
    # Apply x-axis flip
    x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
    flipped_quaternion = original_quaternion * x_axis_flip
    print(f"\nAfter x-axis flip:")
    print(f"Flipped quaternion: {flipped_quaternion}")
    print(f"Flipped rotation matrix:\n{flipped_quaternion.rotation_matrix}")
    
    # Apply coordinate transformation
    transformed_translation = apply_coordinate_transformation(original_translation)
    transformed_quaternion = apply_rotation_transformation(flipped_quaternion)
    
    print(f"\nAfter coordinate transformation:")
    print(f"Transformed translation: {transformed_translation}")
    print(f"Transformed quaternion: {transformed_quaternion}")
    print(f"Transformed rotation matrix:\n{transformed_quaternion.rotation_matrix}")
    
    # Create transformation matrix
    transformed_transform_matrix = quaternion_to_transform_matrix(
        transformed_quaternion, transformed_translation
    )
    print(f"\nFinal 4x4 transformation matrix:\n{transformed_transform_matrix}")
    
    # Extract final rotation matrix
    final_rotation_matrix = transformed_transform_matrix[:3, :3]
    print(f"\nFinal rotation matrix (3x3):\n{final_rotation_matrix}")
    
    # Convert to Euler angles
    yaw, pitch, roll = rotation_matrix_to_euler_robust(final_rotation_matrix, cam_name)
    
    # Extract camera intrinsics
    fl_x = camera_intrinsic[0, 0]
    fl_y = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    
    print(f"\nCamera intrinsics:")
    print(f"fl_x: {fl_x}")
    print(f"fl_y: {fl_y}")
    print(f"cx: {cx}")
    print(f"cy: {cy}")
    
    return {
        'transform_matrix': transformed_transform_matrix,
        'translation': transformed_translation,
        'euler_angles': (yaw, pitch, roll),
        'euler_degrees': (np.degrees(yaw), np.degrees(pitch), np.degrees(roll)),
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': cx,
        'cy': cy
    }


def main():
    """Main function to process two camera examples."""
    
    # Camera data examples
    cam_front = {
        "token": "71ae8dbd707b4dae9ab36a049c1aaf42",
        "sensor_token": "725903f5b62f56118f4094b46a4470d8",
        "translation": [1.671, -0.026, 1.536],
        "rotation": [0.5008123506024099, -0.496820732721925, 0.4963493647221966, -0.5059579598757297],
        "camera_intrinsic": [
            [1262.8093578767177, 0.0, 786.6784634591471],
            [0.0, 1262.8093578767177, 437.9890946201144],
            [0.0, 0.0, 1.0]
        ]
    }
    
    # CAM_FRONT2 data
    cam_front2 = {
        "token": "2e64b091b3b146a390c2606b9081343c",
        "sensor_token": "725903f5b62f56118f4094b46a4470d8",
        "translation": [1.70079118954, 0.0159456324149, 1.51095763913],
        "rotation": [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
        "camera_intrinsic": [
            [1266.417203046554, 0.0, 816.2670197447984],
            [0.0, 1266.417203046554, 491.50706579294757],
            [0.0, 0.0, 1.0]
        ]
    }
    
    # Process both cameras
    result_front = process_camera_data(cam_front, "CAM_FRONT")
    result_front2 = process_camera_data(cam_front2, "CAM_FRONT2")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nCAM_FRONT results:")
    print(f"Final translation: {result_front['translation']}")
    print(f"Final Euler angles (rad): {result_front['euler_angles']}")
    print(f"Final Euler angles (deg): {result_front['euler_degrees']}")
    
    print(f"\nCAM_FRONT2 results:")
    print(f"Final translation: {result_front2['translation']}")
    print(f"Final Euler angles (rad): {result_front2['euler_angles']}")
    print(f"Final Euler angles (deg): {result_front2['euler_degrees']}")
    
    # Compute differences
    trans_diff = result_front2['translation'] - result_front['translation']
    euler_diff = np.array(result_front2['euler_angles']) - np.array(result_front['euler_angles'])
    euler_deg_diff = np.array(result_front2['euler_degrees']) - np.array(result_front['euler_degrees'])
    
    print(f"\nDIFFERENCES (CAM_FRONT2 - CAM_FRONT):")
    print(f"Translation difference: {trans_diff}")
    print(f"Euler angles difference (rad): {euler_diff}")
    print(f"Euler angles difference (deg): {euler_deg_diff}")


if __name__ == "__main__":
    main()