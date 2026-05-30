import numpy as np
import cv2
import glob
import os

def run_calibration():
    # --- CONFIGURATION ---
    # An 8x8 grid of physical squares has exactly 7x7 internal corner intersections
    CHECKERBOARD = (7, 7)
    
    # 1 inch converted to millimeters for precise 3D mapping
    SQUARE_SIZE_MM = 25.4  
    
    # Path where your 17 calibration photos are stored
    IMAGE_FOLDER = "./"

    # Termination criteria for sub-pixel corner accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store 3D physical points and 2D image points
    objpoints = []  # 3D points in real-world space (mm)
    imgpoints = []  # 2D points in image pixel plane

    # Generate ideal 3D coordinates for the grid intersections (Z=0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    # Look for both .jpg and .png extensions just in case
    images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))

    if not images:
        print(f"Error: No images found in '{IMAGE_FOLDER}' folder.")
        print("Please ensure the folder exists and contains your calibration snapshots.")
        return

    print(f"Found {len(images)} images. Processing corner detection...")

    gray = None
    image_shape = None
    successful_detections = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f" Could not read image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape[::-1]  # Format as (width, height)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS
        )

        # If pattern is found, refine and store coordinates
        if ret == True:
            objpoints.append(objp)
            
            # Refine 2D pixels to sub-pixel accuracy
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(refined_corners)
            
            successful_detections += 1
            print(f"  [SUCCESS] Detected corners in: {os.path.basename(fname)}")
        else:
            print(f"  [FAILED]  Could not find 7x7 grid in: {os.path.basename(fname)}")

    print(f"\nDetection complete. Grid found in {successful_detections} out of {len(images)} images.")

    # Proceed to calibration math if we have valid data
    if successful_detections >= 4:
        print("Computing camera intrinsic matrix... (this may take a few seconds)")
        
        # Run OpenCV calibration engine
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None
        )

        # Extract targeted variables
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        # Calculate Total Reprojection Error (measures accuracy)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        reprojection_error = mean_error / len(objpoints)

        # --- OUTPUT RESULTS ---
        print("\n" + "="*45)
        print("         CAMERA CALIBRATION RESULTS         ")
        print("="*45)
        print(f"Image Resolution processed: {image_shape[0]}x{image_shape[1]} px\n")
        print(f"Focal Length X (fx)   : {fx:.4f} pixels")
        print(f"Focal Length Y (fy)   : {fy:.4f} pixels")
        print(f"Principal Point X (cx): {cx:.4f} pixels")
        print(f"Principal Point Y (cy): {cy:.4f} pixels")
        print("-" * 45)
        print(f"Total Reprojection Error: {reprojection_error:.4f} pixels")
        
        if reprojection_error < 0.5:
            print("Status: EXCELLENT calibration quality.")
        elif reprojection_error <= 1.0:
            print("Status: GOOD / Acceptable tracking quality.")
        else:
            print("Status: POOR accuracy. Check for cardboard bending or blurry images.")
        print("="*45)
        
        print("\nCopy-paste this matrix directly into your PnP depth tracking code:")
        print(f"camera_matrix = np.array([\n    [{fx:.1f}, 0.0, {cx:.1f}],\n    [0.0, {fy:.1f}, {cy:.1f}],\n    [0.0, 0.0, 1.0]\n], dtype=np.float32)")
        print(f"dist_coeffs = np.array({np.array2string(dist.ravel(), separator=', ')}, dtype=np.float32)")

    else:
        print("\n[-] Calibration aborted.")
        print("OpenCV needs at least 4 successfully parsed frames to calculate optics math.")
        print("Try retaking photos with better lighting, less tilt angle, or sharper focus.")

if __name__ == "__main__":
    run_calibration()
