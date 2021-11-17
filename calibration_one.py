import numpy as np
import cv2
import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('chass2\\123.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("====ret\n",ret,"\n====mtx\n", mtx,"\n====dist\n", dist,"\n====rvecs\n", rvecs,"\n====tvecs\n", tvecs, "\n====\n")
        cv2.imshow('img',img)
        cv2.waitKey(500)

        ##retval, rvec, tvec = cv2.solvePnP(objectPoints = objp, imgpoints = imgpoints, cameraMatrix = mtx, distCoeffs = dist, rvec = rvecs, tvec = tvecs)
        print("\n=========================================\n")

    else:
        print("xxx")
#print(objpoints)
obj = np.array(objpoints)
img = np.array(imgpoints)

rve = np.array(rvecs)
tve = np.array(tvecs)
print(obj,objpoints)
retval, rvec, tvec = cv2.solvePnP(objectPoints = obj, imagePoints = img, cameraMatrix = mtx, distCoeffs = dist, rvec = rve, tvec = tve)
print( retval ,"/n",rvec,"/n",tvec)
cv2.destroyAllWindows()


mean_error = 0
##cv2.solvePnP(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

for i in range(len(objpoints)):
    imgpoints2,_ = cv2.projectPoints(objpoints[i],rvecs[i],tvecs[i],mtx,dist)
    error = cv2.norm(imgpoints[i],imgpoints2,cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("Total error : {0}".format(mean_error/len(objpoints)))

