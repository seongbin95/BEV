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
images = glob.glob('chass\\*.png')
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
#retval, rvec, tvec = cv2.solvePnP(objectPoints = objpoints, imgpoints = imgpoints, cameraMatrix = mtx, distCoeffs = dist, rvec = rvecs, tvec = tvecs)
cv2.destroyAllWindows()

#print(objpoints, imgpoints)
img = cv2.imread('chass\\original_screenshot_11.02.20164.png')
h, w = img.shape[:2]
newcameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx, dist, None, newcameraMtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibRes.png', dst)
np.savetxt('dist.txt',dist)#,fmt='%3.18f',newline='n',delimiter = ' , ')
rvecs_arr = np.array(rvecs).reshape(np.array(rvecs).shape[0], -1)
np.savetxt('rvecs.txt',rvecs_arr)#,fmt='%3.18f',newline='n',delimiter = ' , ')
tvecs_arr = np.array(tvecs).reshape(np.array(tvecs).shape[0], -1)
np.savetxt('tvecs.txt',tvecs_arr)#,fmt='%3.18f',newline='n',delimiter = ' , ')
np.savetxt('calibret.txt',[ret])#,fmt='%.18f',newline='n',delimiter = ' , ')
np.savetxt('calibmtx.txt', mtx)#, fmt='%.18f',newline='n',delimiter = ' , ')
np.savez('calib.npz',objectPoints = objpoints, imgpoints = imgpoints ,ret=ret,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
print("========1====ret\n",ret,"\n====mtx\n", mtx,"\n====dist\n", dist,"\n====rvecs\n", rvecs,"\n====tvecs\n", tvecs, "\n======1=======\n")

''''#################################################################################
##// matching pairs

vector<Point3f> objpoints;	// 3d world coordinates

vector<Point2f> imgpoints;	// 2d image coordinates



## camera parameters

double m[] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};	##// intrinsic parameters

Mat A(3, 3, CV_64FC1, m);	##// camera matrix



double d[] = {k1, k2, p1, p2};	##// k1,k2: radial distortion, p1,p2: tangential distortion

Mat distCoeffs(4, 1, CV_64FC1, d);



##// estimate camera pose

Mat rvec, tvec;	// rotation & translation vectors

solvePnP(objectPoints, imagePoints, A, distCoeffs, rvec, tvec)



##// extract rotation & translation matrix

Mat R;

Rodrigues(rvec, R);

Mat R_inv = R.inv();



Mat P = -R_inv*tvec;

double* p = (double *)P.data;



##// camera position

printf("x=%lf, y=%lf, z=%lf", p[0], p[1], p[2]);'''


#################################################################################'''
mean_error = 0
##cv2.solvePnP(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

for i in range(len(objpoints)):
    imgpoints2,_ = cv2.projectPoints(objpoints[i],rvecs[i],tvecs[i],mtx,dist)
    error = cv2.norm(imgpoints[i],imgpoints2,cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("Total error : {0}".format(mean_error/len(objpoints)))

