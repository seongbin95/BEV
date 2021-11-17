import cv2
import math
import numpy as np

capture = cv2.VideoCapture('1.avi')
#capture.set(3,640)#320
#capture.set(4,480)#240

capwidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
capheight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

### Camera coordinate to vehicle coordinate(ISO)
def RotX(Pitch):
      Pitch = np.deg2rad(Pitch)
      return [[1, 0, 0], [0, math.cos(Pitch), -math.sin(Pitch)], [0, math.sin(Pitch), math.cos(Pitch)]]

def RotY(Yaw):
      Yaw = np.deg2rad(Yaw)
      return [[math.cos(Yaw), 0, math.sin(Yaw)], [0, 1, 0], [-math.sin(Yaw), 0, math.cos(Yaw)]]

def RotZ(Roll):
      Roll = np.deg2rad(Roll)
      return [[math.cos(Roll), -math.sin(Roll), 0], [math.sin(Roll), math.cos(Roll), 0], [0, 0, 1]]

### Camera parameter setting
ImageSize = (480, 640)
FocalLength = (356.45577614, 327.58890684) #초점 거리
PrinciplePoint = (333.7286132, 225.21681511) # 카메라 렌즈의 중심
IntrinsicMatrix = ((FocalLength[0], 0, 0), (0, FocalLength[1], 0), (PrinciplePoint[0], PrinciplePoint[1], 1)) #내부 파라미터
Height =4.7  #2.1798
Pitch = 3 # 14
Yaw = 0
Roll = 0

### Bird's eye view setting
DistAhead = 30 #30
SpaceToOneSide = 6 #6
BottomOffset = 5 #3.0

OutView = (BottomOffset, DistAhead, -SpaceToOneSide, SpaceToOneSide)
OutImageSize = [math.nan, 250]

WorldHW = (abs(OutView[1]-OutView[0]), abs(OutView[3]-OutView[2]))

Scale = (OutImageSize[1]-1)/WorldHW[1]
ScaleXY = (Scale, Scale)

OutDimFrac = Scale*WorldHW[0]
OutDim = round(OutDimFrac)+1
OutImageSize[0] = OutDim

### Homography Matrix Compute

#Translation Vector
Rotation = np.matmul(np.matmul(RotZ(-Yaw),RotX(90-Pitch)),RotZ(Roll))
TranslationinWorldUnits = (0, 0, Height)
Translation = [np.matmul(TranslationinWorldUnits, Rotation)]

#Rotation Matrix
RotationMatrix = np.matmul(RotY(180), np.matmul(RotZ(-90), np.matmul(RotZ(-Yaw), np.matmul(RotX(90-Pitch), RotZ(Roll)))))

#Camera Matrix
CameraMatrix = np.matmul(np.r_[RotationMatrix, Translation], IntrinsicMatrix)
CameraMatrix2D = np.r_[[CameraMatrix[0]], [CameraMatrix[1]], [CameraMatrix[3]]]

#Compute Vehicle-to-Image Projective Transform
VehicleHomo = np.linalg.inv(CameraMatrix2D)

AdjTransform = ((0, -1, 0), (-1, 0, 0), (0, 0, 1))
BevTransform = np.matmul(VehicleHomo, AdjTransform)

DyDxVehicle = (OutView[3], OutView[1])
tXY = [a*b for a,b in zip(ScaleXY, DyDxVehicle)]

#test = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
ViewMatrix = ((ScaleXY[0], 0, 0), (0, ScaleXY[1], 0), (tXY[0]+1, tXY[1]+1, 1))

T_Bev = np.matmul(BevTransform, ViewMatrix)
T_Bev = np.transpose(T_Bev)



while True:
	ret, frame = capture.read()

	if ret:
		test = frame.copy()
		BirdEyeView = cv2.warpPerspective(frame, T_Bev, (OutImageSize[1], OutImageSize[0]))

		cv2.imshow('test', test)
		cv2.imshow("BEV", BirdEyeView)
		cv2.waitKey(2)
		# cv2.destroyAllWindows()

		if cv2.waitKey(1) > 0:
			break

capture.release()
cv2.destroyAllWindows()
