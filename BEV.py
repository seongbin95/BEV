import numpy as np
import math
import cv2


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
FocalLength = (309.4362, 344.2161) #초점 거리
PrinciplePoint = (318.9034, 257.5352) # 카메라 렌즈의 중심
IntrinsicMatrix = ((FocalLength[0], 0, 0), (0, FocalLength[1], 0), (PrinciplePoint[0], PrinciplePoint[1], 1)) #내부 파라미터
Height = 2.1798
Pitch = 14 # 14
Yaw = 0
Roll = 0

### Bird's eye view setting
DistAhead = 30
SpaceToOneSide = 6
BottomOffset = 3.0

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



### Main
src = cv2.imread("testImage.png", cv2.IMREAD_COLOR)

BirdEyeView = cv2.warpPerspective(src, T_Bev, (OutImageSize[1], OutImageSize[0]))

cv2.imshow("BEV", BirdEyeView)
cv2.waitKey(0)
cv2.destroyAllWindows()
 

### Distance

#Vehicle to Image
toOriginalImage = np.linalg.inv(np.transpose(T_Bev))
Trans = np.linalg.inv(np.matmul(toOriginalImage, VehicleHomo))
VehiclePoint = [[20, 0]]
XV = np.r_[VehiclePoint[0], np.shape(VehiclePoint)[0]]
UV = np.matmul(XV, Trans)

UV[0:2] = UV[0:2]/UV[2]
ImagePoints = list(map(int, UV[0:2]))

print(ImagePoints)
annotatedBEV1 = cv2.drawMarker(BirdEyeView, ImagePoints, (0,0,255))
cv2.putText(annotatedBEV1, "20 meters", ImagePoints, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))

#cv2.imshow("BEV", annotatedBEV1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Image to Vehicle
toOriginalImage = np.linalg.inv(np.transpose(T_Bev))
Trans = np.matmul(toOriginalImage, VehicleHomo)
ImagePoint = [[120, 400]]

UI = ImagePoint
UI = np.r_[ImagePoint[0], np.shape(ImagePoint)[0]]
XI = np.matmul(UI, Trans)

XI[0:2] = XI[0:2]/XI[2]
XAhead = XI[0]

annotatedBEV2 = cv2.drawMarker(BirdEyeView, ImagePoint[0], (0,0,255))
cv2.putText(annotatedBEV2, str(round(XAhead, 2))+" meters", ImagePoint[0], cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))

cv2.imshow("BEV", annotatedBEV2)
cv2.waitKey(0)
cv2.destroyAllWindows()