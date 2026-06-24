import pybullet as p
import pybullet_data

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)

for i in range(100):
    p.stepSimulation()

width = 640
height = 480
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 0.5],
    distance=3,
    yaw=45,
    pitch=-30,
    roll=0,
    upAxisIndex=2
)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=float(width)/height,
    nearVal=0.1,
    farVal=100.0
)

image_data = p.getCameraImage(
    width,
    height,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=p.ER_TINY_RENDERER
)

rgb_image = image_data[2]

p.disconnect()
