import numpy as np
from numpy import pi,cos,sin,sqrt
import csv
import os.path
import matplotlib.colors as mpl
import matplotlib.pyplot as plt

class Magnet():
  #assum the current is 1 ampere
  __segmentCircleCount=16
  __z_offset=0
  __x_offset=0
  __z_step=0
  __x_step=0
  __z_count=0
  __x_count=0
  __density=0.0075 #g/mm^3

  __modifier=dict()
  __tXtranslation=np.array([[-0.1,0,0],[0.1,0,0]])
  __tXrotation=np.tile(np.expand_dims(np.eye(3),0),(2,1,1))
  __tYtranslation=np.array([[0,-0.1,0],[0,0.1,0]])
  __tYrotation=np.tile(np.expand_dims(np.eye(3),0),(2,1,1))
  __tZtranslation=np.array([[0,0,-0.1],[0,0,0.1]])
  __tZrotation=np.tile(np.expand_dims(np.eye(3),0),(2,1,1))
  __rXtranslation=np.zeros((2,3))
  __rXrotation=[]
  __rYtranslation=np.zeros((2,3))
  __rYrotation=[]
  for i in [-1,1]:
    __rXrotation.append(np.array([
        [1,0,0],
        [0,cos(i/180*pi),-sin(i/180*pi)],
        [0,sin(i/180*pi),cos(i/180*pi)]
      ]))
    __rYrotation.append(np.array([
        [cos(i/180*pi),0,sin(i/180*pi)],
        [0,1,0],
        [-sin(i/180*pi),0,cos(i/180*pi)]
      ]))
  __rXrotation=np.array(__rXrotation)
  __rYrotation=np.array(__rYrotation)
  __modifier['tX']=(__tXtranslation,__tXrotation)
  __modifier['tY']=(__tYtranslation,__tYrotation)
  __modifier['tZ']=(__tZtranslation,__tZrotation)
  __modifier['rX']=(__rXtranslation,__rXrotation)
  __modifier['rY']=(__rYtranslation,__rYrotation)
  __gradient=None
  __cache=None

  def __init__(self,
          thickness,
          outerDiameter,
          innerDiameter=0,
          wireDiameter=0.5,
          radiusCount=1,
          currentDensity=720, #current per diameter of wire
         ):
    self.__thickness=thickness
    self.__outerDiameter=outerDiameter
    self.__innerDiameter=innerDiameter
    self.__wireDiameter=wireDiameter
    self.__radiusCount=radiusCount
    self.__currentMagnitude=currentDensity*wireDiameter
    self.__mass=(self.__outerDiameter**2-self.__innerDiameter**2)*pi/4*self.__thickness*self.__density
    self.__inertia=self.__mass/12*(3/4*self.__outerDiameter**2+3/4*self.__innerDiameter**2+self.__thickness**2)
  
  def genCache(self,fileName,parameters,override=False):
    if not override and (type(fileName) is str) and os.path.isfile(fileName):
      with open(fileName, newline='') as csvfile:
        data = list(csv.reader(csvfile))
      dataParameters=data[0]
      allFit=True
      for i in range(len(dataParameters)):
        if parameters[i]!=float(dataParameters[i]):
          allFit=False
          break
      if allFit:
        self.__z_offset=parameters[0]
        self.__x_offset=parameters[1]
        self.__z_step=parameters[2]
        self.__x_step=parameters[3]
        self.__z_count=parameters[4]
        self.__x_count=parameters[5]
        grid=np.array(data[1:]).astype(float).reshape((2,self.__z_count,self.__x_count))
        grid=np.moveaxis(grid,0,-1)
        self.__cache=grid
        return
    self.__z_offset=parameters[0]
    self.__x_offset=parameters[1]
    self.__z_step=parameters[2]
    self.__x_step=parameters[3]
    self.__z_count=parameters[4]
    self.__x_count=parameters[5]

    # generate cache of magnetic field
    grid=[]
    for z in range(self.__z_offset,self.__z_offset+self.__z_step*self.__z_count,self.__z_step):
      for x in range(self.__x_offset,self.__x_offset+self.__x_step*self.__x_count,self.__x_step):
        grid.append(self.magneticField(z,x))
    grid=np.array(grid).reshape((self.__z_count,self.__x_count,3))
    self.__cache=np.stack((grid[:,:,0],grid[:,:,2]),axis=2)

    with open(fileName, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow([self.__z_offset,self.__x_offset,self.__z_step,self.__x_step,self.__z_count,self.__x_count])
      for i in range(grid.shape[0]):
        writer.writerow(grid[i,:,0])
      for i in range(grid.shape[0]):
        writer.writerow(grid[i,:,2])
  
  def genCacheGradient(self,fileName,parameters,override=False):
    if not override and (type(fileName) is str) and os.path.isfile(fileName):
      with open(fileName, newline='') as csvfile:
        data = list(csv.reader(csvfile))
      dataParameters=data[0]
      allFit=True
      for i in range(len(dataParameters)):
        if parameters[i]!=float(dataParameters[i]):
          allFit=False
          break
      if allFit:
        self.__z_offset=int(data[0][0])
        self.__x_offset=int(data[0][1])
        self.__z_step=int(data[0][2])
        self.__x_step=int(data[0][3])
        self.__z_count=int(data[0][4])
        self.__x_count=int(data[0][5])
        gradient=np.array(data[1:]).astype(float).reshape((5,self.__z_count,self.__x_count))
        gradient=np.moveaxis(gradient,0,-1)
        self.__gradient=gradient
        return
    self.__z_offset=parameters[0]
    self.__x_offset=parameters[1]
    self.__z_step=parameters[2]
    self.__x_step=parameters[3]
    self.__z_count=parameters[4]
    self.__x_count=parameters[5]

    gradient=[]
    for z in range(self.__z_offset,self.__z_offset+self.__z_step*self.__z_count,self.__z_step):
      for x in range(self.__x_offset,self.__x_offset+self.__x_step*self.__x_count,self.__x_step):
        gradient.append(self.magneticFieldGradient(z,x))
    gradient=np.array(gradient).reshape((self.__z_count,self.__x_count,3,3))
    self.__gradient=np.stack((gradient[:,:,0,0],gradient[:,:,0,2],gradient[:,:,2,0],gradient[:,:,2,2],gradient[:,:,1,1]),axis=2)

    with open(fileName, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow([self.__z_offset,self.__x_offset,self.__z_step,self.__x_step,self.__z_count,self.__x_count])
      for i in range(gradient.shape[0]):
        writer.writerow(gradient[i,:,0,0])
      for i in range(gradient.shape[0]):
        writer.writerow(gradient[i,:,0,2])
      for i in range(gradient.shape[0]):
        writer.writerow(gradient[i,:,2,0])
      for i in range(gradient.shape[0]):
        writer.writerow(gradient[i,:,2,2])
      for i in range(gradient.shape[0]):
        writer.writerow(gradient[i,:,1,1])
  
  def magneticField(self,z0,x=0,y=0):
    if self.__cache is not None and self.__z_offset<z0<self.__z_offset+self.__z_step*(self.__z_count-1) and self.__x_offset<x<self.__x_offset+self.__x_step*(self.__x_count-1):
      return self.__magneticField(z0,x)
    coilCountAlongZ=round(self.__thickness/self.__wireDiameter)
    segmentCircleCount=self.__segmentCircleCount
    B=np.zeros(3)
    radiusList=[]
    for count in range(self.__radiusCount):
      radiusList.append((self.__outerDiameter/2-self.__wireDiameter*count,1))
      if self.__innerDiameter>0:
        radiusList.append((self.__innerDiameter/2+self.__wireDiameter*count,-1))
    for j in range(segmentCircleCount): # circle segment
      theta=2*pi/segmentCircleCount*j
      for l in range(coilCountAlongZ): # layer of circle
        z=z0+l*self.__wireDiameter-self.__wireDiameter*(coilCountAlongZ-1)/2
        for r,direction in radiusList:
          # B+=direction*np.array([r*z*cos(theta),r*z*sin(theta),r**2-r*x*cos(theta)])/((x-r*cos(theta))**2+(-r*sin(theta))**2+z**2)**1.5*2*pi/segmentCircleCount
          B+=direction*np.array([r*z*cos(theta),r*z*sin(theta),r**2-r*x*cos(theta)-r*y*sin(theta)])/((x-r*cos(theta))**2+(y-r*sin(theta))**2+z**2)**1.5*2*pi/segmentCircleCount
    return B*self.__currentMagnitude # unit is gauss
  
  def __magneticField(self,z,x=0):
    z2=(z-self.__z_offset)/self.__z_step
    x2=(x-self.__x_offset)/self.__x_step
    row=int(z2)
    col=int(x2)
    tz=z2%1
    tx=x2%1
    Bx=self.__cache[row,col,0]*(1-tx)*(1-tz)+self.__cache[row,col+1,0]*tx*(1-tz)+self.__cache[row+1,col,0]*(1-tx)*tz+self.__cache[row+1,col+1,0]*tx*tz
    Bz=self.__cache[row,col,1]*(1-tx)*(1-tz)+self.__cache[row,col+1,1]*tx*(1-tz)+self.__cache[row+1,col,1]*(1-tx)*tz+self.__cache[row+1,col+1,1]*tx*tz
    return np.array([Bx,0,Bz])
  
  def magneticFieldGradient(self,z0,x=0):
    if self.__gradient is not None and self.__z_offset<z0<self.__z_offset+self.__z_step*(self.__z_count-1) and self.__x_offset<x<self.__x_offset+self.__x_step*(self.__x_count-1):
      return self.__magneticFieldGradient(z0,x)
    coilCountAlongZ=round(self.__thickness/self.__wireDiameter)
    segmentCircleCount=self.__segmentCircleCount
    dBdx=np.zeros(3)
    dBdy=np.zeros(3)
    dBdz=np.zeros(3)
    for j in range(segmentCircleCount): # circle segment
      theta=2*pi/segmentCircleCount*j
      for l in range(coilCountAlongZ): # layer of circle
        z=z0+l*self.__wireDiameter-self.__wireDiameter*(coilCountAlongZ-1)/2
        for count in range(self.__radiusCount): # different radius of circle
          r=self.__outerDiameter/2-self.__wireDiameter*count
          scale1=((x-r*cos(theta))**2+(-r*sin(theta))**2+z**2)**-1.5
          vector1=np.array([
            z*r*cos(theta),
            z*r*sin(theta),
            -x*r*cos(theta)+r**2
          ])*(-1.5)*((x-r*cos(theta))**2+(-r*sin(theta))**2+z**2)**-2.5*2
          dBdx+=np.array([0,0,-r*cos(theta)])*scale1+vector1*(x-r*cos(theta))
          dBdy+=np.array([0,0,-r*sin(theta)])*scale1+vector1*(-r*sin(theta))
          dBdz+=np.array([r*cos(theta),r*sin(theta),0])*scale1+vector1*z
    return (np.stack((dBdx,dBdy,dBdz),axis=1)*2*pi/segmentCircleCount)*self.__currentMagnitude # unit is Gauss/mm
  
  def __magneticFieldGradient(self,z,x=0):
    z2=(z-self.__z_offset)/self.__z_step
    x2=(x-self.__x_offset)/self.__x_step
    row=int(z2)
    col=int(x2)
    tz=z2%1
    tx=x2%1
    B00=self.__gradient[row,col,0]*(1-tx)*(1-tz)+self.__gradient[row,col+1,0]*tx*(1-tz)+self.__gradient[row+1,col,0]*(1-tx)*tz+self.__gradient[row+1,col+1,0]*tx*tz
    B02=self.__gradient[row,col,1]*(1-tx)*(1-tz)+self.__gradient[row,col+1,1]*tx*(1-tz)+self.__gradient[row+1,col,1]*(1-tx)*tz+self.__gradient[row+1,col+1,1]*tx*tz
    B20=self.__gradient[row,col,2]*(1-tx)*(1-tz)+self.__gradient[row,col+1,2]*tx*(1-tz)+self.__gradient[row+1,col,2]*(1-tx)*tz+self.__gradient[row+1,col+1,2]*tx*tz
    B22=self.__gradient[row,col,3]*(1-tx)*(1-tz)+self.__gradient[row,col+1,3]*tx*(1-tz)+self.__gradient[row+1,col,3]*(1-tx)*tz+self.__gradient[row+1,col+1,3]*tx*tz
    B11=self.__gradient[row,col,4]*(1-tx)*(1-tz)+self.__gradient[row,col+1,4]*tx*(1-tz)+self.__gradient[row+1,col,4]*(1-tx)*tz+self.__gradient[row+1,col+1,4]*tx*tz
    return np.array([[B00,0,B02],[0,B11,0],[B20,0,B22]])
  
  def magneticField4(self,z,x,y,xOffset,yOffset=0,preTiltAngle=0):
    B=np.zeros(3)
    fourMagnet=np.array([
        [x+xOffset,y+yOffset,z],
        [x-xOffset,y-yOffset,z],
        [x+yOffset,y-xOffset,z],
        [x-yOffset,y+xOffset,z]
    ])
    preTiltY=np.array([
        [cos(preTiltAngle/180*pi),0,sin(preTiltAngle/180*pi)],
        [0,1,0],
        [-sin(preTiltAngle/180*pi),0,cos(preTiltAngle/180*pi)]
      ])
    for i in range(4):
      psi=np.arctan2(fourMagnet[i][1]-y,fourMagnet[i][0]-x)
      rotation=np.array([
        [cos(psi),-sin(psi),0],
        [sin(psi),cos(psi),0],
        [0,0,1]
      ])
      x2,y2,z2=(rotation@preTiltY).T@fourMagnet[i]
      B+=rotation@preTiltY@self.magneticField(z2,x2,y2)
    return B
  
  def magneticField2(self,z0,x,y=0):
    B=np.zeros(3)
    coilCountAlongZ=round(self.__thickness/self.__wireDiameter)
    for i in range(self.__segmentCircleCount):
      for j in range(self.__segmentCircleCount):
        positionX=self.__outerDiameter/2*((i*2+1)/self.__segmentCircleCount-1)
        positionY=self.__outerDiameter/2*((j*2+1)/self.__segmentCircleCount-1)
        if positionX**2+positionY**2>self.__outerDiameter**2/4 or positionX**2+positionY**2<self.__innerDiameter**2/4:
          continue
        for l in range(coilCountAlongZ): # layer of circle
          m=np.array([0,0,(self.__outerDiameter/self.__segmentCircleCount)**2])
          z=z0+l*self.__wireDiameter-self.__wireDiameter*(coilCountAlongZ-1)/2
          r=np.array([x,y,z])
          rscale=sqrt(np.sum(r**2))
          B+=3*r*np.dot(m,r)/rscale**5-m/rscale**3
    return B*self.__currentMagnitude
  
  @property
  def Measurement(self):
    return self.__thickness,self.__outerDiameter,self.__innerDiameter,self.__wireDiameter,self.__radiusCount
  @property
  def Mass(self):
    return self.__mass # unit is g
  @property
  def Inertia(self):
    return self.__inertia # unit is g*mm^2
  @property
  def CurrentMagnitude(self):
    return self.__currentMagnitude
  
  def forceAndTorque(self,z,x,y=0,preTiltAngle=0,translation=np.zeros(3),rotation=np.diag([1,1,1]),floatMagnet=None):
    zAxisAngle=np.arctan2(y,x)
    preTiltZ=np.array([
        [cos(zAxisAngle),-sin(zAxisAngle),0],
        [sin(zAxisAngle),cos(zAxisAngle),0],
        [0,0,1]
      ])
    preTiltY=np.array([
        [cos(preTiltAngle/180*pi),0,sin(preTiltAngle/180*pi)],
        [0,1,0],
        [-sin(preTiltAngle/180*pi),0,cos(preTiltAngle/180*pi)]
      ])
    preTilt=preTiltZ@preTiltY
    x2,y2,z2=preTilt.T@np.array([x,y,z])
    rotation2=preTilt.T@rotation
    translation2=preTilt.T@translation
    result=self.__forceAndTorque(z2,x2,y2,translation2,rotation2,floatMagnet)
    return (preTilt@result.reshape((2,3)).T).T.flatten()
  
  def __forceAndTorque(self,z,x,y=0,translation=np.zeros(3),rotation=np.diag([1,1,1]),floatMagnet=None):
    thickness,outerDiameter,innerDiameter,wireDiameter,radiusCount=3,10,0,0.5,1
    currentMagnitude=self.__currentMagnitude
    if isinstance(floatMagnet,Magnet):
      thickness,outerDiameter,innerDiameter,wireDiameter,radiusCount=floatMagnet.Measurement
      currentMagnitude=floatMagnet.CurrentMagnitude
    force=[]
    arm=[]
    coilCountAlongZ=round(thickness/wireDiameter)
    radiusList=[]
    for count in range(radiusCount):
      radiusList.append((outerDiameter/2-wireDiameter*count,1))
      if innerDiameter>0:
        radiusList.append((innerDiameter/2+wireDiameter*count,-1))

    for j in range(self.__segmentCircleCount):  # segment of circle
      phi=2*pi/self.__segmentCircleCount*j
      for l in range(coilCountAlongZ): # layer of circle
        zDisplacement=l*wireDiameter-wireDiameter*(coilCountAlongZ-1)/2
        for magnetRadius,direction in radiusList: # different radius of circle
          segment=rotation@np.array([magnetRadius*cos(phi),magnetRadius*sin(phi),zDisplacement])
          arm.append(segment.copy())
          segment+=translation
          current=rotation@np.array([-sin(phi),cos(phi),0])*2*pi*magnetRadius/self.__segmentCircleCount
          displacment=segment+np.array([x,y,z])
          dh=sqrt(displacment[0]**2+displacment[1]**2)
          dv=displacment[2]
          psi=np.arctan2(displacment[1],displacment[0])
          rotationB=np.array([
            [cos(psi),-sin(psi),0],
            [sin(psi),cos(psi),0],
            [0,0,1]
          ])
          B=self.magneticField(dv,dh)
          force.append(direction*np.cross(current,rotationB@B))
    force=np.array(force)
    netForce=np.sum(force,axis=0)*0.1 # unit is 10*gauss*mm*A = g*mm/s^2 = 10^-6 N
    arm=np.array(arm)
    torque=np.cross(arm,force)
    netTorque=np.sum(torque,axis=0)*0.1 # unit is 10*gauss*mm*A*mm = g*mm^2/s^2 = 10^-9 N*m
    return np.hstack((netForce,netTorque))*currentMagnitude
  
  def forceAndTorque2(self,z,x,y=0,translation=np.zeros(3),rotation=np.diag([1,1,1]),floatMagnet=None):
    thickness,outerDiameter,innerDiameter,wireDiameter=3,10,0,0.5
    current=self.__currentMagnitude
    if isinstance(floatMagnet,Magnet):
      thickness,outerDiameter,innerDiameter,wireDiameter,_=floatMagnet.Measurement
      current=floatMagnet.CurrentMagnitude
    coilCountAlongZ=round(thickness/wireDiameter)
    force=[]
    arm=[]
    pureTorque=np.zeros(3)
    area=0
    for i in range(self.__segmentCircleCount):
      for j in range(self.__segmentCircleCount):
        positionX=outerDiameter/2*((i*2+1)/self.__segmentCircleCount-1)
        positionY=outerDiameter/2*((j*2+1)/self.__segmentCircleCount-1)

        if positionX**2+positionY**2>outerDiameter**2/4 or positionX**2+positionY**2<innerDiameter**2/4:
          continue
        area+=1
        for l in range(coilCountAlongZ): # layer of circle
          zDisplacement=l*wireDiameter-wireDiameter*(coilCountAlongZ-1)/2
          segment=rotation@np.array([positionX,positionY,zDisplacement])
          arm.append(segment.copy())
          segment+=translation
          m=rotation@np.array([0,0,(outerDiameter/self.__segmentCircleCount)**2])
          displacment=segment+np.array([x,y,z])
          dh=sqrt(displacment[0]**2+displacment[1]**2)
          dv=displacment[2]
          psi=np.arctan2(displacment[1],displacment[0])
          rotationB=np.array([
            [cos(psi),-sin(psi),0],
            [sin(psi),cos(psi),0],
            [0,0,1]
          ])
          deltaB=self.magneticFieldGradient(dv,dh)
          force.append(m@rotationB@deltaB@rotationB.T)
          B=self.magneticField(dv,dh)
          pureTorque+=np.cross(m,rotationB@B)
    area*=(outerDiameter/self.__segmentCircleCount)**2
    force=np.array(force)
    netForce=np.sum(force,axis=0)
    arm=np.array(arm)
    torque=np.cross(arm,force)
    netTorque=np.sum(torque,axis=0)+pureTorque
    return np.hstack((netForce,netTorque))/area*pi*(outerDiameter**2/4-innerDiameter**2/4)*current*0.1
  
  def Linearization(self,z,x,y=0,preTiltAngle=0,operations=['tX','tY','tZ','rX','rY'],floatMagnet=None):
    oneOverMassAndInertia=np.eye(5)
    if isinstance(floatMagnet,Magnet):
      magnetMass=floatMagnet.Mass
      magnetInertia=floatMagnet.Inertia
      oneOverMassAndInertia=np.diag(np.array([1/magnetMass,1/magnetMass,1/magnetMass,1/magnetInertia,1/magnetInertia]))
    A=[]
    for operation in operations:
      translations,rotations=self.__modifier[operation]
      forceAndTorqueList=[]
      for i in range(translations.shape[0]):
        forceAndTorqueList.append(self.forceAndTorque(z,x,y,preTiltAngle,translation=translations[i],rotation=rotations[i],floatMagnet=floatMagnet))
      temp=(forceAndTorqueList[1]-forceAndTorqueList[0])[:-1]
      if operation=='rX' or operation=='rY':
        temp/=2*pi/180
      else:
        temp/=0.2
      A.append(temp.copy())
    A=np.array(A).T # acc unit is mm/s^2, angular acc unit is 1/s^2
    return oneOverMassAndInertia@A
  
  def LinearizationA(self,z,x,y=0,preTiltAngle=0,floatMagnet=None):
    B1=self.forceAndTorque(z,x,y,preTiltAngle,floatMagnet=floatMagnet)[:-1]
    if isinstance(floatMagnet,Magnet):
      magnetMass=floatMagnet.Mass
      magnetInertia=floatMagnet.Inertia
      oneOverMassAndInertia=np.array([1/magnetMass,1/magnetMass,1/magnetMass,1/magnetInertia,1/magnetInertia])
    if x==0 and y==0:
      A1=self.Linearization(z,x,y,preTiltAngle,operations=['tX','rY','tZ'],floatMagnet=floatMagnet)
      # return A1[(0,-1),:-1],A1[2,-1],(oneOverMassAndInertia*B1)[2]
      return np.hstack((A1[(0,-1),:-1].flatten(),np.array([A1[2,-1],(oneOverMassAndInertia*B1)[2]])))
    else:
      A1=self.Linearization(z,x,y,preTiltAngle,floatMagnet=floatMagnet)
      # A2=np.array([
      #   [A1[0,0]+A1[1,1],0,0,0,A1[0,4]-A1[1,3]],
      #   [0,A1[0,0]+A1[1,1],0,A1[1,3]-A1[0,4],0],
      #   [0,0,A1[2,2]*2,0,0],
      #   [0,A1[3,1]-A1[4,0],0,A1[3,3]+A1[4,4],0],
      #   [A1[4,0]-A1[3,1],0,0,0,A1[3,3]+A1[4,4]]
      # ])*2
      # return (A2[(0,-1),:])[:,(0,-1)],A2[2,2],B1[2]*4
      A3=np.array([[A1[0,0]+A1[1,1],A1[0,4]-A1[1,3]],[A1[4,0]-A1[3,1],A1[3,3]+A1[4,4]]])*2
      # return A3,A1[2,2]*4,(oneOverMassAndInertia*B1)[2]*4
      return np.hstack((A3.flatten(),np.array([A1[2,2]*4,(oneOverMassAndInertia*B1)[2]*4])))
  
  def LinearizationB(self,z,x,y=0,preTiltAngle=0,floatMagnet=None):
    B1=self.forceAndTorque(z,x,y,preTiltAngle,floatMagnet=floatMagnet)[:-1]
    if isinstance(floatMagnet,Magnet):
      magnetMass=floatMagnet.Mass
      magnetInertia=floatMagnet.Inertia
      oneOverMassAndInertia=np.diag(np.array([1/magnetMass,1/magnetMass,1/magnetMass,1/magnetInertia,1/magnetInertia]))
    if x==0 and y==0:
      return oneOverMassAndInertia@(B1.reshape((5,1)))
    else:
      B2=oneOverMassAndInertia@np.array([
          [B1[0],B1[1],B1[2],B1[3],B1[4]], # coil at -x
          [-B1[0],-B1[1],B1[2],-B1[3],-B1[4]],  # coil at x
          [-B1[1],B1[0],B1[2],-B1[4],B1[3]],  # coil at -y
          [B1[1],-B1[0],B1[2],B1[4],-B1[3]],  # coil at y
      ]).T
      return B2


def compoundMagnet(magnets,baseMagnet,xoffset,centerZ_ref=0):
  mass=0
  inertia=0
  centerZ=0
  for i in range(len(magnets)):
    mass+=magnets[i][1].Mass
    inertia+=magnets[i][1].Inertia
    centerZ+=magnets[i][0]*magnets[i][1].Mass
  centerZ/=mass
  if centerZ_ref!=0:
    centerZ=centerZ_ref
  distances=[]
  for i in range(len(magnets)):
    d=magnets[i][0]-centerZ
    distances.append(d)
    inertia+=magnets[i][1].Mass*d**2
  oneOverMassAndInertia=np.array([1/mass,1/mass,1/inertia,1/inertia,1/mass,1/mass])
  def compoundMatrix(d):
    matrix=np.array([
      [1,0,0,0,0,0],
      [d,1,0,0,0,0],
      [d,0,1,0,0,0],
      [d**2,d,d,1,0,-d],
      [0,0,0,0,1,0],
      [0,0,0,0,0,1]
    ])
    return matrix
  total=np.zeros(6)
  for i in range(len(magnets)):
    magnet=magnets[i][1]
    MassAndInertia=np.array([magnet.Mass,magnet.Mass,magnet.Inertia,magnet.Inertia,magnet.Mass,magnet.Mass])
    forceAndTorque=magnets[i][2]*baseMagnet.LinearizationA(magnets[i][0],xoffset,floatMagnet=magnet)*MassAndInertia
    total=total+compoundMatrix(distances[i])@forceAndTorque
  return total*oneOverMassAndInertia

#Orange:50,255,115,10
#Yellow:70,252,182,11
#Light Green:100,135,196,5
#Green:150,0,156,51
#Blue:200,0,116,255
#Dark Blue:300,20,42,146
cdict = {'red':[
      [0.0,  1.0, 1.0],
      [0.167,  1.0, 1.0],
      [0.233,  1.0, 1.0],
      [0.333,  0.53, 0.53],
      [0.5,  0.0, 0.0],
      [0.667,  0.0, 0.0],
      [1.0,  0.08, 0.08]],
    'green':[
      [0.0,  1.0, 1.0],
      [0.167,  0.45, 0.45],
      [0.233,  0.71, 0.71],
      [0.333,  0.77, 0.77],
      [0.5,  0.61, 0.61],
      [0.667,  0.45, 0.45],
      [1.0,  0.16, 0.16]],
    'blue':[
      [0.0,  1.0, 1.0],
      [0.167,  0.04, 0.04],
      [0.233,  0.04, 0.04],
      [0.333,  0.02, 0.02],
      [0.5,  0.2, 0.2],
      [0.667,  1.0, 1.0],
      [1.0,  0.57, 0.57]]}
magneticCmap=mpl.LinearSegmentedColormap('magneticCmap', segmentdata=cdict, N=256)

def plot_linearmap(newcmp):
  rgba = newcmp(np.linspace(0, 1, 256))
  fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
  col = ['r', 'g', 'b']
  for xx in [0.25, 0.5, 0.75]:
      ax.axvline(xx, color='0.7', linestyle='--')
  for i in range(3):
      ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
  ax.set_xlabel('index')
  ax.set_ylabel('RGB')
  plt.show()