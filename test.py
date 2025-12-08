import pygame
import numpy as np
import copy

TAILLEX=500
TAILLEY=500
MAXDISTANCE=np.sqrt(TAILLEX**2+TAILLEY**2)
screen = pygame.display.set_mode((TAILLEX,TAILLEY))

FPSM=30
FPSA=60
pygame.init()
PX=100
n=0
def checkTime(n):
   
    if n==0:
        return 1
    else:
        return 0

def norme(vect):
    return np.sqrt(vect[0]**2+vect[1]**2)

def Vnormale(vect,norme):
    if norme!=0:
        return (vect[0]/norme,vect[1]/norme)
    return (0,0)

class Runner:
    def __init__(self):
        self.x = 100
        self.y = 100
        self.vPs=0
        self.actv=0
        self.Acoef=0.2
        self.Fcoef=0.5
        self.vector=np.array([0,0])

        self.color=(255,255,255)
        self.radius=5

    def updateVector(self,Mx,My,Va):
        vector = [Mx-self.x, My-self.y]
        vector= np.array(vector)
        nvector = norme(vector)
        vnormale = Vnormale(vector,nvector)
        
        target_v = Va * PX
        if target_v < self.actv:
            v=self.vPs+self.Fcoef * (target_v - self.vPs)
        else:
            v=self.vPs+self.Acoef * (target_v - self.vPs)

        nvector1 = norme(self.vector)
        vnormale1 = Vnormale(self.vector,nvector1)

        vector2=np.add(vnormale1,vnormale)
        nvector2 = norme(vector2)
        vnormale2 = Vnormale(vector2,nvector2)

        return v,vnormale2

    def reset(self):
        self.x = 100
        self.y = 100
        self.v=0
        self.vector=np.array([0,0])
        return

    def inTrajectory(self,Mx,My,Second,First):
        cher=np.array([Mx,My])
        v = np.subtract(Second,First)          
        w = cher - First      
        f = cher + (np.dot(w, v) / np.dot(v, v)) * v #RuntimeWarning: invalid value encountered in scalar divide
        return norme(np.subtract(f, First))
    
    def move(self,mx,my,va):
        v,vec=self.updateVector(mx,my,va)
        if self.inTrajectory(mx,my,np.array([self.x+v*vec[0],self.y+v*vec[1]]),np.array([self.x,self.y])) < self.radius:
            v,vec=self.updateVector(mx,my,0)
            pass
        self.vPs=v
        self.vector=vec
        self.x=self.x+self.vector[0]*v*1/FPSM
        self.y=self.y+self.vector[1]*v*1/FPSM
        self.actv+=v*1/FPSM
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.x),int(self.y)),self.radius)
        pygame.draw.line(screen,(255,0,0),(int(self.x),int(self.y)),(int(self.x+self.vector[0]),int(self.y+self.vector[1])),2)
        return

running = True
clock=pygame.time.Clock()
runner=Runner()
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pos = pygame.mouse.get_pos()
    runner.move(pos[0], pos[1],1)
    
    runner.draw(screen)
    print(runner.actv)
    pygame.draw.circle(screen,(0,255,0),(pos[0],pos[1]),5)
    pygame.display.update()
    clock.tick(FPSA)

    