import pygame
import numpy as np
import copy

TAILLEX=800
TAILLEY=800
screen = pygame.display.set_mode((TAILLEX,TAILLEY))

FPSM=30
FPSA=60
pygame.init()

def normalise(vect):
    norm = np.linalg.vector_norm(vect)
    if norm > 1e-8:
        return vect / norm
    return np.zeros_like(vect)

class Runner:
    def __init__(self):
        self.v=0
        self.Ec=0
        self.xy=np.array([100,100])
        self.Vmouvement=np.array([0,0])
        self.Acoef=0.8
       
        self.color=(255,255,255)
        self.radius=5

    def updateVector(self,target,Va):
        t1=Va*2*self.Acoef #assume m=1 then momentum is v; Force norme we oppose to Vmouvement
        dire=target-self.xy
        dire=normalise(dire)

        Force=dire*t1*1/FPSM
        new=Force+self.Vmouvement
        v=np.linalg.norm(new)
        return v,new

    def reset(self):
        self.xy=np.array([0,0])
        self.v=0
        self.Vmouvement=np.array([0,0])
        return
    
    def move(self,mx,my,va):
        v,vec=self.updateVector(np.array([mx,my]),va)
        self.v=v
        self.Vmouvement=vec
        self.xy=np.add(self.xy,self.Vmouvement)
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.xy[0].item()),int(self.xy[1].item())),self.radius)
        #pygame.draw.line(screen,(255,0,0),(int(self.x),int(self.y)),(int(self.x+self.vector[0]),int(self.y+self.vector[1])),2)
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
    print(runner.v)
    pygame.draw.circle(screen,(0,255,0),(pos[0],pos[1]),5)
    pygame.display.update()
    clock.tick(FPSM)