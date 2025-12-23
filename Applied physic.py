import pygame
import numpy as np
import copy

TAILLEX=800
TAILLEY=800
screen = pygame.display.set_mode((TAILLEX,TAILLEY))

FPSM=120
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
        self.Voldspeed=np.array([0,0])
        self.Force=5000
        self.MaxSpeed=500 #pixels per second

        self.ForceMulti=10
        self.MouvementMulti=0.99

        self.color=(255,255,255)
        self.radius=5

    def updateVector(self,target):
        dire=target-self.xy
        dire=normalise(dire) #to where

        desired_velocity = dire * self.MaxSpeed  # desired velocity
        steering= desired_velocity - self.Voldspeed

        if np.linalg.norm(steering) > self.Force:
            steering = steering * self.Force/np.linalg.norm(steering)

        steering = steering/100

        velocity=steering + self.Voldspeed
        if np.linalg.norm(velocity) > self.MaxSpeed:
            velocity = velocity * self.MaxSpeed/np.linalg.norm(velocity)

        position=self.xy + velocity*(1/FPSM)

        return position,velocity

    def reset(self):
        self.xy=np.array([0,0])
        self.v=0
        self.Voldspeed=np.array([0,0])
        return
    
    def move(self,mx,my): 
        xy,vec=self.updateVector(np.array([mx,my]))
        self.xy=xy
        self.Voldspeed=vec
        
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.xy[0].item()),int(self.xy[1].item())),self.radius)
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
    runner.move(pos[0], pos[1])
    
    runner.draw(screen)
    pygame.draw.circle(screen,(0,255,0),(pos[0],pos[1]),5)
    pygame.display.update()
    clock.tick(FPSM)