import pygame
import numpy as np
import copy
import torch

dtype=torch.float 
device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
device='cpu'
print(f"Using {device} device")
torch.set_default_device(device)


TAILLEX=1000
TAILLEY=800
MAXDISTANCE=torch.sqrt(torch.tensor(TAILLEX**2+TAILLEY**2))
screen = pygame.display.set_mode((TAILLEX,TAILLEY))

FPSM=30 #number of frames for movement calculation
FPSA=60 #number of frames for animation
pygame.init()
PX=100 #max speed, pixels per second


def checkTime(n):
    if n==0:
        return 1
    else:
        return 0

def norme(vect): 
    return torch.sqrt(vect[0]**2+vect[1]**2)

def Vnormale(vect,norme):
    if norme!=0:
        return (vect[0]/norme,vect[1]/norme)
    return (0,0)

class Runner:
    def __init__(self):
        self.StartPos=(TAILLEX//2,TAILLEY//2)
        self.x = self.StartPos[0]
        self.y = self.StartPos[1]
        self.v=0
        self.Acoef=0.01
        self.Fcoef=0.5
        self.vector=torch.tensor([0,0])

        self.color=(255,255,255)
        self.radius=5

    def updateVector(self,Mx,My,Va): #Mx,My : target position , Va : target speed (-1 to 1)
        vector = [Mx-self.x, My-self.y]
        vector= torch.tensor(vector)
        nvector = norme(vector)
        vnormale = Vnormale(vector,nvector)
        
        target_v = Va * PX 
        if target_v < self.v: #acceleration or freinage
            v=self.v+self.Fcoef * (target_v - self.v)
            if round(float(v),6)==0: #avoid big negative like -1e-200
                v=0
        else:
            v=self.v+self.Acoef * (target_v - self.v)

        nvector1 = norme(self.vector)
        vnormale1 = Vnormale(self.vector,nvector1)

        vector2=torch.add(torch.tensor(vnormale1),torch.tensor(vnormale))
        nvector2 = norme(vector2)
        vnormale2 = Vnormale(vector2,nvector2)


        return v,vnormale2 #v : new speed , vnormale2 : new direction vector

    def reset(self): #reset everything, use at the start of test
        self.x = self.StartPos[0]
        self.y = self.StartPos[1]
        self.v=0
        self.vector=torch.tensor([0,0])
        return

    def inTrajectory(self,Mx,My,Second,First): #if runner skip the point he asked to reach he would brake
        cher=torch.tensor([Mx,My])
        v = torch.subtract(Second,First)
        if norme(v)==0:
            return norme(torch.subtract(cher, First))
        else:
            w = cher - First      
            f = cher + (torch.dot(w, v) / torch.dot(v, v)) * v 
            return norme(torch.subtract(f, First)) #projection orthogonal point distance to the line segment
    
    def move(self,mx,my,va):
        v,vec=self.updateVector(mx,my,va)
        if self.inTrajectory(mx,my,torch.tensor([self.x+v*vec[0],self.y+v*vec[1]]),torch.tensor([self.x,self.y])) < self.radius:
            v,vec=self.updateVector(mx,my,0)
            pass
        
        self.vector=vec
        self.v=v
        self.x=self.x+self.vector[0]*v*1/FPSM
        self.y=self.y+self.vector[1]*v*1/FPSM #update position 1 frame by 1 frame
        
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.x),int(self.y)),self.radius)
        pygame.draw.line(screen,(255,0,0),(int(self.x),int(self.y)),(int(self.x+self.vector[0]*self.v),int(self.y+self.vector[1]*self.v)),2)
        return

    def getall(self):
        return (self.x,self.y,self.v,self.vector)

class NeuralNetwork:
    def __init__(self):
        self.runner=Runner()
        self.ANNDisp = [7, 8, 16, 8, 3] #7 input,3 output, 2 hidden layers
        self.WBGenRef={'W':[torch.rand(self.ANNDisp[i+1], self.ANNDisp[i],dtype=dtype) for i in range(len(self.ANNDisp)-1)],
                       'b':[torch.rand(self.ANNDisp[i], 1,dtype=dtype) for i in range(1,len(self.ANNDisp))]} #generation reference ( proud of this fucking way to create weights and biases )
        
        self.Ntest=100
        self.Ngen=50 #100 mutation tested on 50 generations
        self.WBBestInGen={} #best weights and biases in current generation
        self.BestScoreInGen=0 #best score in current generation

        self.ActTestScore=0 #current score in test
        self.timeDuration=10 #seconds
        self.objectif=torch.tensor([100,100]) #target position
        self.sigmoid = torch.nn.Sigmoid()
        
    def Sigmoid(self,x):
        return self.sigmoid(x)
    
    def feed_forward(self,A0,dico): #can't explain shortly, it's basic ANN feed forward with sigmoid activation
        Alist=A0
        for i in range(len(self.ANNDisp)-1):
            Zi=dico['W'][i] @ Alist + dico['b'][i]
            Alist=self.Sigmoid(Zi)
        y_hat = Alist
        return y_hat

    def ScoreUpdate(self,x,y,posx,posy): #every frame score updpate, need to to do better system later
        c=torch.sqrt((x - posx) ** 2 + (y - posy) ** 2)
        ct=1-(c/MAXDISTANCE)
        self.ActTestScore+=self.Sigmoid(ct)
        return 


    def start(self):
        for i in range(self.Ngen): #generation loop
            for y in range(self.Ntest): #mutation test loop
                print("loop :",y,"gen : ",i)
                d=copy.deepcopy(self.WBGenRef)

                w=torch.stack((d["W"]*100))
                b=torch.stack((d["b"]*100))
                mw=[torch.randn(d["W"].shape)*0.2 for _ in range(100)]
                mb=[torch.randn(d["b"].shape)*0.2 for _ in range(100)]
                w+=torch.stack(mw)
                b+=torch.stack(mb)


                self.runner.reset()
                for j in range(len(d["W"])): #create mutation
                    d["W"][j]+=torch.randn(d["W"][j].shape) * 0.2 #gaussion mutation
                for h in range(len(d["b"])):
                    d["W"][h]+=torch.randn(d["b"][h].shape) * 0.2
                self.test(d)

            print("generation :",i,"best score :",self.BestScoreInGen)
            self.WBGenRef=self.WBBestInGen
            self.runner.reset()
            self.BestScoreInGen=0
            self.WBBestInGen={}
            self.test_pygame(self.WBGenRef)

    def test(self,dico):
        time=FPSM*self.timeDuration 
      
        for _ in range(time):
            pos = self.objectif
            x,y,v,vec=self.runner.getall()
            input_data = torch.tensor([
                [x / TAILLEX], 
                [y / TAILLEY], 
                [v / PX], 
                [0.5*(1+vec[0])], 
                [0.5*(1+vec[1])],
                [pos[0] / TAILLEX],
                [pos[1] / TAILLEY]])
            
            output = self.feed_forward(input_data,dico)
            self.runner.move(TAILLEX * output[0][0], TAILLEY * output[1][0], output[2][0])
            self.ScoreUpdate(x,y,pos[0],pos[1])

        if self.ActTestScore>self.BestScoreInGen: #update best score and weights/biases if needed
            self.WBBestInGen=dico
            self.BestScoreInGen=self.ActTestScore
        self.ActTestScore=0
        return
    
    def test_pygame(self,dico): #visualisation
        time=FPSA*self.timeDuration
        atime=0
        clock=pygame.time.Clock()
        running=True
        n=0
        while running:
            screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    quit()
            if atime>=time:
                running = False
                self.runner.reset()
            else : 
                if n==0:
                    n=checkTime(n)
                    pos = self.objectif
                    x,y,v,vec=self.runner.getall()
                    input_data = torch.tensor([[x / TAILLEX], [y / TAILLEY], [v / PX], [vec[0]], [vec[1]],[pos[0] / TAILLEX],[pos[1] / TAILLEY]])
                    output = self.feed_forward(input_data,dico)
                    self.runner.move(TAILLEX * output[0][0], TAILLEY * output[1][0], output[2][0])
                else:
                    n=checkTime(n)

                self.runner.draw(screen)
                pygame.draw.circle(screen,(0,255,0),(pos[0],pos[1]),5)
                pygame.display.update()
                clock.tick(FPSA)
                atime+=1
        
        return
    
n=NeuralNetwork()
n.start()
quit()

#Upgrad score system