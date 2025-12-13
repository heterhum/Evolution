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

#def normalise(vect): # dim 0 [x,y]
#    vect = vect/vect.sum(0).expand_as(vect) 
#    vect[torch.isnan(vect)]=0 
#    return vect

def normalise(vect):
    norm = torch.linalg.vector_norm(vect)
    if norm > 1e-8:
        return vect / norm
    return torch.zeros_like(vect)
class Runner:
    def __init__(self,startpos):
        self.StartPos=startpos
        self.ActPos=startpos
        self.v=0
        self.Acoef=0.01
        self.Fcoef=0.5
        self.vector=torch.tensor([0,0])

        self.color=(255,255,255)
        self.radius=5

    def updateVector(self,target: torch.tensor,Va: torch.tensor): #target : target position , Va : target speed (-1 to 1)
        
        target_v = Va * PX 
        if target_v < self.v: #acceleration or freinage
            v=self.v+self.Fcoef * (target_v - self.v)
            if v<=10e-4: #avoid big negative like -1e-200
                v=0
        else:
            v=self.v+self.Acoef * (target_v - self.v)

        vector = target-self.ActPos
        vector = normalise(vector)

        vector2=vector+self.vector
        vector2=normalise(vector2)

        return v,vector2 #v : new speed , vnormale2 : new direction vector

    def reset(self): #reset everything, use at the start of test
        self.ActPos=self.StartPos
        self.v=0
        self.vector=torch.tensor([0,0])
        return

    def inTrajectory(self,target,Second,First): #if runner skip the point he asked to reach he would brake
        v = torch.subtract(Second,First)
        if torch.linalg.vector_norm(v)==0:
            return torch.linalg.vector_norm(torch.subtract(target, First))
        else:
            w = target - First      
            f = target + (torch.dot(w, v) / torch.dot(v, v)) * v 
            return torch.linalg.vector_norm(torch.subtract(f, First)) #projection orthogonal point distance to the line segment
    
    def move(self,target,va):
        v,vec=self.updateVector(target,va)
        OldPos=self.ActPos
        NewPos=self.ActPos+(torch.multiply(vec,v/FPSM))  
        if self.inTrajectory(target,NewPos,OldPos) < self.radius:
            v,vec=self.updateVector(target,0)
            self.ActPos=self.ActPos+(torch.multiply(vec,v/FPSM))        
        else:
            self.v=v
            self.ActPos=NewPos
        self.vector=vec
        self.vector=vec
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.ActPos[0].item()),int(self.ActPos[1].item())),self.radius)
        #pygame.draw.line(screen,(255,0,0),(int(self.ActPos[0].item()),int(self.ActPos[1].item())),(int(self.x+self.vector[0]*self.v),int(self.y+self.vector[1]*self.v)),2)
        return

    def getall(self):
        return (self.ActPos[0].item(),self.ActPos[1].item(),self.v,self.vector)
    
    def getallANN(self):
        c=torch.tensor([
                [self.ActPos[0].item() / TAILLEX], 
                [self.ActPos[1].item() / TAILLEY], 
                [self.v / PX], 
                [0.5*(1+self.vector[0].item())], 
                [0.5*(1+self.vector[1].item())]
                ])
        return c

class Brunner():
    def __init__(self):
        pass

class NeuralNetwork:
    def __init__(self):
        
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

    def ScoreUpdate(self,pos,target): #every frame score updpate, need to to do better system later
        c=torch.norm(pos-target)
        ct=1-(c/MAXDISTANCE)
        return self.Sigmoid(ct)


    def start(self):
        startpos=torch.tensor([TAILLEX//2,TAILLEY//2])
        for i in range(self.Ngen): #generation loop
            print("gen : ",i)
            d=copy.deepcopy(self.WBGenRef)

            w=[[i.clone() for i in d["W"]] for _ in range(self.Ntest)]
            b=[[i.clone() for i in d["b"]] for _ in range(self.Ntest)]
            
            w= [torch.stack([d['W'][i].clone() for _ in range(self.Ntest)]) for i in range(len(d['W']))]
            b= [torch.stack([d['b'][i].clone() for _ in range(self.Ntest)]) for i in range(len(d['b']))]

            mw=[torch.randn(w[i].shape)*0.05 for i in range(len(d['W']))]
            mb=[torch.randn(b[i].shape)*0.05 for i in range(len(d['b']))]
            
            for iw in range(len(w)):
                w[iw]=torch.add(mw[iw],w[iw])
            for ib in range(len(b)):
                print(ib,type(b[ib]),type(mb[ib]))
                b[ib]=torch.add(b[ib],mb[ib])

            pos = self.objectif
            cl=[Runner(startpos) for _ in range(self.Ntest)]
            score=[0 for _ in range(self.Ntest)]

            for j in range(FPSM*self.timeDuration): #movement loop
                print(j)
                data=[torch.vstack([i.getallANN(),torch.tensor([[pos[0] / TAILLEX],[pos[1] / TAILLEY]])]) for i in cl]
                data=torch.stack(data)
                for i in range(len(d['W'])):
                    data=w[i] @ data + b[i]
                    data=self.Sigmoid(data)

                for i in range(len(cl)):
                    targ=torch.tensor([data[i][0]*TAILLEX,data[i][1]*TAILLEY])
                    cl[i].move(targ,data[i][2]) #<-
                    score[i]+=self.ScoreUpdate(pos,targ)

            max=float('-inf')
            at=0
            for i in range(len(score)):
                if score[i]>max:
                    max=score[i]
                    at=i
            self.BestScoreInGen=max
            c=[w[i][at] for i in range(len(d['W']))]
            cc=[b[i][at] for i in range(len(d['b']))]
            self.WBGenRef={'W':c,'b':cc}

            print("generation :",i,"best score :",self.BestScoreInGen)
            self.BestScoreInGen=0
            self.test_pygame(self.WBGenRef)

    def feed_forward(self,A0,dico): #can't explain shortly, it's basic ANN feed forward with sigmoid activation
        Alist=A0
        for i in range(len(self.ANNDisp)-1):
            Zi=dico['W'][i] @ Alist + dico['b'][i]
            Alist=self.sigmoid(Zi)
        y_hat = Alist
        return y_hat

    def test_pygame(self,dico): #visualisation
        runner=Runner(torch.tensor([TAILLEX//2,TAILLEY//2]))
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
                runner.reset()
            else : 
                if n==0:
                    n=checkTime(n)
                    pos = self.objectif
                    x,y,v,vec=runner.getall()
                    input_data = torch.tensor([[x / TAILLEX], [y / TAILLEY], [v / PX], [vec[0]], [vec[1]],[pos[0] / TAILLEX],[pos[1] / TAILLEY]])
                    output = self.feed_forward(input_data,dico)
                    targ=torch.tensor([output[0][0]*TAILLEX,output[1][0]*TAILLEY])
                    runner.move(targ, output[2][0])
                else:
                    n=checkTime(n)

                runner.draw(screen)
                pygame.draw.circle(screen,(0,255,0),(pos[0],pos[1]),5)
                pygame.display.update()
                clock.tick(FPSA)
                atime+=1
        
        return
    
n=NeuralNetwork()
n.start()
quit()

#Upgrad score system