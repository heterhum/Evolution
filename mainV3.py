import pygame
import numpy as np
import copy

TAILLEX=1000
TAILLEY=800
MAXDISTANCE=np.sqrt(TAILLEX**2+TAILLEY**2)
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
    return np.sqrt(vect[0]**2+vect[1]**2)

def Vnormale(vect,norme):
    if norme!=0:
        return (vect[0]/norme,vect[1]/norme)
    return (0,0)

def normalise(vect):
    norm = np.linalg.vector_norm(vect)
    if norm > 1e-8:
        return vect / norm
    return np.zeros_like(vect)

class Runner:
    def __init__(self):
        self.v=0
        self.xy=np.array([100,100])
        self.Vmouvement=np.array([0,0])
        self.Acoef=0.8
        self.MaxSpeed=4

        self.ForceMulti=10
        self.MouvementMulti=0.99

        self.color=(255,255,255)
        self.radius=5

    def updateVector(self,target,Va):
        t1=Va*self.Acoef #assume m=1 then momentum is v; Force norme we oppose to Vmouvement
        dire=target-self.xy
        dire=normalise(dire) #to where

        Force=dire*t1*(1/FPSM)*self.ForceMulti
        new=Force+self.Vmouvement*self.MouvementMulti #magical shit
        v=np.linalg.norm(new)
        
        if v>self.MaxSpeed:
            new=new* self.MaxSpeed/v
        return v,new

    def reset(self):
        self.xy=np.array([100,100])
        self.v=0
        self.Vmouvement=np.array([0,0])
        return
    
    def move(self,mx,my,va): 
        v,vec=self.updateVector(np.array([mx,my]),va)
        self.v=v
        self.Vmouvement=vec
        self.xy=np.add(self.xy,self.Vmouvement) #why, how it's working, idk, it's not physic it's fisique
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.xy[0].item()),int(self.xy[1].item())),self.radius)
        #pygame.draw.line(screen,(255,0,0),(int(self.x),int(self.y)),(int(self.x+self.vector[0]),int(self.y+self.vector[1])),2)
        return

    def getall(self):
        go=normalise(self.Vmouvement)
        return (self.xy[0].item(),self.xy[1].item(),self.v,go[0].item(),go[1].item())

class NeuralNetwork:
    def __init__(self):
        self.runner=Runner()
        self.ANNDisp = [7, 8, 16, 8, 3] #7 input,3 output, 2 hidden layers
        self.WBGenRef={'W':[np.random.randn(self.ANNDisp[i+1], self.ANNDisp[i]) for i in range(len(self.ANNDisp)-1)],
                       'b':[np.random.randn(self.ANNDisp[i], 1) for i in range(1,len(self.ANNDisp))]} #generation reference ( proud of this fucking way to create weights and biases )
        
        self.Ntest=100
        self.Ngen=10 #100 mutation tested on 50 generations
        self.WBBestInGen={} #best weights and biases in current generation
        self.BestScoreInGen=0 #best score in current generation

        self.ActTestScore=0 #current score in test
        self.timeDuration=10 #seconds
        self.objectif=np.array([400,100]) #target position
        
        self.Wscale=0.2
        self.Bscale=0.2

    def sigmoid(self,a):
      return 1 / (1 + np.exp(-a))
    
    def feed_forward(self,A0,dico): #can't explain shortly, it's basic ANN feed forward with sigmoid activation
        Alist=A0
        for i in range(len(self.ANNDisp)-1):
            Zi=dico['W'][i] @ Alist + dico['b'][i]
            Alist=self.sigmoid(Zi)
        y_hat = Alist
        return y_hat

    def ScoreUpdate(self,x,y,posx,posy): #every frame score updpate, need to to do better system later
        c=np.sqrt((x - posx) ** 2 + (y - posy) ** 2)
        ct=1-(c/MAXDISTANCE)
        self.ActTestScore+=self.sigmoid(ct)
        return 

    def start(self):
        for i in range(self.Ngen): #generation loop
            for y in range(self.Ntest): #mutation test loop
                print("loop :",y,"gen : ",i)
                d=copy.deepcopy(self.WBGenRef)
                self.runner.reset()
                for j in range(len(d["W"])): #create mutation
                    d["W"][j]+=np.random.normal(0,self.Wscale,d["W"][j].shape) #gaussion mutation
                for h in range(len(d["b"])):
                    d["b"][h]+=np.random.normal(0,self.Bscale,d["b"][h].shape)
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
            x,y,v,vecx,vecy=self.runner.getall()
            input_data = np.array([
                [x / TAILLEX], 
                [y / TAILLEY], 
                [v / PX], 
                [0.5*(1+vecx)], 
                [0.5*(1+vecy)],
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
                    x,y,v,vecx,vecy=self.runner.getall()
                    input_data = np.array([[x / TAILLEX], [y / TAILLEY], [v / PX], [0.5*(1+vecx)], [0.5*(1+vecy)],[pos[0] / TAILLEX],[pos[1] / TAILLEY]])
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