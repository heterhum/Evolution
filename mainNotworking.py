import pygame
import numpy as np
import copy

TAILLEX=1000
TAILLEY=800
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
        self.StartPos=(TAILLEX//2,TAILLEY//2)
        self.x = self.StartPos[0]
        self.y = self.StartPos[1]
        self.v=0
        self.Acoef=0.01
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
        if target_v < self.v:
            v=self.v+self.Fcoef * (target_v - self.v)
            if round(v,6)==0:
                v=0
        else:
            v=self.v+self.Acoef * (target_v - self.v)

        nvector1 = norme(self.vector)
        vnormale1 = Vnormale(self.vector,nvector1)

        vector2=np.add(vnormale1,vnormale)
        nvector2 = norme(vector2)
        vnormale2 = Vnormale(vector2,nvector2)

        return v,vnormale2

    def reset(self):
        self.x = self.StartPos[0]
        self.y = self.StartPos[1]
        self.v=0
        self.vector=np.array([0,0])
        return

    def inTrajectory(self,Mx,My,Second,First):
        cher=np.array([Mx,My])
        v = np.subtract(Second,First)
        if norme(v)==0:
            return norme(np.subtract(cher, First))
        else:
            w = cher - First      
            f = cher + (np.dot(w, v) / np.dot(v, v)) * v 
            return norme(np.subtract(f, First))
    
    def move(self,mx,my,va):
        v,vec=self.updateVector(mx,my,va)
        if self.inTrajectory(mx,my,np.array([self.x+v*vec[0],self.y+v*vec[1]]),np.array([self.x,self.y])) < self.radius:
            v,vec=self.updateVector(mx,my,0)
            pass
        
        self.vector=vec
        self.v=v
        self.x=self.x+self.vector[0]*v*1/FPSM
        self.y=self.y+self.vector[1]*v*1/FPSM
        
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
        self.WBGenRef={'W':[np.random.randn(self.ANNDisp[i+1], self.ANNDisp[i]) for i in range(len(self.ANNDisp)-1)],
                       'b':[np.random.randn(self.ANNDisp[i], 1) for i in range(1,len(self.ANNDisp))]} #generation reference

        
        self.Ntest=100
        self.Ngen=50 #100 mutation tested on 50 generations
        self.WBBestInGen={} #best weights and biases in current generation
        self.BestScoreInGen=0 #best score in current generation

        self.ActTestScore=0 #current score in test
        self.timeDuration=10 #seconds
        self.objectif=np.array([100,100]) #target position
        

    def sigmoid(self,arr):
      return 1 / (1 + np.exp(-1 * arr))
    
    # 5. create feed forward process
    def feed_forward(self,A0,dico):
        Alist=A0
        for i in range(len(self.ANNDisp)-1):
            Zi=dico['W'][i] @ Alist + dico['b'][i]
            Alist=self.sigmoid(Zi)
        y_hat = Alist
        return y_hat

    def ScoreUpdate(self,x,y,posx,posy):
        c=np.sqrt((x - posx) ** 2 + (y - posy) ** 2)
        ct=1-(c/MAXDISTANCE)
        self.ActTestScore+=self.sigmoid(ct)
        return 


    def start(self):
        for i in range(self.Ngen):
            for y in range(self.Ntest):
                print("loop :",y,"gen : ",i)
                d=copy.deepcopy(self.WBGenRef)
                self.runner.reset()
                for j in range(len(d["W"])):
                    d["W"][j]+=np.random.normal(0,0.2,d["W"][j].shape)
                    #d["W"][i]+=-1/2+np.random.rand(d["W"][i].shape[0],d["W"][i].shape[1])
                for h in range(len(d["b"])):
                    d["W"][h]+=np.random.normal(0,0.2,d["b"][h].shape)
                    #d["b"][i]+=-1/2+np.random.rand(d["b"][i].shape[0],d["b"][i].shape[1])


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
            input_data = np.array([
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

        if self.ActTestScore>self.BestScoreInGen:
            self.WBBestInGen=dico
            self.BestScoreInGen=self.ActTestScore
        self.ActTestScore=0
        return
    
    def test_pygame(self,dico):
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
                    input_data = np.array([[x / TAILLEX], [y / TAILLEY], [v / PX], [vec[0]], [vec[1]],[pos[0] / TAILLEX],[pos[1] / TAILLEY]])
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