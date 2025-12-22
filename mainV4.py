import pygame
import numpy as np
import copy
import physicV4 as physic
TAILLEX=1000
TAILLEY=800
MAXDISTANCE=np.sqrt(TAILLEX**2+TAILLEY**2)
screen = pygame.display.set_mode((TAILLEX,TAILLEY))

FPSM=30 #number of frames for movement calculation
FPSA=60 #number of frames for animation
pygame.init()
PX=100 #max speed, pixels per second

class Tagger:
    def __init__(self,color,radius,startpos):
        self.v=0
        self.xy=startpos
        #self.physic=physic_instance
        self.Vmouvement=np.array([0,0])
        self.color=color
        self.radius=radius
        self.score=0
    
    def move(self,mx,my,va): 
        self.v,self.Vmouvement,self.xy=physic.move(mx,my,va,self.xy,self.Vmouvement)
        return

    def scoreUpdate(self,posx,posy):
        c=np.linalg.norm(self.xy - np.array([posx,posy]))
        ct=1-(c/MAXDISTANCE)
        self.score+=physic.sigmoid(ct)
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.xy[0].item()),int(self.xy[1].item())),self.radius)
        return
    
    def reset(self):
        self.xy=np.array([100,100])
        self.v=0
        self.Vmouvement=np.array([0,0])
        self.score=0
        return
    
    def getall(self):
        Vdir=physic.normalise(self.Vmouvement)
        return (self.xy[0].item(),self.xy[1].item(),self.v,Vdir[0].item(),Vdir[1].item())
    

class Runner:
    def __init__(self,color,radius,startpos):
        self.v=0
        self.xy=startpos
        #self.physic=physic_instance
        self.Vmouvement=np.array([0,0])
        self.color=color
        self.radius=radius
        self.score=0
    
    def move(self,mx,my,va): 
        self.v,self.Vmouvement,self.xy=physic.move(mx,my,va,self.xy,self.Vmouvement)
        return

    def scoreUpdate(self,posx,posy):
        c=np.linalg.norm(self.xy - np.array([posx,posy]))
        ct=1-(c/MAXDISTANCE)
        self.score+=physic.sigmoid(ct)
        return

    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(int(self.xy[0].item()),int(self.xy[1].item())),self.radius)
        return
    
    def reset(self):
        self.xy=np.array([100,100])
        self.v=0
        self.Vmouvement=np.array([0,0])
        self.score=0
        return
    
    def getall(self):
        Vdir=physic.normalise(self.Vmouvement)
        return (self.xy[0].item(),self.xy[1].item(),self.v,Vdir[0].item(),Vdir[1].item())


class NeuralNetwork:
    def __init__(self):
        self.tagger=Tagger((255,255,255),5,np.array([100,100]))
        self.runner=Runner((0,0,255),5,np.array([800,400]))
        self.ANNDisp = [10, 14, 16, 8, 3] #10 input,3 output, 2 hidden layers

        self.WBGenRef= {"runner":{'W':[np.random.randn(self.ANNDisp[i+1], self.ANNDisp[i]) for i in range(len(self.ANNDisp)-1)],
                                  'b':[np.random.randn(self.ANNDisp[i], 1) for i in range(1,len(self.ANNDisp))]},
                        "tagger":{'W':[np.random.randn(self.ANNDisp[i+1], self.ANNDisp[i]) for i in range(len(self.ANNDisp)-1)],
                                  'b':[np.random.randn(self.ANNDisp[i], 1) for i in range(1,len(self.ANNDisp))]}} #generation reference ( proud of this fucking way to create weights and biases )
        
        self.Ntest=100
        self.Ngen=10 #100 mutation tested on 50 generations
        self.WBBestInGen={"runner":None,"tagger":None} #best weights and biases in current generation
        self.BestScoreInGen={"runner":0,"tagger":0}#best score in current generation

        self.ActTestScore=0 #current score in test
        self.timeDuration=10 #seconds
        self.objectif=np.array([100,400]) #target position
        
        self.Wscale=0.2
        self.Bscale=0.2
    
    def reset(self):
        self.tagger.reset()
        self.runner.reset()
        return

    def feed_forward(self,A0,dico): #can't explain shortly, it's basic ANN feed forward with sigmoid activation
        Alist=A0
        for i in range(len(self.ANNDisp)-1):
            Zi=dico['W'][i] @ Alist + dico['b'][i]
            Alist=physic.sigmoid(Zi)
        y_hat = Alist
        return y_hat

    def ScoreUpdate(self): #every frame score updpate, need to to do better system later
        Rx,Ry=self.runner.xy
        Tx,Ty=self.tagger.xy
        self.tagger.scoreUpdate(Rx,Ry)
        self.runner.scoreUpdate(Tx,Ty)
        return 

    def start(self):
        for i in range(self.Ngen): #generation loop
            for y in range(self.Ntest): #mutation test loop
                print("loop :",y,"gen : ",i)
                d=copy.deepcopy(self.WBGenRef)
                self.runner.reset()
                for j in range(len(d["runner"]["W"])): #create mutation
                    d["runner"]["W"][j]+=np.random.normal(0,self.Wscale,d["runner"]["W"][j].shape) #gaussion mutation
                for h in range(len(d["runner"]["b"])):
                    d["runner"]["b"][h]+=np.random.normal(0,self.Bscale,d["runner"]["b"][h].shape)

                for j in range(len(d["tagger"]["W"])): #create mutation
                    d["tagger"]["W"][j]+=np.random.normal(0,self.Wscale,d["tagger"]["W"][j].shape) #gaussion mutation
                for h in range(len(d["tagger"]["b"])):
                    d["tagger"]["b"][h]+=np.random.normal(0,self.Bscale,d["tagger"]["b"][h].shape)

                self.test(d)
                self.UpdateBest(d)
                self.reset()

            print("generation :",i,"best score :",self.BestScoreInGen)
            self.WBGenRef=self.WBBestInGen
            self.BestScoreInGen={"runner":0,"tagger":0}
            self.WBBestInGen={"runner":None,"tagger":None}

            self.test_pygame(self.WBGenRef)

    def UpdateBest(self,dico):
        if self.tagger.score>self.BestScoreInGen["tagger"]: #update best score and weights/biases if needed
            self.WBBestInGen["tagger"]=dico["tagger"]
            self.BestScoreInGen["tagger"]=self.tagger.score
        if self.runner.score>self.BestScoreInGen["runner"]: #update best score and weights/biases if needed
            self.WBBestInGen["runner"]=dico["runner"]
            self.BestScoreInGen["runner"]=self.runner.score
        return

    def test(self,dico):
        time=FPSM*self.timeDuration 
      
        for _ in range(time):
            Rx,Ry,Rv,Rvecx,Rvecy=self.runner.getall()
            Tx,Ty,Tv,Tvecx,Tvecy=self.tagger.getall()
            input_data = np.array([
                [Rx / TAILLEX], 
                [Ry / TAILLEY], 
                [Rv / PX], 
                [0.5*(1+Rvecx)], 
                [0.5*(1+Rvecy)],

                [Tx / TAILLEX], 
                [Ty / TAILLEY], 
                [Tv / PX], 
                [0.5*(1+Tvecx)], 
                [0.5*(1+Tvecy)]])
            
            Routput = self.feed_forward(input_data,dico["runner"])
            Toutput = self.feed_forward(input_data,dico["tagger"])
            self.runner.move(TAILLEX * Routput[0][0], TAILLEY * Routput[1][0], Routput[2][0])
            self.tagger.move(TAILLEX * Toutput[0][0], TAILLEY * Toutput[1][0], Toutput[2][0])
            self.ScoreUpdate()
        return
    
    def test_pygame(self,dico): #visualisation
        time=FPSM*self.timeDuration

        for _ in range(time):
            screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    quit()
            Rx,Ry,Rv,Rvecx,Rvecy=self.runner.getall()
            Tx,Ty,Tv,Tvecx,Tvecy=self.tagger.getall()
            input_data = np.array([
                [Rx / TAILLEX], 
                [Ry / TAILLEY], 
                [Rv / PX], 
                [0.5*(1+Rvecx)], 
                [0.5*(1+Rvecy)],

                [Tx / TAILLEX], 
                [Ty / TAILLEY], 
                [Tv / PX], 
                [0.5*(1+Tvecx)], 
                [0.5*(1+Tvecy)]])
            
            Routput = self.feed_forward(input_data,dico["runner"])
            Toutput = self.feed_forward(input_data,dico["tagger"])
            self.runner.move(TAILLEX * Routput[0][0], TAILLEY * Routput[1][0], Routput[2][0])
            self.tagger.move(TAILLEX * Toutput[0][0], TAILLEY * Toutput[1][0], Toutput[2][0])

            self.runner.draw(screen)
            self.tagger.draw(screen)
            pygame.display.update()

        return
    
n=NeuralNetwork()
n.start()
quit()

#Upgrad score system