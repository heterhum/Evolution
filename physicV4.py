import numpy as np
class physic:
    def __init__(self):
        self.Acoef=0.8
        self.MaxSpeed=4

        self.ForceMulti=10
        self.MouvementMulti=0.99

    def updateVector(self,target,Va,xy,Vmouvement,fpsm=120):
        t1=Va*self.Acoef #assume m=1 then momentum is v; Force norme we oppose to Vmouvement
        dire=target-xy
        dire=self.normalise(dire) #to where

        Force=dire*t1*(1/fpsm)*self.ForceMulti
        new=Force+Vmouvement*self.MouvementMulti #magical shit
        v=np.linalg.norm(new)
        
        if v>self.MaxSpeed:
            new=new* self.MaxSpeed/v
        return v,new
    
    def move(self,mx,my,va,xy,Vmouvement,fpsm=120): 
        v,vec=self.updateVector(np.array([mx,my]),va,xy,Vmouvement,fpsm)
        xy=np.add(xy,vec) #why, how it's working, idk, it's not physic it's fisique
        return v,vec,xy
    
    def normalise(self,vect):
        norm = np.linalg.vector_norm(vect)
        if norm > 1e-8:
            return vect / norm
        return np.zeros_like(vect)
    
    def sigmoid(self,a):
      return 1 / (1 + np.exp(-a))