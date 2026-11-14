"""Approved APTUM reveal. Import LogoAnimation; rgba(t) returns an alpha frame.

Use t=0..4 seconds for a compact closing. Does not encode files, connect services
or write to the skill directory. Requires Pillow and NumPy. Artwork is unchanged.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def ease(value):
    t=np.clip(value,0,1)
    return t*t*t*(t*(t*6-15)+10)


class LogoAnimation:
    def __init__(self,source=None,size=(1920,1080),width=None):
        source=Image.open(source or Path(__file__).with_name("logo-claim.png")).convert("RGBA")
        if source.size!=(716,132):raise ValueError("This reveal is mapped to the supplied 716x132 APTUM artwork")
        self.size=tuple(size)
        width=width or round(size[0]*1320/1920)
        self.scale=width/source.width
        self.logo=source.resize((width,round(source.height*self.scale)),Image.Resampling.LANCZOS)
        self.origin=((size[0]-self.logo.width)//2,(size[1]-self.logo.height)//2)
        self.parts={}
        boxes={
            "a":(0,0,96,85),"p":(96,0,176,85),"t":(176,0,255,85),
            "u":(255,0,337,85),"m":(337,0,440,85),"research":(0,85,440,132),
            "divider":(440,0,490,132),"claim_top":(490,0,716,66),"claim_bottom":(490,66,716,132),
        }
        for name,box in boxes.items():
            scaled=tuple(round(v*self.scale) for v in box)
            self.parts[name]=(self.logo.crop(scaled),scaled[:2])
        part,_=self.parts["a"]
        yy,xx=np.mgrid[:part.height,:part.width]
        points=np.stack([xx/self.scale,yy/self.scale],axis=-1)
        left=np.array([1.,78.]);apex=np.array([43.,4.]);right=np.array([87.,78.])
        def project(start,end):
            vector=end-start
            fraction=np.clip(np.sum((points-start)*vector,axis=-1)/np.dot(vector,vector),0,1)
            distance=np.sum((points-(start+fraction[...,None]*vector))**2,axis=-1)
            return fraction,distance
        f1,d1=project(left,apex);f2,d2=project(apex,right)
        self.stroke_progress=np.where(d1<=d2,f1*.5,.5+f2*.5)

    def paste(self,canvas,name,progress,drift=0,mask=None):
        if progress<=0:return
        part,offset=self.parts[name];image=part.copy()
        alpha=np.asarray(image.getchannel("A"),dtype=np.float32)*progress
        if mask is not None:alpha*=mask
        image.putalpha(Image.fromarray(np.clip(alpha,0,255).astype(np.uint8)))
        shift=round(drift*(1-progress))
        canvas.alpha_composite(image,(self.origin[0]+offset[0],self.origin[1]+offset[1]+shift))

    def rgba(self,seconds):
        canvas=Image.new("RGBA",self.size,(0,0,0,0))
        if seconds>=2.75:
            canvas.alpha_composite(self.logo,self.origin)
            return canvas
        draw_progress=float(ease((seconds-.15)/1.10))
        stroke_mask=ease((draw_progress*1.065-self.stroke_progress)/.065)
        self.paste(canvas,"a",1 if seconds>.15 else 0,mask=stroke_mask)
        for name,start in [("p",.64),("t",.75),("u",.86),("m",.97)]:
            self.paste(canvas,name,float(ease((seconds-start)/.70)),drift=13)
        self.paste(canvas,"research",float(ease((seconds-1.10)/.80)),drift=8)
        divider_progress=float(ease((seconds-1.35)/.65))
        divider,_=self.parts["divider"]
        y=np.arange(divider.height)
        distance=np.abs((y+.5)/divider.height-.5)*2
        divider_mask=ease((divider_progress*1.08-distance)/.08)[:,None]
        self.paste(canvas,"divider",1,mask=divider_mask)
        self.paste(canvas,"claim_top",float(ease((seconds-1.65)/.80)),drift=7)
        self.paste(canvas,"claim_bottom",float(ease((seconds-1.85)/.80)),drift=7)
        return canvas
