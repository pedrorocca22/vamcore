"""Reusable editorial composition. Visuals and metrics must come from the product.

Requires Pillow. Assets default to the sibling APTUM preset; custom brands can
provide style_file and fonts_dir. This module never writes into the skill folder.
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps


class Composer:
    def __init__(self, style_file=None, fonts_dir=None):
        preset=Path(__file__).resolve().parents[1]/"assets/aptum"
        self.style=json.loads(Path(style_file or preset/"style.json").read_text(encoding="utf-8"))
        self.fonts_dir=Path(fonts_dir or preset/"fonts")
        self.w,self.h=self.style["width"],self.style["height"]
        self.vw,self.tx=self.style["visual_width"],self.style["text_x"]
        self.sx,self.sy=self.w/1920,self.h/1080
        self.colors=self.style["palette"];self._fonts={}

    def font(self,size,role="body",weight=400):
        key=(size,role,weight)
        if key not in self._fonts:
            face=ImageFont.truetype(str(self.fonts_dir/self.style["fonts"][role]),round(size*self.sy))
            try:axes=face.get_variation_axes()
            except OSError:axes=[]
            if axes:
                values=[]
                for axis in axes:
                    name=axis["name"].decode()
                    value=72 if "Optical" in name else weight if "Weight" in name else axis["default"]
                    values.append(max(axis["minimum"],min(axis["maximum"],value)))
                face.set_variation_by_axes(values)
            self._fonts[key]=face
        return self._fonts[key]

    def fit(self,lines,size,role="body",weight=400,min_size=None,width=None):
        width=width or self.w-self.tx-self.style["right_margin"]
        min_size=min_size or size
        for candidate in range(size,min_size-1,-1):
            face=self.font(candidate,role,weight)
            if max((face.getlength(line) for line in lines),default=0)<=width:return face
        raise ValueError("Text does not fit the editorial column; shorten it or adjust the layout.")

    def text(self,draw,xy,value,face,color,anchor=None):
        box=draw.textbbox(xy,value,font=face,anchor=anchor)
        if box[0]<0 or box[1]<0 or box[2]>self.w or box[3]>self.h:
            raise ValueError(f"Text outside the canvas: {value!r}; bounds={box}")
        draw.text(xy,value,font=face,fill=color,anchor=anchor)

    def chapter(self,*,visual,index,count,label,title,body,product,metric=None,
                evidence_label="",beta=False,t=None):
        """Compose one frame; title <=2 lines, body <=3 lines, optional metric tuple.

        Pass the changing product visual on EVERY frame, synchronized to the
        chapter's actual data state. This function creates no demonstration data.
        """
        if not 1<=index<=count:raise ValueError("Chapter index must be 1..count")
        titles=title.splitlines()
        if len(titles)>2 or len(body)>3:raise ValueError("Use <=2 title lines and <=3 body lines")
        if isinstance(visual,Image.Image):source=visual.convert("RGB")
        else:source=Image.fromarray(visual).convert("RGB")
        source=ImageOps.contain(source,(self.vw,self.h),Image.Resampling.LANCZOS)
        frame=Image.new("RGB",(self.w,self.h),self.colors["background"])
        frame.paste(source,((self.vw-source.width)//2,(self.h-source.height)//2))
        d=ImageDraw.Draw(frame);d.line((self.vw,160*self.sy,self.vw,1000*self.sy),fill=self.colors["rule"])
        if evidence_label:
            face=self.fit([evidence_label],18,"mono",min_size=16,width=self.vw-152*self.sx)
            self.text(d,(76*self.sx,92*self.sy),evidence_label,face,self.colors["subtle"])
        panel=Image.new("RGBA",frame.size,(0,0,0,0));d=ImageDraw.Draw(panel)
        chapter=f"{index:02d}  ·  {label.upper()}"
        face=self.fit([chapter],self.style["font_sizes"]["label"],"mono",min_size=18)
        self.text(d,(self.tx,202*self.sy),chapter,face,self.colors["accent"])
        face=self.fit(titles,self.style["font_sizes"]["title"],"display",500,54)
        for i,line in enumerate(titles):self.text(d,(self.tx,(266+i*70)*self.sy),line,face,self.colors["ink"])
        face=self.fit(body,self.style["font_sizes"]["body"],min_size=24)
        for i,line in enumerate(body):self.text(d,(self.tx,(455+i*39)*self.sy),line,face,self.colors["muted"])
        if metric is not None:
            value,caption=map(str,metric)
            face=self.fit([value],self.style["font_sizes"]["metric"],"mono",min_size=72)
            self.text(d,(self.tx,663*self.sy),value,face,self.colors["accent"])
            face=self.fit([caption],self.style["font_sizes"]["caption"],min_size=18)
            self.text(d,(self.tx,788*self.sy),caption,face,self.colors["subtle"])
        available=self.w-self.tx-self.style["right_margin"]
        step=available/count;bar=step*.81
        for i in range(count):
            x=self.tx+i*step
            d.rounded_rectangle((x,936*self.sy,x+bar,941*self.sy),radius=2*self.sy,
                                fill=self.colors["accent"] if i<index else self.colors["rule"])
        face=self.fit([product],24,weight=600,min_size=20,width=available*.8)
        self.text(d,(self.tx,977*self.sy),product,face,self.colors["ink"])
        if beta:self.text(d,(self.w-self.style["right_margin"],983*self.sy),"BETA",self.font(15,"mono"),self.colors["subtle"],anchor="ra")
        if t is not None:
            opacity=max(0,min(1,t/.10,(1-t)/.10))
            panel.putalpha(panel.getchannel("A").point(lambda x:round(x*opacity)))
        return Image.alpha_composite(frame.convert("RGBA"),panel).convert("RGB")

    def opening(self,product,claim,kicker="SOFTWARE SHOWCASE",footer=""):
        """A complete opening frame; deliberately no entry fade or blank lead-in."""
        frame=Image.new("RGB",(self.w,self.h),self.colors["background"]);d=ImageDraw.Draw(frame)
        width=self.w-240*self.sx
        self.text(d,(self.w/2,280*self.sy),kicker,self.fit([kicker],18,"mono",min_size=16,width=width),self.colors["accent"],anchor="mm")
        self.text(d,(self.w/2,470*self.sy),product,self.fit([product],166,weight=600,min_size=90,width=width),self.colors["ink"],anchor="mm")
        self.text(d,(self.w/2,635*self.sy),claim,self.fit([claim],46,"display",min_size=34,width=width),self.colors["muted"],anchor="mm")
        if footer:self.text(d,(self.w/2,946*self.sy),footer,self.fit([footer],20,min_size=18,width=width),self.colors["subtle"],anchor="mm")
        return frame
