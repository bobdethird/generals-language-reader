"""Render a real policy snapshot and the reader's unedited, verified outputs."""
import argparse
import json
import math
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def render(result,observation,output):
    width,height=1540,1010
    im=Image.new("RGB",(width,height),(246,247,249));d=ImageDraw.Draw(im)
    font=ROOT/"vendor/generals-bots/generals/assets/fonts/Quicksand-SemiBold.ttf"
    fonts={n:ImageFont.truetype(str(font),n) for n in (12,15,17,20,23,28,36)}
    ink=(26,44,58);muted=(89,110,126);green=(0,123,102);red=(181,64,69)
    def text(x,y,s,size=20,fill=ink,**kwargs):d.text((x,y),str(s),font=fonts[size],fill=fill,**kwargs)
    def paragraph(x,y,s,chars=54,size=20,fill=ink):
        for line in textwrap.wrap(s,width=chars):
            text(x,y,line,size,fill);y+=size+8
        return y
    text(38,24,"Reading checkpoint 3,800",36)
    text(40,78,"Actual player observation · move first, explanation afterward",20,muted)
    x0,y0,cell=57,146,29
    for i in range(24):
        text(x0+(i+.5)*cell,y0-18,i+1,12,muted,anchor="mm")
        text(x0-20,y0+(i+.5)*cell,i+1,12,muted,anchor="mm")
    for r in range(24):
        for c in range(24):
            hidden=observation[12,r,c]>0 or observation[13,r,c]>0
            friendly=observation[10,r,c]>0;enemy=observation[11,r,c]>0
            fill=(73,95,120) if hidden else (73,137,209) if friendly else (208,84,86) if enemy else (222,228,234)
            if observation[8,r,c]>0 and not hidden:fill=(116,124,134)
            left,top=x0+c*cell,y0+r*cell
            d.rectangle((left,top,left+cell-1,top+cell-1),fill=fill)
            if observation[6,r,c]>0 and not hidden:
                d.rectangle((left+3,top+3,left+cell-4,top+cell-4),outline=(255,223,116),width=2)
            if observation[7,r,c]>0 and not hidden:
                d.rectangle((left+3,top+3,left+cell-4,top+cell-4),outline=(235,235,235),width=1)
            army=int(observation[0,r,c])
            if army and not hidden:text(left+cell/2,top+cell/2,army,12 if army>=100 else 15,(255,255,255) if friendly or enemy else ink,anchor="mm")
    action=result["action"]
    if not action["pass"]:
        dr,dc={"north":(-1,0),"south":(1,0),"west":(0,-1),"east":(0,1)}[action["direction"]]
        r,c=action["row"],action["col"]
        a=(x0+(c+.5)*cell,y0+(r+.5)*cell);b=(a[0]+dc*cell,a[1]+dr*cell)
        d.line((*a,*b),fill=(255,232,107),width=5)
        angle=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([b,(b[0]-10*math.cos(angle-.6),b[1]-10*math.sin(angle-.6)),
            (b[0]-10*math.cos(angle+.6),b[1]-10*math.sin(angle+.6))],fill=(255,232,107))
    text(57,866,"Blue: own troops   Red: visible enemy   Dark: fog",17,muted)
    text(57,895,"Yellow outline: general   Arrow: actual selected move",17,muted)
    x,y=800,145
    text(x,y,"WHAT THE READER SAYS",17,muted);y+=35
    y=paragraph(x,y,result["raw_what"],51,23)+14
    text(x,y,"Verified" if result["what_verified"] else "Unverified — reader error retained",17,green if result["what_verified"] else red);y+=62
    text(x,y,"WHY — MEASURED INFLUENCE",17,muted);y+=35
    y=paragraph(x,y,result["raw_why"],51,23)+14
    text(x,y,"Verified against policy tests" if result["why_verified"] else "Unverified — reader error retained",17,green if result["why_verified"] else red);y+=65
    effect=result.get("reported_influence");audit=result["probe_measurements"]
    if effect is not None and effect["probe"]>=0:
        k=effect["probe"];delta=audit["margin_changes"][k]
        text(x,y,"PREFERENCE CHANGE IN THE TWO TESTS",15,muted);y+=34
        text(x,y,f"Milder: {delta[0]:+.3f}    Stronger: {delta[1]:+.3f}",23);y+=33
        y=paragraph(x,y,"Change in the chosen move's logit margin over its strongest alternative.",61,17,muted)
    elif not any(audit["consistent"]):
        y=paragraph(x,y,"These probes did not establish a stable influence. The interpreter should not invent a reason.",57,20,muted)
    else:
        y=paragraph(x,y,"Measured influences exist, but this generated explanation was not verified.",57,20,muted)
    paragraph(800,876,"This tests local input sensitivity. It does not prove a long-term plan such as attacking or defending.",65,17,muted)
    text(40,966,f"Frozen EMA 3800 · {result['player_sha256'][:16]} · test snapshot {result['snapshot_index']}",15,muted)
    output.parent.mkdir(parents=True,exist_ok=True);im.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--result",type=Path,required=True)
    p.add_argument("--samples",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=json.loads(a.result.read_text())
    with np.load(a.samples,allow_pickle=False) as z:o=z["observations"][result["snapshot_index"]]
    render(result,o,a.output)


if __name__=="__main__":main()
