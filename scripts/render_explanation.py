"""Show plain-English summaries with the unedited model output underneath."""
import argparse
import json
import math
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def plain_explanation(result):
    """Simplify supported claims without adding a motive or future outcome."""
    action=result["action"]
    direction=None if action["pass"] else {"north":"up","south":"down","west":"left","east":"right"}[action["direction"]]
    move=("Wait. Don't move any troops this turn." if action["pass"] else
          f"Move {'half the troops' if action['half'] else 'all but one troop'} {direction} "
          f"from row {action['row']+1}, column {action['col']+1}.")
    effect=result.get("reported_influence")
    if not result["why_verified"] or effect is None:
        return move,"The reader's explanation didn't pass our checks. We can't rely on it here.",None
    k=effect["probe"]
    if k<0:
        return move,"We couldn't tell why it chose this move from the changes we tested.",None
    if k<8:
        place=("the starting tile","the destination tile","the closest visible enemy army","our general's tile")[k//2]
        verb="added troops to" if k%2 else "reduced troops on"
        if k//2==2:verb="increased" if k%2 else "reduced"
        change=f"{verb} {place}"
    else:
        change=f"hid recent changes in {'our' if k==8 else 'enemy'} troop numbers"
    if effect["switches"]:
        reason=f"When we {change}, the AI chose a different move."
    else:
        reason=f"When we {change}, the AI favored this move {'more' if effect['effect']=='strengthens' else 'less'}. "
        reason+=("It still chose to wait." if action["pass"] else f"It still chose to move {direction}.")
    detail=result["intervention_details"][k]
    if detail["kind"]=="troop_count":
        tested=f"We changed that tile from {detail['original']:g} to {detail['changed'][1]:g} troops and asked the AI to choose again."
    else:
        tested="We hid the recent troop-change information, kept the current board the same, and asked the AI to choose again."
    return move,reason,tested


def render(result,observation,output):
    width,height=1540,1210
    im=Image.new("RGB",(width,height),(246,247,249));d=ImageDraw.Draw(im)
    font=ROOT/"vendor/generals-bots/generals/assets/fonts/Quicksand-SemiBold.ttf"
    fonts={n:ImageFont.truetype(str(font),n) for n in (12,15,17,20,23,28,36)}
    ink=(26,44,58);muted=(89,110,126);green=(0,123,102);red=(181,64,69)
    def text(x,y,s,size=20,fill=ink,**kwargs):d.text((x,y),str(s),font=fonts[size],fill=fill,**kwargs)
    def paragraph(x,y,s,chars=54,size=20,fill=ink):
        for line in textwrap.wrap(s,width=chars):
            text(x,y,line,size,fill);y+=size+8
        return y
    text(38,24,"Understanding this move",36)
    text(40,78,"AI trained for 3,800 updates · plain-English summary",20,muted)
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
    move,reason,tested=plain_explanation(result)
    x,y=800,145
    text(x,y,"THE MOVE",17,muted);y+=35
    y=paragraph(x,y,move,43,28)+18
    text(x,y,"Taken from the move the AI actually chose",17,muted);y+=64
    text(x,y,"WHAT AFFECTED ITS CHOICE",17,muted);y+=35
    y=paragraph(x,y,reason,43,28)+18
    text(x,y,"Checked by changing the input and trying again" if result["why_verified"] else "Explanation failed the check",17,green if result["why_verified"] else red);y+=62
    if tested:
        text(x,y,"HOW WE CHECKED",17,muted);y+=35
        y=paragraph(x,y,tested,55,20,muted)
    paragraph(800,870,"This shows what can affect its choice. It doesn't tell us its long-term plan.",65,17,muted)
    d.line((40,942,1500,942),fill=(214,221,228),width=1)
    text(40,961,"ORIGINAL MODEL OUTPUT — kept unchanged",15,muted)
    raw_y=paragraph(40,993,"Move reader: "+result["raw_what"],158,15,muted)+8
    if not result["what_verified"]:
        raw_y=paragraph(40,raw_y,"The move reader's description contains an error; it is not verified.",158,15,red)+8
    paragraph(40,raw_y,"Explanation reader: "+result["raw_why"],158,15,muted)
    text(40,1170,f"Snapshot {result['snapshot_index']} · plain wording added for display · original checkpoint {result['player_sha256'][:16]}",15,muted)
    output.parent.mkdir(parents=True,exist_ok=True);im.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--result",type=Path,required=True)
    p.add_argument("--samples",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=json.loads(a.result.read_text())
    with np.load(a.samples,allow_pickle=False) as z:o=z["observations"][result["snapshot_index"]]
    render(result,o,a.output)


if __name__=="__main__":main()
