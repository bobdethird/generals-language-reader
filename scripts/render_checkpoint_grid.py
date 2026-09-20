"""Render four recorded checkpoint games in a synchronized 2x2 spectator grid."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
QUADRANT = 1080
WIDTH = HEIGHT = 2 * QUADRANT
TICKS_PER_SECOND = 4  # 2x the standard game clock.
RED, BLUE = (213, 68, 77), (53, 111, 211)


def read_game(folder):
    with np.load(folder / "game.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    record = json.loads((folder / "game.json").read_text())
    assert len(data["time"]) == record["ticks"] + 1
    return data, record


class Board:
    def __init__(self, folder):
        self.folder = folder
        self.data, self.record = read_game(folder)
        self.size = self.record["grid_size"]
        self.cell = min((QUADRANT - 20) // self.size, (QUADRANT - 88) // self.size)
        self.x = (QUADRANT - self.cell * self.size) // 2
        self.y = 80
        self.labels = self.record.get("player_labels", [self.record["checkpoint_label"]] * 2)
        self.army = (self.data["armies"][:, None] * self.data["ownership"]).sum(axis=(2, 3))
        self.land = self.data["ownership"].sum(axis=(2, 3))
        font = ROOT / "vendor/generals-bots/generals/assets/fonts/Quicksand-SemiBold.ttf"
        self.fonts = {size: ImageFont.truetype(str(font), size) for size in (16, 18, 20)}
        self.finished_frame = None

    def frame(self, tick):
        index = min(tick, self.record["ticks"])
        finished = tick >= self.record["ticks"]
        if finished and self.finished_frame is not None:
            return self.finished_frame
        im = Image.new("RGB", (QUADRANT, QUADRANT), (20, 26, 35))
        d = ImageDraw.Draw(im)
        own = self.data["ownership"][index]
        armies = self.data["armies"][index]
        mountains = self.data["mountains"][index]
        generals = self.data["generals"][index]
        castles = self.data["castles"][index]

        def text(x, y, value, size=18, color=(239, 244, 249), **kwargs):
            d.text((x, y), str(value), font=self.fonts[size], fill=color, **kwargs)

        for r in range(self.size):
            for c in range(self.size):
                x, y = self.x + c * self.cell, self.y + r * self.cell
                owner = 0 if own[0, r, c] else 1 if own[1, r, c] else -1
                fill = (RED, BLUE)[owner] if owner >= 0 else (222, 229, 237)
                if mountains[r, c]:
                    fill = (75, 91, 106)
                elif castles[r, c] or generals[r, c]:
                    fill = tuple(int(v * .78) for v in fill)
                d.rectangle((x, y, x + self.cell - 1, y + self.cell - 1), fill=fill, outline=(172, 184, 197))
                center = x + self.cell / 2
                if mountains[r, c]:
                    d.polygon([(x+8,y+self.cell-9),(center,y+10),(x+self.cell-8,y+self.cell-9)],fill=(158,175,189))
                elif generals[r, c]:
                    d.polygon([(x+8,y+9),(x+13,y+14),(center,y+6),(x+self.cell-13,y+14),
                               (x+self.cell-8,y+9),(x+self.cell-11,y+20),(x+11,y+20)],fill=(255,224,131))
                elif castles[r, c]:
                    d.rectangle((center-10,y+10,center+10,y+21),fill=(231,237,245))
                    for dx in (-10,-2,6):
                        d.rectangle((center+dx,y+6,center+dx+4,y+11),fill=(231,237,245))
                if armies[r,c] > 0 and not mountains[r,c]:
                    cy = y + (self.cell-10 if castles[r,c] or generals[r,c] else self.cell/2)
                    text(center,cy,int(armies[r,c]),16 if armies[r,c]>=100 else 20,
                         (255,255,255) if owner>=0 else (27,44,59),anchor="mm")

        if not finished:
            for action in self.data["actions"][index]:
                passed,r,c,direction,half=map(int,action)
                if passed:
                    continue
                dr,dc=((-1,0),(1,0),(0,-1),(0,1))[direction]
                x1,y1=self.x+(c+.5)*self.cell,self.y+(r+.5)*self.cell
                x2,y2=x1+dc*self.cell*.7,y1+dr*self.cell*.7
                d.line((x1,y1,x2,y2),fill=(20,30,45),width=6)
                d.line((x1,y1,x2,y2),fill=(255,233,154),width=3)
                angle=math.atan2(y2-y1,x2-x1)
                d.polygon([(x2,y2),(x2-9*math.cos(angle-.5),y2-9*math.sin(angle-.5)),
                           (x2-9*math.cos(angle+.5),y2-9*math.sin(angle+.5))],fill=(255,233,154))

        # Compact top-right overlay lives above the board, so no tiles are hidden.
        left,right=QUADRANT-398,QUADRANT-14
        d.rounded_rectangle((left,4,right,73),radius=8,fill=(32,42,55))
        result=""
        if finished:
            result=" · "+("RED WINS" if self.record["winner"]==0 else "BLUE WINS" if self.record["winner"]==1 else "DRAW")
        text(right-12,8,f"CHECKPOINTS {self.labels[0]} vs {self.labels[1]}{result}",18,anchor="ra")
        text(left+12,32,f"{self.labels[0]}: {int(self.army[index,0])} army / {int(self.land[index,0])} land",16,(255,146,153))
        text(left+12,52,f"{self.labels[1]}: {int(self.army[index,1])} army / {int(self.land[index,1])} land",16,(141,184,255))
        text(right-12,41,f"T {index/2:g} · 2×",16,(187,201,216),anchor="ra")
        if finished:
            self.finished_frame=im
        return im


def render(args):
    boards=[Board(folder.resolve()) for folder in args.games]
    # The same map and starting sides isolate the opponent change for this recording.
    for board in boards[1:]:
        for field in ("armies","ownership","generals","castles","mountains"):
            if not np.array_equal(board.data[field][0],boards[0].data[field][0]):
                raise ValueError("Games must start on identical maps and player positions")
        if board.record["checkpoint_sha256"] != boards[0].record["checkpoint_sha256"]:
            raise ValueError("The red model must be identical in all quadrants")
    final_tick=max(board.record["ticks"] for board in boards)
    args.output.mkdir(parents=True,exist_ok=True)
    movie=args.output/"checkpoint-3000-four-games.mp4"
    temporary=movie.with_name(movie.stem+".tmp.mp4")
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    command=[ffmpeg,"-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{WIDTH}x{HEIGHT}","-framerate",str(TICKS_PER_SECOND),"-i","pipe:0","-an",
             "-vf","fps=30","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p",
             "-movflags","+faststart",str(temporary)]

    def composite(tick):
        im=Image.new("RGB",(WIDTH,HEIGHT),(20,26,35))
        for i,board in enumerate(boards):
            im.paste(board.frame(tick),((i%2)*QUADRANT,(i//2)*QUADRANT))
        d=ImageDraw.Draw(im)
        # Explain checkpoint numbers once, in the unused top-left margin.
        # Keep the legend above the map and the individual match overlays small.
        d.text((46,8),"Same AI at different stages of training",font=boards[0].fonts[20],fill=(239,244,249))
        d.text((46,33),"Numbers = completed training iterations, not Elo ratings.",
               font=boards[0].fonts[16],fill=(187,201,216))
        d.text((46,54),"Each iteration learns from a batch of self-play experience.",
               font=boards[0].fonts[16],fill=(187,201,216))
        d.line((QUADRANT,0,QUADRANT,HEIGHT),fill=(87,103,121),width=2)
        d.line((0,QUADRANT,WIDTH,QUADRANT),fill=(87,103,121),width=2)
        return im

    with (args.output/"encoding.log").open("w") as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            first=composite(0)
            first.save(args.output/"first-frame.png")
            for _ in range(8):
                proc.stdin.write(first.tobytes())
            for tick in range(1,final_tick+1):
                frame=composite(tick)
                proc.stdin.write(frame.tobytes())
                if tick in (min(216,final_tick),final_tick//2):
                    frame.save(args.output/f"preview-tick-{tick}.png")
                if tick%200==0:
                    print(f"Rendered synchronized tick {tick}/{final_tick}",flush=True)
            frame.save(args.output/"final-frame.png")
            for _ in range(16):
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            if proc.wait()!=0:
                raise RuntimeError("Video encoding failed; see encoding.log")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    temporary.replace(movie)
    manifest={"video":str(movie.resolve()),"layout":"2x2", "width":WIDTH,"height":HEIGHT,"fps":30,
              "duration_seconds":(final_tick+24)/TICKS_PER_SECOND,"playback_speed":2,
              "finished_games":"Hold final board while the other matches finish",
              "checkpoint_numbers":"Training iterations, not Elo ratings",
              "on_screen_legend":"Numbers = completed training iterations, not Elo ratings. Each iteration learns from a batch of self-play experience.",
              "sha256":hashlib.sha256(movie.read_bytes()).hexdigest(),
              "games":[{"position":position,"folder":str(board.folder),"labels":board.labels,
                        "ticks":board.record["ticks"],"winner":board.record["winner"]}
                       for position,board in zip(("top-left","top-right","bottom-left","bottom-right"),boards)]}
    (args.output/"grid.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games",type=Path,nargs=4,required=True,help="Top-left, top-right, bottom-left, bottom-right")
    parser.add_argument("--output",type=Path,required=True)
    render(parser.parse_args())
