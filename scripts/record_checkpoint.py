"""Simulate frozen-policy self-play or a checkpoint matchup and record a spectator MP4."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(ROOT / ".cache/human-play"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def simulate(args):
    from reader.runtime import import_upstream, install_simulator_compat
    import_upstream()
    install_simulator_compat()
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from config import Config
    from generals.core.action import compute_valid_move_mask
    from generals.core.env import GeneralsEnv
    from generals.core.game import get_observation
    from networks import build_network, get_network_bundle, obs_to_array

    cfg = Config.from_yaml(args.config)
    if not cfg.min_grid_size <= args.grid_size <= cfg.max_grid_size:
        raise ValueError("Map size must be within the trained range")
    stage = cfg.curriculum_stages[-1] if cfg.curriculum_stages else cfg
    env = GeneralsEnv(
        grid_dims=(args.grid_size, args.grid_size), pad_to=cfg.pad_to, pool_size=1,
        min_generals_distance=stage.min_generals_distance,
        max_generals_distance=stage.max_generals_distance,
        castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
        num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
        mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max),
        truncation=cfg.truncation, perfect_info=False)
    model = eqx.tree_deserialise_leaves(args.checkpoint, build_network(cfg, jax.random.PRNGKey(0)))
    opponent_path = getattr(args, "opponent", None) or args.checkpoint
    opponent = model if opponent_path == args.checkpoint else eqx.tree_deserialise_leaves(opponent_path, model)
    models = (model, opponent)
    for network in models:
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(eqx.filter(network, eqx.is_array))):
            raise ValueError("Nonfinite checkpoint")
    bundle = get_network_bundle(cfg.network)
    histories = [bundle["init_obs_state"](cfg.pad_to, cfg.pad_to) for _ in range(2)]
    pool, state = env.reset(jax.random.PRNGKey(args.seed))

    @eqx.filter_jit
    def choose(network, observation, history):
        augmented, history = bundle["augment_obs"](obs_to_array(observation), history)
        mask = compute_valid_move_mask(observation.armies, observation.owned_cells, observation.mountains)
        temporal = jnp.stack([history.opponent_army_history, history.opponent_land_history])
        return bundle["greedy_action"](network, augmented, mask, temporal), history

    step_fn = jax.jit(env.step)
    fields = ("armies", "ownership", "generals", "castles", "mountains", "time", "winner")
    frames = {field: [np.asarray(getattr(state, field)).copy()] for field in fields}
    actions_log = [np.array([[1, 0, 0, 0, 0]] * 2, dtype=np.int32)]
    hidden_counts = [0, 0]
    print("Simulating checkpoint matchup with normal fog of war...", flush=True)
    for tick in range(1, cfg.truncation + 1):
        actions = []
        for player in range(2):
            # Full state is used only for the recording; each policy sees its own fogged observation.
            observation = get_observation(state, player)
            hidden_counts[player] += int(np.count_nonzero(np.asarray(observation.fog_cells)))
            action, histories[player] = choose(models[player], observation, histories[player])
            actions.append(action)
        actions = jnp.stack(actions)
        timestep, state = step_fn(state, actions, pool)
        # Preserve the terminal board, not the environment's automatic reset.
        for field in fields:
            frames[field].append(np.asarray(getattr(timestep.last_state, field)).copy())
        actions_log.append(np.asarray(actions).copy())
        if tick % 100 == 0:
            print(f"Simulated {tick} ticks", flush=True)
        if bool(timestep.terminated) or bool(timestep.truncated):
            break
    arrays = {field: np.stack(values) for field, values in frames.items()}
    arrays["actions"] = np.stack(actions_log)
    assert all(n > 0 for n in hidden_counts), "Expected fogged policy observations on both sides"
    assert np.array_equal(arrays["time"], np.arange(tick + 1))
    winner = int(timestep.info.winner)
    if bool(timestep.terminated):
        assert winner in (0, 1)
    else:
        assert bool(timestep.truncated) and winner == -1
    np.savez_compressed(args.output / "game.npz", **arrays)
    record = {"created_at": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "opponent_checkpoint": str(opponent_path),
              "opponent_sha256": hashlib.sha256(opponent_path.read_bytes()).hexdigest(),
              "player_labels": [args.label, getattr(args, "opponent_label", None) or args.label],
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "upstream": json.loads((ROOT / "upstream-lock.json").read_text()),
              "checkpoint_label": args.label, "seed": args.seed, "grid_size": args.grid_size,
              "ticks": tick, "winner": winner, "terminated": bool(timestep.terminated),
              "truncated": bool(timestep.truncated), "policy_observation": "normal fog of war",
              "spectator_visibility": "entire map", "action_selection": "greedy, same as checkpoint evaluations",
              "hidden_cell_observations": hidden_counts,
              "generals_distance": [env.min_generals_distance, env.max_generals_distance],
              "device": str(jax.devices()[0]), "frozen": True}
    (args.output / "game.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"simulation": "complete", "ticks": tick, "winner": winner}), flush=True)
    return arrays, record


class SpectatorRenderer:
    WIDTH, HEIGHT = 1920, 1080
    RED, BLUE = (213, 68, 77), (53, 111, 211)
    INK, MUTED = (27, 44, 59), (101, 118, 134)

    def __init__(self, data, record):
        self.data, self.record = data, record
        font_path = ROOT / "vendor/generals-bots/generals/assets/fonts/Quicksand-SemiBold.ttf"
        self.fonts = {n: ImageFont.truetype(str(font_path), n) for n in (16, 19, 22, 26, 32, 44, 52)}
        self.size = record["grid_size"]
        self.cell = 920 // self.size
        self.x, self.y = 46, 122
        self.army = (data["armies"][:, None] * data["ownership"]).sum(axis=(2, 3))
        self.land = data["ownership"].sum(axis=(2, 3))

    def frame(self, index, ending=False):
        im = Image.new("RGB", (self.WIDTH, self.HEIGHT), (241, 245, 249))
        d = ImageDraw.Draw(im)

        def text(x, y, value, size=22, color=None, **kwargs):
            d.text((x, y), str(value), font=self.fonts[size], fill=color or self.INK, **kwargs)

        text(46, 30, f"GENERALS  /  CHECKPOINT {self.record['checkpoint_label']}", 44)
        same_weights = self.record.get("opponent_sha256", self.record["checkpoint_sha256"]) == self.record["checkpoint_sha256"]
        mode = "SELF-PLAY" if same_weights else "CHECKPOINT MATCH"
        text(47, 84, f"{mode}  ·  FULL-MAP SPECTATOR VIEW", 19, self.MUTED)
        tick = int(self.data["time"][index])
        text(1866, 42, f"Turn {tick / 2:g}  ·  2× playback", 26, anchor="ra")
        own = self.data["ownership"][index]
        armies = self.data["armies"][index]
        mountains = self.data["mountains"][index]
        generals = self.data["generals"][index]
        castles = self.data["castles"][index]
        colors = (self.RED, self.BLUE)
        for r in range(self.size):
            for c in range(self.size):
                x, y = self.x + c * self.cell, self.y + r * self.cell
                owner = 0 if own[0, r, c] else 1 if own[1, r, c] else -1
                fill = colors[owner] if owner >= 0 else (222, 229, 237)
                if mountains[r, c]:
                    fill = (75, 91, 106)
                elif castles[r, c] or generals[r, c]:
                    fill = tuple(int(v * .78) for v in fill)
                d.rectangle((x, y, x+self.cell-1, y+self.cell-1), fill=fill, outline=(172, 184, 197))
                center = x + self.cell / 2
                if mountains[r, c]:
                    d.polygon([(x+7, y+29), (center, y+10), (x+self.cell-7, y+29)], fill=(158, 175, 189))
                elif generals[r, c]:
                    d.polygon([(x+8,y+9),(x+13,y+14),(center,y+6),(x+self.cell-13,y+14),
                               (x+self.cell-8,y+9),(x+self.cell-11,y+20),(x+11,y+20)], fill=(255,224,131))
                elif castles[r, c]:
                    d.rectangle((x+11,y+10,x+self.cell-11,y+21), fill=(231,237,245))
                    for dx in (11,18,25):
                        d.rectangle((x+dx,y+6,x+dx+3,y+11), fill=(231,237,245))
                if armies[r, c] > 0 and not mountains[r, c]:
                    cy = y + (29 if castles[r, c] or generals[r, c] else self.cell/2)
                    text(center, cy, int(armies[r,c]), 16 if armies[r,c] >= 100 else 19,
                         (255,255,255) if owner >= 0 else (27,44,59), anchor="mm",
                         stroke_width=1 if owner >= 0 else 0, stroke_fill=fill)

        # Draw the most recent requested move for each player.
        for player, action in enumerate(self.data["actions"][index]):
            passed, r, c, direction, half = map(int, action)
            if passed:
                continue
            dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[direction]
            x1, y1 = self.x+(c+.5)*self.cell, self.y+(r+.5)*self.cell
            x2, y2 = x1+dc*self.cell*.7, y1+dr*self.cell*.7
            d.line((x1,y1,x2,y2), fill=(20,30,45), width=6)
            d.line((x1,y1,x2,y2), fill=(255,233,154), width=3)
            angle = math.atan2(y2-y1,x2-x1)
            d.polygon([(x2,y2),(x2-9*math.cos(angle-.5),y2-9*math.sin(angle-.5)),
                       (x2-9*math.cos(angle+.5),y2-9*math.sin(angle+.5))], fill=(255,233,154))

        px, right = 1030, 1868
        for player, title in enumerate(("RED", "BLUE")):
            top = 125 + player * 215
            d.rounded_rectangle((px,top,right,top+187), radius=20, fill="white")
            d.rounded_rectangle((px,top,px+8,top+187), radius=4, fill=colors[player])
            label = self.record.get("player_labels", [self.record["checkpoint_label"]] * 2)[player]
            text(px+28,top+18,f"{title}  ·  ITERATION {label} EMA",26,colors[player])
            cities = int((own[player] & castles).sum())
            for offset, label, value in ((28,"ARMY",self.army[index,player]),
                                        (325,"LAND",self.land[index,player]),(600,"CITIES",cities)):
                text(px+offset,top+65,int(value),52)
                text(px+offset,top+132,label,19,self.MUTED)

        text(px,585,"Army over time",26)
        cx, cy, cw, ch = px, 645, right-px, 180
        d.rounded_rectangle((cx,cy-10,right,cy+ch+14),radius=16,fill="white")
        maximum = max(10, int(self.army[:index+1].max()) + 1)
        for frac in (.25,.5,.75):
            y=cy+ch*(1-frac)
            d.line((cx+14,y,right-14,y),fill=(226,233,240),width=1)
        for player in range(2):
            points=[(cx+16+(cw-32)*i/max(index,1),cy+ch-ch*int(v)/maximum)
                    for i,v in enumerate(self.army[:index+1,player])]
            if len(points)>1:
                d.line(points,fill=colors[player],width=4)
        text(cx,cy+ch+25,"Start",16,self.MUTED)
        text(right,cy+ch+25,f"Turn {tick/2:g}",16,self.MUTED,anchor="ra")
        text(px,912,"Both bots see normal fog of war.",26)
        text(px,952,"The spectator sees every tile and both armies.",22,self.MUTED)
        text(px,993,f"{self.size} × {self.size} map  ·  Seed {self.record['seed']}  ·  Frozen weights",19,self.MUTED)

        if ending:
            winner=self.record["winner"]
            title = ("RED WINS" if winner==0 else "BLUE WINS") if winner >= 0 else "DRAW — TIME LIMIT"
            d.rounded_rectangle((self.x+100,self.y+360,self.x+self.size*self.cell-100,self.y+525),
                                radius=18,fill=(20,33,47),outline=(255,255,255),width=2)
            text(self.x+self.size*self.cell/2,self.y+403,title,44,(255,255,255),anchor="mm")
            text(self.x+self.size*self.cell/2,self.y+466,f"Final turn: {tick/2:g}",26,(208,221,233),anchor="mm")
        return im


def render(args, data, record):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install ffmpeg to encode the recording")
    renderer = SpectatorRenderer(data, record)
    movie = args.output / f"checkpoint-{record['checkpoint_label']}-spectator.mp4"
    temporary = movie.with_name(movie.stem + ".tmp.mp4")
    # Four ticks/second is twice the normal two ticks/second game speed.
    command = [ffmpeg,"-hide_banner","-loglevel","warning","-y","-f","rawvideo",
               "-pix_fmt","rgb24","-s","1920x1080","-framerate","4","-i","pipe:0",
               "-an","-vf","fps=30","-c:v","libx264","-preset","fast","-crf","20",
               "-pix_fmt","yuv420p","-movflags","+faststart",str(temporary)]
    with (args.output / "encoding.log").open("w") as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            first = renderer.frame(0)
            first.save(args.output / "first-frame.png")
            for _ in range(8):
                proc.stdin.write(first.tobytes())
            for i in range(1, len(data["time"])):
                frame = renderer.frame(i)
                proc.stdin.write(frame.tobytes())
                if i == len(data["time"])//2:
                    frame.save(args.output / "midpoint.png")
                if i % 200 == 0:
                    print(f"Rendered {i}/{record['ticks']} ticks",flush=True)
            last = renderer.frame(len(data["time"])-1, ending=True)
            last.save(args.output / "final-frame.png")
            for _ in range(20):
                proc.stdin.write(last.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("Encoding failed; see encoding.log")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    temporary.replace(movie)
    record.update(video=str(movie), playback_speed=2, width=1920, height=1080, fps=30,
                  duration_seconds=(record["ticks"]+28)/4,
                  video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    (args.output / "game.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"video":str(movie),"seconds":record['duration_seconds'],"bytes":movie.stat().st_size}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"runs/spectator-3000/L_7d_gae90_ema_3000.eqx")
    parser.add_argument("--opponent",type=Path,help="Optional different network-only EMA checkpoint for blue")
    parser.add_argument("--opponent-label",help="Blue checkpoint iteration label")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/averagejoe-published.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/spectator-3000/game-seed-3000")
    parser.add_argument("--seed",type=int,default=3000)
    parser.add_argument("--grid-size",type=int,default=23)
    parser.add_argument("--label",default="3000")
    parser.add_argument("--render-only",action="store_true")
    parser.add_argument("--simulate-only",action="store_true")
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.checkpoint=args.checkpoint.resolve()
    if args.opponent is not None:
        args.opponent=args.opponent.resolve()
    args.config=args.config.resolve()
    if args.render_only:
        with np.load(args.output/"game.npz",allow_pickle=False) as archive:
            data={key:archive[key] for key in archive.files}
        record=json.loads((args.output/"game.json").read_text())
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        data,record=simulate(args)
    if not args.simulate_only:
        render(args,data,record)


if __name__=="__main__":
    main()
