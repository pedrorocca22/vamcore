"""Verify a real rendered video and extract review artifacts; no project code runs.

Requires ffmpeg/ffprobe and Pillow. Exit 1 reports failed technical checks or a
nearly flat opening. Contact sheets still require human/agent visual inspection.
"""
from pathlib import Path
import argparse
from fractions import Fraction
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageStat


def verify(video,out,size=(1920,1080),fps=24,duration_min=30,duration_max=60,chapters=None):
    video=Path(video).resolve();out=Path(out).resolve()
    if not video.is_file():raise ValueError(f"Video not found: {video}")
    if out==video or out in video.parents:
        raise ValueError("Choose a dedicated QA output folder, not a parent of the input video")
    ffprobe,ffmpeg=shutil.which("ffprobe"),shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:raise RuntimeError("ffprobe and ffmpeg must be on PATH")
    out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(subprocess.check_output([ffprobe,"-v","error","-count_frames","-show_streams","-show_format","-of","json",str(video)],text=True))
    streams=[s for s in metadata["streams"] if s.get("codec_type")=="video" and not s.get("disposition",{}).get("attached_pic")]
    if not streams:raise ValueError("No video stream")
    stream=streams[0];duration=float(metadata["format"]["duration"])
    rate=float(Fraction(stream.get("avg_frame_rate","0/1")))
    errors=[]
    if (stream["width"],stream["height"])!=tuple(size):errors.append("Unexpected resolution")
    if abs(rate-fps)>.01:errors.append("Unexpected frame rate")
    if not duration_min<=duration<=duration_max:errors.append("Duration outside requested bounds")
    if stream["codec_name"]!="h264":errors.append("Expected H.264 for the MP4 delivery")
    if stream.get("pix_fmt")!="yuv420p":errors.append("Expected yuv420p for broad playback support")
    decode=subprocess.run([ffmpeg,"-v","error","-i",str(video),"-f","null","-"],capture_output=True,text=True)
    if decode.returncode or decode.stderr.strip():errors.append("Decode errors: "+decode.stderr.strip()[:500])

    def sample(seconds):
        command=[ffmpeg,"-v","error","-ss",f"{seconds:.6f}","-i",str(video),"-frames:v","1","-f","image2pipe","-c:v","png","-"]
        return Image.open(io.BytesIO(subprocess.check_output(command))).convert("RGB")

    first=sample(0);first.save(out/"first_frame.png")
    opening_std=max(ImageStat.Stat(first).stddev)
    if opening_std<3:errors.append("Opening is nearly uniform; check that product/title is already visible")
    if chapters is None:
        chapters=[dict(label=f"Sample {i+1}",start=duration*i/8,end=duration*(i+1)/8) for i in range(8)]
    for chapter in chapters:
        if not 0<=float(chapter["start"])<float(chapter["end"])<=duration+.06:
            raise ValueError(f"Chapter times outside actual video: {chapter}")
    tiles=[];motion=[]
    for i,chapter in enumerate(chapters):
        start,end=float(chapter["start"]),float(chapter["end"])
        end=min(end,max(0,duration-1/max(rate,1)))
        middle=(start+end)/2
        image=sample(middle);image.save(out/f"scene_{i+1:02d}.png")
        tile=image.resize((480,270),Image.Resampling.LANCZOS)
        ImageDraw.Draw(tile).text((12,12),f"{i+1:02d} / {middle:.2f}s",fill="black",stroke_width=1,stroke_fill="white")
        tiles.append(tile)
        for position,t in enumerate((start+(end-start)*.10,middle,start+(end-start)*.90)):
            image=sample(t);image.save(out/f"motion_{i+1:02d}_{position}.png")
            motion.append(image.resize((480,270),Image.Resampling.LANCZOS))
    sheet=Image.new("RGB",(1920,270*math.ceil(len(tiles)/4)),"#EEF2F3")
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*480,(i//4)*270))
    sheet.save(out/"contact_sheet.jpg",quality=94)
    sheet=Image.new("RGB",(1440,270*len(chapters)),"#EEF2F3")
    for i,tile in enumerate(motion):sheet.paste(tile,((i%3)*480,(i//3)*270))
    sheet.save(out/"motion_contact_sheet.jpg",quality=94)
    report=dict(result="failed" if errors else "passed",video=str(video),duration_seconds=duration,
                size=[stream["width"],stream["height"]],fps=rate,frames=int(stream.get("nb_read_frames",stream.get("nb_frames",0))),
                opening_channel_stddev=opening_std,errors=errors,
                sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                note="Technical checks only; inspect first_frame.png, contact_sheet.jpg and motion_contact_sheet.jpg visually.")
    (out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video",type=Path);parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--size",default="1920x1080");parser.add_argument("--fps",type=float,default=24)
    parser.add_argument("--duration-min",type=float,default=30);parser.add_argument("--duration-max",type=float,default=60)
    parser.add_argument("--chapters",type=Path)
    args=parser.parse_args()
    chapters=json.loads(args.chapters.read_text(encoding="utf-8")) if args.chapters else None
    report=verify(args.video,args.out,tuple(map(int,args.size.lower().split("x"))),args.fps,args.duration_min,args.duration_max,chapters)
    print(json.dumps(report,indent=2));sys.exit(bool(report["errors"]))


if __name__=="__main__":main()
