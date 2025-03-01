import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pathlib import Path
import cv2
import tempfile
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageOps
import soundfile as sf

def create_fade_mask(duration_seconds, sr=44100, fade_duration=2):
  """フェードアウトマスクを作成する関数"""
  total_samples = int(duration_seconds * sr)
  fade_samples = int(fade_duration * sr)
  mask = np.ones(total_samples)
  fade_start = total_samples - fade_samples
  fade_curve = np.linspace(1.0, 0.0, fade_samples)
  mask[fade_start:] = fade_curve
  return mask

def prepare_audio(mp3_path, loop_count, effect):
  """音声の読み込み、ループやフェードエフェクトの適用、及びテンポラリ音声ファイルの作成"""
  y, sr = librosa.load(mp3_path)
  single_duration = librosa.get_duration(y=y, sr=sr)

  if loop_count > 1:
    y = np.tile(y, loop_count)

  if effect == "fade":
    total_samples = len(y)
    fade_start_time = single_duration * (loop_count - 1)
    fade_start_sample = int(fade_start_time * sr)
    last_loop_mask = create_fade_mask(single_duration, sr)
    mask = np.ones(total_samples)
    mask[fade_start_sample:] = last_loop_mask
    y = y * mask

  duration = single_duration * loop_count

  with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
    temp_audio_path = temp_audio.name
  sf.write(temp_audio_path, y, sr, format='WAV')

  return y, sr, duration, temp_audio_path

def setup_figure(video_size, bg_color):
  """図と軸のセットアップを行う"""
  fig, ax = plt.subplots(figsize=(video_size[0]/100, video_size[1]/100), facecolor=bg_color)
  ax.set_facecolor(bg_color)
  ax.set_xlim(0, video_size[0])
  ax.set_ylim(0, video_size[1])
  ax.set_xticks([])
  ax.set_yticks([])
  return fig, ax

def draw_background(ax, video_size, bg_image_path, bg_image_type):
  """背景画像が指定されていれば描画する"""
  if bg_image_path:
    img = Image.open(bg_image_path)
    if bg_image_type == "streach":
      img_ratio = img.width / img.height
      video_ratio = video_size[0] / video_size[1]
      if img_ratio > video_ratio:
        new_width = video_size[0]
        new_height = int(new_width / img_ratio)
      else:
        new_height = video_size[1]
        new_width = int(new_height * img_ratio)
      img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
      left = (video_size[0] - new_width) // 2
      bottom = (video_size[1] - new_height) // 2
      ax.imshow(img, extent=[left, left + new_width, bottom, bottom + new_height])
    elif bg_image_type == "center":
      w, h = img.size
      left = (video_size[0] - w) / 2
      bottom = (video_size[1] - h) / 2
      ax.imshow(img, extent=[left, left + w, bottom, bottom + h])
    elif bg_image_type == "tile":
      new_img = Image.new('RGB', video_size)
      w, h = img.size
      for i in range(0, video_size[0], w):
        for j in range(0, video_size[1], h):
          new_img.paste(img, (i, j))
      ax.imshow(new_img, extent=[0, video_size[0], 0, video_size[1]])

def setup_visualizer(ax, video_size, viz_size, viz_color, n_bars=50):
  """ビジュアライザーのバーを作成する"""
  viz_left = (video_size[0] - viz_size[0]) / 2
  viz_bottom = (video_size[1] - viz_size[1]) / 2

  if viz_color == "gradation":
    colors = LinearSegmentedColormap.from_list("custom", ["red", "blue"])(np.linspace(0, 1, n_bars))
  else:
    colors = [viz_color] * n_bars

  bar_positions = np.linspace(viz_left, viz_left + viz_size[0], n_bars)
  bars = ax.bar(
    bar_positions,
    np.zeros(n_bars),
    width=viz_size[0] / (n_bars * 1.5),
    color=colors,
    bottom=viz_bottom
  )
  return bars

def update_frame(frame, y, sr, duration, bars, viz_size, video_size, n_bars=50):
  """アニメーション更新用のコールバック関数"""
  time = (frame / 30) % duration
  start_idx = int(time * sr)
  end_idx = start_idx + sr // 30

  if end_idx <= len(y):
    chunk = y[start_idx:end_idx]
    fft = np.abs(np.fft.fft(chunk))
    fft = fft[:len(fft) // 2]

    fft_max = np.max(fft)
    if fft_max > 0:
      fft = fft / fft_max
    fft = fft * viz_size[1]

    new_heights = np.interp(np.linspace(0, len(fft), n_bars),
                            np.arange(len(fft)),
                            fft)
    for bar, val in zip(bars, new_heights):
      bar.set_height(val)

  return bars

def create_animation(fig, y, sr, duration, bars, viz_size, video_size, n_bars=50):
  """FuncAnimationによるアニメーションの作成"""
  total_frames = int(duration * 30)
  anim = FuncAnimation(
    fig,
    lambda frame: update_frame(frame, y, sr, duration, bars, viz_size, video_size, n_bars),
    frames=total_frames,
    interval=1000 / 30,
    blit=True
  )
  return anim

def combine_video_audio(temp_video_path, temp_audio_path, output_path):
  """動画と音声を合成して最終動画を出力する"""
  video = VideoFileClip(temp_video_path)
  audio = AudioFileClip(temp_audio_path)
  final_video = video.with_audio(audio)

  final_video.write_videofile(
    output_path,
    codec='libx264',
    audio_codec='aac',
    fps=30,
    preset='medium',
    bitrate='5000k'
  )
  video.close()
  audio.close()
  final_video.close()

def create_audio_visualizer(
  mp3_path,
  output_path="audio.mp4",
  video_size=(1280, 720),
  viz_size=(1080, 520),
  bg_color="black",
  viz_color="gradation",
  loop_count=1,
  bg_image_path=None,
  bg_image_type="streach",
  effect=""
):
  # 音声の準備
  y, sr, duration, temp_audio_path = prepare_audio(mp3_path, loop_count, effect)

  # 図と軸のセットアップ
  fig, ax = setup_figure(video_size, bg_color)
  draw_background(ax, video_size, bg_image_path, bg_image_type)

  # ビジュアライザーの準備
  n_bars = 50
  bars = setup_visualizer(ax, video_size, viz_size, viz_color, n_bars)

  # アニメーション生成
  anim = create_animation(fig, y, sr, duration, bars, viz_size, video_size, n_bars)

  # テンポラリ動画ファイルに保存
  with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
    temp_video_path = temp_video.name
  anim.save(temp_video_path, writer='ffmpeg', fps=30)

  try:
    combine_video_audio(temp_video_path, temp_audio_path, output_path)
  finally:
    plt.close(fig)
    Path(temp_video_path).unlink()
    Path(temp_audio_path).unlink()

def main():
  parser = argparse.ArgumentParser(description='Create audio visualizer video from MP3')
  parser.add_argument('mp3_path', help='Path to input MP3 file')
  parser.add_argument('--output', default='audio.mp4', help='Output MP4 file path')
  parser.add_argument('--video-width', type=int, default=1280, help='Video width in pixels')
  parser.add_argument('--video-height', type=int, default=720, help='Video height in pixels')
  parser.add_argument('--viz-width', type=int, default=1080, help='Visualizer width in pixels')
  parser.add_argument('--viz-height', type=int, default=520, help='Visualizer height in pixels')
  parser.add_argument('--bg-color', default='black', help='Background color')
  parser.add_argument('--viz-color', default='gradation', help='Visualizer color (use "gradation" for red-to-blue gradient)')
  parser.add_argument('--loop-count', type=int, default=1, help='Number of times to loop the audio')
  parser.add_argument('--bg-image', help='Path to background image file')
  parser.add_argument('--bg-image-type', default='streach', help='Background image type (streach, center, tile)')
  parser.add_argument('--effect', default='', help='Audio effect (use "fade" for fade-out effect)')

  args = parser.parse_args()

  create_audio_visualizer(
    args.mp3_path,
    args.output,
    (args.video_width, args.video_height),
    (args.viz_width, args.viz_height),
    args.bg_color,
    args.viz_color,
    args.loop_count,
    args.bg_image,
    args.bg_image_type,
    args.effect
  )

if __name__ == '__main__':
  main()