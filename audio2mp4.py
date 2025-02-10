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
from PIL import Image
import soundfile as sf

def create_fade_mask(duration_seconds, sr=44100, fade_duration=2):
  """フェードアウトマスクを作成する関数"""
  total_samples = int(duration_seconds * sr)
  fade_samples = int(fade_duration * sr)

  # 通常の音量部分（1.0）を作成
  mask = np.ones(total_samples)

  # フェードアウト部分を作成
  fade_start = total_samples - fade_samples
  fade_curve = np.linspace(1.0, 0.0, fade_samples)
  mask[fade_start:] = fade_curve

  return mask

def create_audio_visualizer(
  mp3_path,
  output_path="audio.mp4",
  video_size=(1280, 720),
  viz_size=(1080, 520),
  bg_color="black",
  viz_color="gradation",
  loop_count=1,
  bg_image_path=None,
  effect=""
):
  # Load audio file
  y, sr = librosa.load(mp3_path)
  single_duration = librosa.get_duration(y=y, sr=sr)

  # If loop count is greater than 1, extend the audio
  if loop_count > 1:
      y = np.tile(y, loop_count)

  # Apply fade effect if specified (only to the last loop)
  if effect == "fade":
    # 全体の長さのマスクを作成
    total_samples = len(y)
    mask = np.ones(total_samples)

    # 最後のループの開始位置を計算
    fade_start_time = single_duration * (loop_count - 1)
    fade_start_sample = int(fade_start_time * sr)

    # 最後のループ部分にフェードマスクを適用
    last_loop_mask = create_fade_mask(single_duration, sr)
    mask[fade_start_sample:] = last_loop_mask

    # マスクを適用
    y = y * mask

  duration = single_duration * loop_count

  # Create temporary audio file with modifications
  with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
    temp_audio_path = temp_audio.name
    sf.write(temp_audio_path, y, sr, format='WAV')

  # Set up the figure
  fig, ax = plt.subplots(figsize=(video_size[0]/100, video_size[1]/100), facecolor=bg_color)
  ax.set_facecolor(bg_color)

  # Set background image if provided
  if bg_image_path:
    img = Image.open(bg_image_path)
    img = img.resize(video_size)
    ax.imshow(img, extent=[0, video_size[0], 0, video_size[1]])

  # Calculate visualization area
  viz_left = (video_size[0] - viz_size[0]) / 2
  viz_bottom = (video_size[1] - viz_size[1]) / 2

  # Set axis limits and remove ticks
  ax.set_xlim(0, video_size[0])
  ax.set_ylim(0, video_size[1])
  ax.set_xticks([])
  ax.set_yticks([])

  # Number of bars in visualization
  n_bars = 50

  # Create color array for bars
  if viz_color == "gradation":
    colors = LinearSegmentedColormap.from_list("custom", ["blue", "red"])(np.linspace(0, 1, n_bars))
  else:
    colors = [viz_color] * n_bars

  # Calculate bar positions to center them in the visualization area
  bar_positions = np.linspace(viz_left, viz_left + viz_size[0], n_bars)

  # Create bars with individual colors
  bars = ax.bar(
    bar_positions,
    np.zeros(n_bars),
    width=viz_size[0]/(n_bars*1.5),
    color=colors,
    bottom=viz_bottom
  )

  # Frame update function
  def update(frame):
    time = (frame / 30) % duration
    start_idx = int(time * sr)
    end_idx = start_idx + sr // 30

    if end_idx <= len(y):
      chunk = y[start_idx:end_idx]
      fft = np.abs(np.fft.fft(chunk))
      fft = fft[:len(fft)//2]

      fft = fft / np.max(fft) if np.max(fft) > 0 else fft
      fft = fft * viz_size[1]

      for bar, val in zip(bars, np.interp(np.linspace(0, len(fft), n_bars), np.arange(len(fft)), fft)):
        bar.set_height(val)

    return bars

  # Create animation
  total_frames = int(duration * 30)
  anim = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=1000/30,
    blit=True
  )

  # Save animation to temporary file
  with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
    temp_video_path = temp_video.name
    anim.save(temp_video_path, writer='ffmpeg', fps=30)

  try:
    # Combine with audio
    video = VideoFileClip(temp_video_path)
    audio = AudioFileClip(temp_audio_path)
    final_video = video.with_audio(audio)

    # Write final video
    final_video.write_videofile(
      output_path,
      codec='libx264',
      audio_codec='aac',
      fps=30,
      preset='medium',
      bitrate='5000k'
    )

  finally:
    # Clean up
    plt.close()
    if 'video' in locals():
      video.close()
    if 'audio' in locals():
      audio.close()
    if 'final_video' in locals():
      final_video.close()
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
  parser.add_argument('--viz-color', default='gradation', help='Visualizer color (use "gradation" for blue-to-red gradient)')
  parser.add_argument('--loop-count', type=int, default=1, help='Number of times to loop the audio')
  parser.add_argument('--bg-image', help='Path to background image file')
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
    args.effect
  )

if __name__ == '__main__':
  main()