import argparse
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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
from mutagen import File as MutagenFile  # 追加: mutagenでID3タグ読み込み

def create_fade_mask(duration_seconds: int, sr: int=44100, fade_duration: int=2) -> np.ndarray:
  """
  フェードアウトマスクを作成する関数

  duration_seconds: 音声の長さ（秒）
  sr: サンプリングレート
  fade_duration: フェードアウトの長さ（秒）
  """
  total_samples = int(duration_seconds * sr)
  fade_samples = int(fade_duration * sr)
  mask = np.ones(total_samples)
  fade_start = total_samples - fade_samples
  fade_curve = np.linspace(1.0, 0.0, fade_samples)
  mask[fade_start:] = fade_curve
  return mask

def prepare_audio(mp3_path: str, loop_count: int, effect: str) -> tuple[np.ndarray, int, float, str]:
  """
  音声の読み込み、ループやフェードエフェクトの適用、及びテンポラリ音声ファイルの作成

  mp3_path: 入力MP3ファイルのパス
  loop_count: 音声のループ回数
  effect: フェードアウトエフェクトの指定（現時点では"fade"のみ）
  """
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

def setup_figure(video_size: tuple[int, int], bg_color: str) -> tuple[Figure, Axes]:
  """
  図と軸のセットアップを行う

  video_size: 動画のサイズ
  bg_color: 背景色
  """
  fig, ax = plt.subplots(figsize=(video_size[0]/100, video_size[1]/100), facecolor=bg_color)
  ax.set_facecolor(bg_color)
  ax.set_xlim(0, video_size[0])
  ax.set_ylim(0, video_size[1])
  ax.set_xticks([])
  ax.set_yticks([])
  return fig, ax

def draw_background(ax: Axes, video_size: tuple[int, int], bg_image_path: str, bg_image_type: str) -> None:
  """
  背景画像が指定されていれば描画する

  ax: 軸オブジェクト
  video_size: 動画のサイズ
  bg_image_path: 背景画像のパス
  bg_image_type: 背景画像の描画方法（streach, center, tile）
  """
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

def setup_visualizer(ax: Axes, video_size: tuple[int, int], viz_size: tuple[int, int], viz_color: str, n_bars: int=50) -> list[plt.Rectangle]:
  """
  ビジュアライザーのバーを作成する

  ax: 軸オブジェクト
  video_size: 動画のサイズ
  viz_size: ビジュアライザーのサイズ
  viz_color: ビジュアライザーの色
  n_bars: バーの本数
  """
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

def draw_texts(fig: Figure, ax: Axes, video_size: tuple[int, int], texts: dict[str, str], init_alpha: float=0) -> list[plt.Text]:
  """
  テキストを描画する。タイトル、サブタイトル、詳細文をそれぞれ描画する

  fig: 図オブジェクト
  ax: 軸オブジェクト
  video_size: 動画のサイズ
  texts: タイトル、サブタイトル、詳細文のテキストを格納した辞書
    title: タイトルのテキスト
    subtitle: サブタイトルのテキスト
    summary: 詳細文のテキスト
  init_alpha: テキストの初期透明度（デフォルトは0/テスト時などは1を設定）
  """
  text_objs = []
  position = 0
  def add_text(text, fontsize=16, bgcolor="darkblue", bgalpha=0.5):
    nonlocal position
    text = draw_text(ax, video_size, text, init_alpha=init_alpha)
    text.set_fontsize(fontsize)  # タイトルの文字サイズ
    text.set_bbox(dict(facecolor=bgcolor, alpha=bgalpha))  # タイトルの背景色
    text.set_position((10, position + video_size[1] - 30))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    position += text.get_window_extent(renderer).height + 30
    return text

  if "summary" in texts:
    text_objs.append(add_text(texts["summary"], 14, "darkred", 0.5))
  if "subtitle" in texts:
    text_objs.append(add_text(texts["subtitle"], 16, "darkgreen", 0.5))
  if "title" in texts:
    text_objs.append(add_text(texts["title"], 20, "darkblue", 0.5))
  return text_objs

def draw_text(ax: Axes, video_size: tuple[int, int], text: str, font_name: str="BIZ UDGothic", init_alpha:float=0) -> plt.Text:
  """
  テキストを描画する関数。

  ax: 軸オブジェクト
  video_size: 動画のサイズ
  text: 描画するテキスト
  font_name: フォント名（デフォルトは BIZ UDGothic）
  init_alpha: テキストの初期透明度（デフォルトは0/テスト時などは1を設定）
  """
  txt = ax.text(
    10, video_size[1] - 10, text,
    color="white",
    fontsize=16,
    verticalalignment="top",
    alpha=init_alpha,
    fontname=font_name,
    bbox=dict(facecolor="black", alpha=0.5)
  )
  return txt

def update_frame(frame: Artist, y: np.ndarray, sr: int, duration: int, bars: list[Rectangle],
                 viz_size: tuple[int, int], video_size: tuple[int, int], n_bars: int=50,
                 text_objs:list[plt.Text]=None, fade_start: int=2, fade_duration: int=1) -> tuple[Rectangle]:
  """
  アニメーション更新用のコールバック関数。バーの更新に加え、指定があればテキストのフェードインも行う

  frame: 現在のフレーム
  y: 音声データ
  sr: サンプリングレート
  duration: 音声の長さ（秒）
  bars: バーのリスト
  viz_size: ビジュアライザーのサイズ
  video_size: 動画のサイズ
  n_bars: バーの本数
  text_objs: テキストオブジェクトのリスト
  fade_start: フェードイン開始時間
  fade_duration: フェードイン時間
  """
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

  if text_objs:
    for obj in text_objs:
      if time >= fade_start:
        alpha = min((time - fade_start) / fade_duration, 1.0)
      else:
        alpha = 0
      obj.set_alpha(alpha)
    return bars + tuple(text_objs)
  return bars

def create_animation(fig: Figure, y: np.ndarray, sr: int, duration: int, bars: list[Rectangle],
                     viz_size: tuple[int, int], video_size: tuple[int,int], n_bars: int=50,
                     text_objs: list[plt.Text]=None, fade_start: int=2, fade_duration: int=1) -> FuncAnimation:
  """
  FuncAnimation によるアニメーションの作成。テキストオブジェクト群があればフェードインも更新する

  fig: 図オブジェクト
  y: 音声データ
  sr: サンプリングレート
  duration: 音声の長さ（秒）
  bars: バーのリスト
  viz_size: ビジュアライザーのサイズ
  video_size: 動画のサイズ
  n_bars: バーの本数
  text_objs: テキストオブジェクトのリスト
  fade_start: フェードイン開始時間
  fade_duration: フェードイン時間
  """
  total_frames = int(duration * 30)
  anim = FuncAnimation(
    fig,
    lambda frame: update_frame(frame, y, sr, duration, bars, viz_size, video_size, n_bars, text_objs, fade_start, fade_duration),
    frames=total_frames,
    interval=1000 / 30,
    blit=True
  )
  return anim

def combine_video_audio(temp_video_path: str, temp_audio_path: str, output_path: str) -> None:
  """
  動画と音声を合成して最終動画を出力する

  temp_video_path: テンポラリ動画ファイルのパス
  temp_audio_path: テンポラリ音声ファイルのパス
  output_path: 出力動画ファイルのパス
  """
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
  mp3_path: str,
  output_path: str="audio.mp4",
  video_size: tuple[int, int]=(1280, 720),
  viz_size: tuple[int, int]=(1080, 520),
  bg_color: str="black",
  viz_color: str="gradation",
  loop_count: int=1,
  bg_image_path: str=None,
  bg_image_type: str="streach",
  effect: str="",
  title: str=None,
  subtitle: str=None,
  summary: str=None
) -> None:
  """
  Main

  mp3_path: 入力MP3ファイルのパス
  output_path: 出力動画ファイルのパス
  video_size: 動画のサイズ
  viz_size: ビジュアライザーのサイズ
  bg_color: 背景色
  viz_color: ビジュアライザーの色
  loop_count: 音声のループ回数
  bg_image_path: 背景画像のパス
  bg_image_type: 背景画像の描画方法（streach, center, tile）
  effect: フェードアウトエフェクトの指定（現時点では"fade"のみ）
  title: タイトルテキスト
  subtitle: サブタイトルテキスト
  summary: 詳細文テキスト
  """
  # ID3タグから情報取得（省略時）
  audiofile = MutagenFile(mp3_path, easy=True)
  if title is None and 'album' in audiofile:
    title = audiofile['album'][0]
  if subtitle is None and 'title' in audiofile:
    subtitle = audiofile['title'][0]
  if summary is None and 'comment' in audiofile:
    summary = audiofile['comment'][0].splitlines()[0]

  # 音声の準備
  y, sr, duration, temp_audio_path = prepare_audio(mp3_path, loop_count, effect)

  # 図と軸のセットアップ
  fig, ax = setup_figure(video_size, bg_color)
  draw_background(ax, video_size, bg_image_path, bg_image_type)

  # タイトル、サブタイトル、詳細文はそれぞれ個別に描画
  text_objs = draw_texts(fig, ax, video_size, {
    "title": title,
    "subtitle": subtitle,
    "summary": summary
    })

  # ビジュアライザーの準備
  n_bars = 50
  bars = setup_visualizer(ax, video_size, viz_size, viz_color, n_bars)

  # アニメーション生成（テキストオブジェクト群があればフェードインも更新）
  anim = create_animation(fig, y, sr, duration, bars, viz_size, video_size, n_bars, text_objs, fade_start=2, fade_duration=1)

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
  parser.add_argument('--title', help='Title text to display (default: ID3 album name)')
  parser.add_argument('--subtitle', help='Subtitle text to display (default: ID3 title)')
  parser.add_argument('--summary', help='Summary text to display (default: first line of ID3 comment)')

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
    args.effect,
    args.title,
    args.subtitle,
    args.summary
  )

if __name__ == '__main__':
  main()