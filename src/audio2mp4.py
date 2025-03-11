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
from mutagen import File as MutagenFile # 追加: mutagenでID3タグ読み込み

from draw_text import draw_texts

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
  ax.axis("off")
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

def update_frame(frame: Artist, y: np.ndarray, sr: int, duration: int, bars: list[Rectangle],
                 viz_size: tuple[int, int], video_size: tuple[int, int], n_bars: int=50,
                 text_objs:list[plt.Text]=None, fade_start: int=2, fade_duration: int=1) -> tuple[Rectangle]:
  """
  アニメーション更新用のコールバック関数。バーの更新に加え、指定があればテキストのフェードインも行う
  各テキストオブジェクトは、fade_start属性がある場合はその値を用いる。

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
      # 各オブジェクトは属性fade_startが設定されていればその値を用いる（なければ共通のfade_startを使用）
      obj_delay = getattr(obj, "fade_start", fade_start)
      if time >= obj_delay:
        alpha = min((time - obj_delay) / fade_duration, 1.0)
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
  title_bg_color: str="darkblue",
  title_edge_color: str="black",
  title_text_color: str="white",        # 新規追加：タイトル文字色
  subtitle: str=None,
  subtitle_bg_color: str="darkgreen",
  subtitle_edge_color: str="black",
  subtitle_text_color: str="white",      # 新規追加：サブタイトル文字色
  summary: str=None,
  summary_bg_color: str="darkred",
  summary_edge_color: str="black",
  summary_text_color: str="white",       # 新規追加：詳細文文字色
  textarea_bg_color: str="white",        # 新規追加：テキストエリア全体の背景色
  logo_image: str=None,                  # 新規追加：ロゴ画像パス
  logo_width: int=0,                     # 新規追加：ロゴ画像幅
  logo_height: int=0                     # 新規追加：ロゴ画像高さ
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
  title_bg_color: タイトル背景色
  title_edge_color: タイトル枠線色
  title_text_color: タイトル文字色
  subtitle: サブタイトルテキスト
  subtitle_bg_color: サブタイトル背景色
  subtitle_edge_color: サブタイトル枠線色
  subtitle_text_color: サブタイトル文字色
  summary: 詳細文テキスト
  summary_bg_color: 詳細文背景色
  summary_edge_color: 詳細文枠線色
  summary_text_color: 詳細文文字色
  textarea_bg_color: テキストエリア背景色
  logo_image: ロゴ画像のパス
  logo_width: ロゴ画像の幅
  logo_height: ロゴ画像の高さ
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

  # テキストやロゴの表示（テキスト領域全体の背景色、ロゴ配置等の処理を必要に応じて追加可能）
  # ここではdraw_textsに渡すテキスト辞書に、新たな「text_color」を項目として追加しています。
  text_objs, text_area_position = draw_texts(fig, ax, video_size,
    {
      "title": {
        "text": title,
        "bgcolor": title_bg_color,
        "edgecolor": title_edge_color,
        "text_color": title_text_color
      },
      "subtitle": {
        "text": subtitle,
        "bgcolor": subtitle_bg_color,
        "edgecolor": subtitle_edge_color,
        "text_color": subtitle_text_color
      },
      "summary": {
        "text": summary,
        "bgcolor": summary_bg_color,
        "edgecolor": summary_edge_color,
        "text_color": summary_text_color
      }
    }
  )

  # ※ ここで textarea_bg_color や logo_image, logo_width, logo_height を使った処理を追加可能
  # 例： ax.imshow(logo_img) など
  # text_area_position を使用して、次の描画位置を調整できます。

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
  import json, os
  parser = argparse.ArgumentParser(description='Create audio visualizer video from MP3')
  parser.add_argument('mp3_path', help='Path to input MP3 file')
  parser.add_argument('--config', help='Path to JSON configuration file')
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
  parser.add_argument('--title-bg-color', default='darkblue', help='Title background color')
  parser.add_argument('--title-edge-color', default='black', help='Title edge color')
  parser.add_argument('--title-text-color', default='white', help='Title text color')  # 新規追加：タイトル文字色
  parser.add_argument('--subtitle', help='Subtitle text to display (default: ID3 title)')
  parser.add_argument('--subtitle-bg-color', default='darkgreen', help='Subtitle background color')
  parser.add_argument('--subtitle-edge-color', default='black', help='Subtitle edge color')
  parser.add_argument('--subtitle-text-color', default='white', help='Subtitle text color')  # 新規追加：サブタイトル文字色
  parser.add_argument('--summary', help='Summary text to display (default: first line of ID3 comment)')
  parser.add_argument('--summary-bg-color', default='darkred', help='Summary background color')
  parser.add_argument('--summary-edge-color', default='black', help='Summary edge color')
  parser.add_argument('--summary-text-color', default='white', help='Summary text color')  # 新規追加：詳細文文字色
  parser.add_argument('--textarea-bg-color', default='white', help='Textarea background color')  # 新規追加：テキストエリア全体の背景色
  parser.add_argument('--logo-image', help='Path to logo image file')  # 新規追加：ロゴ画像パス
  parser.add_argument('--logo-width', type=int, default=0, help='Logo image width')  # 新規追加：ロゴ画像幅
  parser.add_argument('--logo-height', type=int, default=0, help='Logo image height')  # 新規追加：ロゴ画像高さ

  args = parser.parse_args()

  config = {}
  if args.config and os.path.isfile(args.config):
    with open(args.config, 'r', encoding='utf-8') as f:
      config = json.load(f)

  # 動画サイズ（コマンドライン > JSON > デフォルト）
  video_width = args.video_width if args.video_width is not None else config.get("video", {}).get("width", 1280)
  video_height = args.video_height if args.video_height is not None else config.get("video", {}).get("height", 720)
  video_size = (video_width, video_height)

  # ビジュアライザーサイズと色
  viz_width = args.viz_width if args.viz_width is not None else config.get("vizualizer", {}).get("width", 1080)
  viz_height = args.viz_height if args.viz_height is not None else config.get("vizualizer", {}).get("height", 520)
  viz_size = (viz_width, viz_height)
  viz_color = args.viz_color if args.viz_color is not None else config.get("vizualizer", {}).get("color", "gradation")

  # 背景関連
  bg_color = args.bg_color if args.bg_color is not None else config.get("background", {}).get("color", "black")
  bg_image = args.bg_image if args.bg_image is not None else config.get("background", {}).get("image", None)
  bg_image_type = args.bg_image_type if args.bg_image_type is not None else config.get("background", {}).get("image_type", "streach")

  # その他パラメータ
  loop_count = args.loop_count if args.loop_count is not None else config.get("loop_count", 1)
  effect = args.effect if args.effect is not None else config.get("effect", "")

  # テキスト領域（textarea）から値を取得（コマンドライン優先）
  textarea = config.get("textarea", {})
  title_text = args.title if args.title is not None else textarea.get("title", {}).get("text", None)
  title_bg_color = args.title_bg_color if args.title_bg_color is not None else textarea.get("title", {}).get("bg_color", "darkblue")
  title_edge_color = args.title_edge_color if args.title_edge_color is not None else textarea.get("title", {}).get("edge_color", "black")
  title_text_color = args.title_text_color if args.title_text_color is not None else textarea.get("title", {}).get("text_color", "white")

  subtitle_text = args.subtitle if args.subtitle is not None else textarea.get("subtitle", {}).get("text", None)
  subtitle_bg_color = args.subtitle_bg_color if args.subtitle_bg_color is not None else textarea.get("subtitle", {}).get("bg_color", "darkgreen")
  subtitle_edge_color = args.subtitle_edge_color if args.subtitle_edge_color is not None else textarea.get("subtitle", {}).get("edge_color", "black")
  subtitle_text_color = args.subtitle_text_color if args.subtitle_text_color is not None else textarea.get("subtitle", {}).get("text_color", "white")

  summary_text = args.summary if args.summary is not None else textarea.get("summary", {}).get("text", None)
  summary_bg_color = args.summary_bg_color if args.summary_bg_color is not None else textarea.get("summary", {}).get("bg_color", "darkred")
  summary_edge_color = args.summary_edge_color if args.summary_edge_color is not None else textarea.get("summary", {}).get("edge_color", "black")
  summary_text_color = args.summary_text_color if args.summary_text_color is not None else textarea.get("summary", {}).get("text_color", "white")

  textarea_bg_color = args.textarea_bg_color if args.textarea_bg_color is not None else textarea.get("bg_color", "white")
  logo_image = args.logo_image if args.logo_image is not None else config.get("logo", {}).get("image", None)
  logo_width = args.logo_width if args.logo_width is not None else config.get("logo", {}).get("width", 0)
  logo_height = args.logo_height if args.logo_height is not None else config.get("logo", {}).get("height", 0)

  create_audio_visualizer(
    args.mp3_path,
    args.output,
    video_size,
    viz_size,
    bg_color,
    viz_color,
    loop_count,
    bg_image,
    bg_image_type,
    effect,
    title_text,
    title_bg_color,
    title_edge_color,
    title_text_color,
    subtitle_text,
    subtitle_bg_color,
    subtitle_edge_color,
    subtitle_text_color,
    summary_text,
    summary_bg_color,
    summary_edge_color,
    summary_text_color,
    textarea_bg_color,
    logo_image,
    logo_width,
    logo_height
  )

if __name__ == '__main__':
  main()