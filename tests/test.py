import unittest
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from src.audio2mp4 import (
  create_fade_mask,
  prepare_audio,
  setup_figure,
  draw_background,
  setup_visualizer,
  update_frame,
  create_animation,
  combine_video_audio
)

OUT_DIR = Path(__file__).parent / "out"

class TestAudio2MP4(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # テスト実行開始時に out フォルダ内の既存ファイルを削除
    if OUT_DIR.exists():
      for f in OUT_DIR.glob("*"):
        f.unlink()
    else:
      OUT_DIR.mkdir()

  def test_create_fade_mask(self):
    duration = 4
    sr = 1000
    fade_duration = 1
    mask = create_fade_mask(duration, sr, fade_duration)
    self.assertEqual(len(mask), duration * sr)
    # 初期は1、終端は0に近い値になっているかチェック
    self.assertAlmostEqual(mask[0], 1.0)
    self.assertAlmostEqual(mask[-1], 0.0, delta=1e-3)

  def test_prepare_audio(self):
    # 短いサイン波を用いて out フォルダ内に WAV ファイルを作成
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    temp_wav_path = str(OUT_DIR / "test_temp.wav")
    import soundfile as sf
    sf.write(temp_wav_path, y, sr)

    # prepare_audio の呼び出し。ループ回数2の場合、durationは約2秒になる
    y_loaded, sr_loaded, duration, temp_audio_path = prepare_audio(temp_wav_path, 2, "")
    self.assertEqual(sr, sr_loaded)
    self.assertAlmostEqual(duration, 2, places=1)
    self.assertTrue(Path(temp_audio_path).exists())
    # ※削除タイミングは次回テスト実行時 setUpClass で実施するため、ここでは削除しない

  def test_setup_figure(self):
    video_size = (1280, 720)
    bg_color = "black"
    fig, ax = setup_figure(video_size, bg_color)
    self.assertIsInstance(fig, plt.Figure)
    self.assertIsNotNone(ax)
    plt.close(fig)

  def test_draw_background(self):
    # out フォルダ内に背景用画像を作成
    img_path = str(OUT_DIR / "test_bg.png")
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    video_size = (500, 500)
    fig, ax = plt.subplots()
    # "streach", "center", "tile" の各パターンをテスト
    draw_background(ax, video_size, img_path, "streach")
    draw_background(ax, video_size, img_path, "center")
    draw_background(ax, video_size, img_path, "tile")
    plt.close(fig)
    # ファイル削除は setUpClass 時に行う

  def test_setup_visualizer(self):
    video_size = (1280, 720)
    viz_size = (1080, 520)
    fig, ax = plt.subplots()
    bars = setup_visualizer(ax, video_size, viz_size, "gradation", n_bars=10)
    # 10本のバーが作成されているかチェック
    self.assertEqual(len(bars), 10)
    plt.close(fig)

  def test_update_frame(self):
    # サンプル用にサイン波データを作成（サンプルレートを 60 に変更）
    sr = 60
    duration = 1  # 1秒間
    t = np.linspace(0, duration, sr, endpoint=False)
    y = np.sin(2 * np.pi * t)
    video_size = (640, 480)
    viz_size = (600, 400)
    fig, ax = plt.subplots()
    bars = setup_visualizer(ax, video_size, viz_size, "red", n_bars=5)

    # フレーム0 のときに更新
    updated_bars = update_frame(0, y, sr, duration, bars, viz_size, video_size, n_bars=5)
    # 初期値が0だったバーが更新されているか確認
    for bar in updated_bars:
      self.assertNotEqual(bar.get_height(), 0)
    plt.close(fig)

  def test_create_animation(self):
    sr = 60
    duration = 1
    t = np.linspace(0, duration, sr, endpoint=False)
    y = np.sin(2 * np.pi * t)
    video_size = (640, 480)
    viz_size = (600, 400)
    fig, ax = plt.subplots()
    bars = setup_visualizer(ax, video_size, viz_size, "red", n_bars=5)
    anim = create_animation(fig, y, sr, duration, bars, viz_size, video_size, n_bars=5)
    from matplotlib.animation import FuncAnimation
    self.assertIsInstance(anim, FuncAnimation)
    plt.close(fig)

  def test_combine_video_audio(self):
    # combine_video_audio 内の VideoFileClip と AudioFileClip をモックする
    with patch("src.audio2mp4.VideoFileClip") as mock_video, \
         patch("src.audio2mp4.AudioFileClip") as mock_audio:
      video_instance = MagicMock()
      audio_instance = MagicMock()
      final_instance = MagicMock()
      mock_video.return_value = video_instance
      mock_audio.return_value = audio_instance
      video_instance.with_audio.return_value = final_instance

      temp_video_path = "dummy_video.mp4"
      temp_audio_path = "dummy_audio.wav"
      output_path = str(OUT_DIR / "output.mp4")

      combine_video_audio(temp_video_path, temp_audio_path, output_path)

      video_instance.with_audio.assert_called_with(audio_instance)
      final_instance.write_videofile.assert_called_with(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=30,
        preset='medium',
        bitrate='5000k'
      )
      video_instance.close.assert_called()
      audio_instance.close.assert_called()
      final_instance.close.assert_called()

  def test_draw_text(self):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.audio2mp4 import draw_text

    # テスト用の図と軸の作成
    video_size = (640, 480)
    fig, ax = plt.subplots(figsize=(video_size[0]/100, video_size[1]/100))
    ax.set_facecolor("black")
    ax.set_xlim(0, video_size[0])
    ax.set_ylim(0, video_size[1])

    # テスト用テキストとフォント名
    sample_text = "これはテストです"
    sample_font = "Meiryo"

    # draw_text関数の呼び出し
    text_obj = draw_text(ax, video_size, sample_text, sample_font, 1)

    # テキストオブジェクトの各プロパティの検証
    self.assertIsNotNone(text_obj)
    self.assertEqual(text_obj.get_text(), sample_text)
    self.assertEqual(text_obj.get_alpha(), 1)  # 初期は非表示（alpha=0）になっている
    self.assertEqual(text_obj.get_fontname(), sample_font)

    # 作成した画像を tests/out に保存（削除はしない）
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "draw_text_test.png"
    fig.savefig(str(save_path))

    plt.close(fig)

if __name__ == '__main__':
  unittest.main()