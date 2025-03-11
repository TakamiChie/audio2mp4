from pathlib import Path
import unittest


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

  def test_draw_text(self):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.draw_text import draw_text

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

  def test_draw_texts(self):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.draw_text import draw_texts

    video_size = (640, 480)
    # 図と軸の作成（背景は黒に設定）
    fig, ax = plt.subplots(figsize=(video_size[0]/100, video_size[1]/100))
    ax.set_facecolor("black")
    ax.set_xlim(0, video_size[0])
    ax.set_ylim(0, video_size[1])

    # テスト用のテキスト辞書（キー：summary, subtitle, title）
    texts = {
      "title": "タイトルテスト",
      "subtitle": "サブタイトルテスト",
      "summary": "概要テスト"
    }

    # draw_texts 呼び出し
    text_objs = draw_texts(fig, ax, video_size, texts, init_alpha=1)

    # キーは summary, subtitle, title の順で描画されるので、リストの順序がそれに一致することを確認
    self.assertEqual(len(text_objs), 3)
    self.assertEqual(text_objs[0].get_text(), "概要テスト")
    self.assertEqual(text_objs[1].get_text(), "サブタイトルテスト")
    self.assertEqual(text_objs[2].get_text(), "タイトルテスト")

    # フォントサイズの検証
    self.assertEqual(text_objs[0].get_fontsize(), 14)
    self.assertEqual(text_objs[1].get_fontsize(), 16)
    self.assertEqual(text_objs[2].get_fontsize(), 20)

    # テスト結果画像を tests/out に保存（削除はしない）
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "draw_texts_test.png"
    fig.savefig(str(save_path))
    plt.close(fig)
