from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def draw_text_rect(
    fig: Figure,
    ax: Axes,
    video_size: tuple[int, int],
    height: int,
    bgcolor: str,
    alpha: float = 0.5,
    edgecolor: str = "black",
    linewidth: float = 0.0
) -> Rectangle:
    """
    テキスト描画用の矩形を描画する関数。

    Args:
        fig: 図オブジェクト
        ax: 軸オブジェクト
        video_size: 動画のサイズ (width, height)
        height: 矩形の高さ
        bgcolor: 矩形の色
        alpha: 矩形の透明度 (デフォルト: 0.5)
        edgecolor: 矩形の枠線の色 (デフォルト: "black")
        linewidth: 矩形の枠線の太さ (デフォルト: 1.0)

    Returns:
        Rectangle: 描画された矩形オブジェクト
    """
    width = video_size[0]  # 横幅は目一杯
    x = 0 # 左端から開始
    y = video_size[1] - height # 上端を基準に位置決め
    rect = Rectangle(
        (9, y),
        width,
        height,
        facecolor=bgcolor,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False
    )
    ax.add_patch(rect)
    return rect

def draw_texts(fig: Figure, ax: Axes, video_size: tuple[int, int], texts: dict[str, dict[str, str]], init_alpha: float=0, fade_base: float=2.0, textarea_bg_color:str=None) -> tuple[list[plt.Text], int]:
  """
  テキストを描画する。タイトル、サブタイトル、詳細文をそれぞれ描画する
  表示開始は、タイトル: fade_base, サブタイトル: fade_base + 0.1, 詳細文: fade_base + 0.2

  fig: 図オブジェクト
  ax: 軸オブジェクト
  video_size: 動画のサイズ
  texts: タイトル、サブタイトル、詳細文のテキストを格納した辞書
    title: タイトルのテキスト
    subtitle: サブタイトルのテキスト
    summary: 詳細文のテキスト
  init_alpha: テキストの初期透明度（デフォルトは0/テスト時などは1を設定）
  fade_base: フェードイン開始時間（デフォルトは2.0秒）
  textarea_bg_color: テキストエリア全体の背景色
  Returns:
    tuple[list[plt.Text], int]: 描画されたテキストオブジェクトのリストと、最終的な垂直方向のポジション
  """
  text_objs = []
  position = 0
  text_height = 0
  def add_text(text, fontsize=16, bgcolor="darkblue", edgecolor="black", bgalpha=0.5, text_color="white"):
    nonlocal position
    nonlocal text_height
    t_obj = draw_text(ax, video_size, text, init_alpha=init_alpha, text_color=text_color)
    t_obj.set_fontsize(fontsize)
    t_obj.set_bbox(dict(facecolor=bgcolor, edgecolor=edgecolor, alpha=bgalpha))
    t_obj.set_position((10, position + video_size[1] - 30))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_height += t_obj.get_window_extent(renderer).height + 30
    position += t_obj.get_window_extent(renderer).height + 30
    return t_obj

  # 上から順に: title, subtitle, summary
  if "summary" in texts:
    t_obj = add_text(texts["summary"]["text"], 14,
                     texts['summary']["bgcolor"],
                     texts['summary']["edgecolor"], 0.5,
                     texts['summary']["text_color"])
    t_obj.fade_start = fade_base + 0.4    # 詳細文は+0.4秒後
    text_objs.append(t_obj)
  if "subtitle" in texts:
    t_obj = add_text(texts["subtitle"]["text"], 16,
                     texts['subtitle']["bgcolor"],
                     texts['subtitle']["edgecolor"], 0.5,
                     texts['subtitle']["text_color"])
    t_obj.fade_start = fade_base + 0.2    # サブタイトルは+0.2秒後
    text_objs.append(t_obj)
  if "title" in texts:
    t_obj = add_text(texts["title"]["text"], 20,
                     texts['title']["bgcolor"],
                     texts['title']["edgecolor"], 0.5,
                     texts['title']["text_color"])
    t_obj.fade_start = fade_base        # タイトルは基準時刻
    text_objs.append(t_obj)
  if textarea_bg_color:
    # テキストエリアの背景を描画する
    draw_text_rect(fig, ax, video_size, text_height, textarea_bg_color)
  return text_objs, position

def draw_text(ax: Axes, video_size: tuple[int, int], text: str, font_name: str="BIZ UDGothic", init_alpha:float=0, text_color: str="white") -> plt.Text:
  """
  テキストを描画する関数。

  ax: 軸オブジェクト
  video_size: 動画のサイズ
  text: 描画するテキスト
  font_name: フォント名（デフォルトは BIZ UDGothic）
  init_alpha: テキストの初期透明度（デフォルトは0/テスト時などは1を設定）
  text_color: テキストの色（デフォルトは白）
  """
  txt = ax.text(
    10, video_size[1] - 10, text,
    color=text_color,
    fontsize=16,
    verticalalignment="top",
    alpha=init_alpha,
    fontname=font_name,
    bbox=dict(facecolor="black", alpha=0.5)
  )
  return txt
