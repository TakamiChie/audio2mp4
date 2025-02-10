# Audio2MP4

## 概要
MP3音声ファイルからオーディオビジュアライザー付きのMP4動画を生成するPythonプログラムです。
音声の波形を視覚的に表示しながら動画化し、背景画像やフェードエフェクトの適用も可能できます。

実行例は高見知英のYouTube個人チャンネルなどをみるとあるかもしれない。
[個人YouTube](https://www.youtube.com/@TakamiChie)
---

## 1. ユーザー向け使用方法

### 必要環境
- Python 3.x
- `pipenv` による環境管理

### インストール

#### 1. `pipenv` のインストール（未導入の場合）
```sh
pip install pipenv
```

#### 2. 仮想環境の作成と必要ライブラリのインストール
以下のコマンドを実行し、必要なPythonライブラリをインストールしてください：
```sh
pipenv install
```

### 仮想環境の有効化
```sh
pipenv shell
```

### 実行方法
基本的な実行例：
```sh
python audio2mp4.py input.mp3
```
このコマンドは `input.mp3` のビジュアライザー付き動画 `audio.mp4` を作成します。

実行後しばらくの間何も表示されませんので、気長にしばらくお待ちください。

#### オプション
| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--output` | 出力MP4ファイル名 | `audio.mp4` |
| `--video-width` | 動画の幅 (px) | `1280` |
| `--video-height` | 動画の高さ (px) | `720` |
| `--viz-width` | ビジュアライザーの幅 (px) | `1080` |
| `--viz-height` | ビジュアライザーの高さ (px) | `520` |
| `--bg-color` | 背景色 (`black`, `white` など) | `black` |
| `--viz-color` | ビジュアライザーの色 (`gradation` で青→赤グラデーション) | `gradation` |
| `--loop-count` | 音声のループ回数 | `1` |
| `--bg-image` | 背景画像パス | なし |
| `--effect` | `fade` を指定するとフェードアウト効果 | なし |

#### 実行例
1. 赤色のビジュアライザーで出力：
   ```sh
   python audio2mp4.py input.mp3 --viz-color red
   ```
2. 背景画像を適用：
   ```sh
   python audio2mp4.py input.mp3 --bg-image background.jpg
   ```
3. 3回ループ再生：
   ```sh
   python audio2mp4.py input.mp3 --loop-count 3
   ```
4. フェードアウト効果付き：
   ```sh
   python audio2mp4.py input.mp3 --effect fade
   ```

---

## 2. 補足情報
- `moviepy` の処理速度を向上させるため、`ffmpeg` のインストールを推奨します。
- `librosa` を使用して音声波形を解析し、`matplotlib` でアニメーション化しています。

このスクリプトを活用し、オーディオビジュアライザー付きの動画を作成してみてください！

### AIでどこまでいけるか試す試み

とりあえずAIでどこまで行けるか試しながらやってみる試み実行中。

会話ログについてはなるべく以下のAI問答集にまとめています。

- [AI問答集トップページ](https://takamichie.notion.site/67c7609855084fd186f9e059ab70f327?v=12aa62c17a4640b0b4e087eea283d7ab&pvs=4)
  - [基本コード](https://takamichie.notion.site/Audio-Visualizer-with-Customizable-MP4-Video-Generation-19460d1e6e79804ca6ecc040546d1f9a?pvs=4)
  - [README](https://takamichie.notion.site/19660d1e6e7980d0939ede83107f3fed?pvs=4)

以降についてはGitHub Copilotで補足しているのでここに書いてない場合があります。