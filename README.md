# TOMATO-TOMATO

**you get tomato'd when you thumbs down !!**

a small project to learn and play with opencv and mediapipe. unfortunately not well organized/designed as priority was function. i would fix the tomato hitbox + sizing, but i'm learning graphics right now.

`b` to toggle hitboxes
`q` to quit

models are from mediapipe

tomato is from green-screen video, used ffmpeg to convert to transparent (mostly) pngs

`ffmpeg -i tomato.mp4 -vf "chromakey=#08f508:0.3:0.5" frames/%02d.png`

## unaddressed issues

- tomato glitches if immediatly triggered due to async update of facial coordinates

## .env

`PROJECT_DIR=<path_to_prjct_root>`

## dependencies:

- main things to note:
    - python=3.10
    - `pip install opencv-python mediapipe`
- see `environment.yml` for anything else

## future plans:

- add rps --> tomato'd when you lose rps (challenge with thumbs down)
- rewrite in c++
- learn graphics
- replace tomato with rendered tomato
- maybe also add graphics for rps
