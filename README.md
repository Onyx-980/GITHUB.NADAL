# Felicitació de Nadal — Xaranga Animada en Python

Projecte creatiu desenvolupat en **Python** que genera de manera 100% procedimental una animació nadalenca en format vertical (1080x1920, 30 fps) amb música sintetitzada i sincronitzada al ritme d'una xaranga.

## Característiques principals

* **Animació 2D procedimental:** Dibuix i articulació de 9 músics (trompetes, saxos, trombons, helicorn, bombo i caixa) amb física de rebot, parpelleig, moviment d'instruments, neu, confeti i paisatge marítim nocturn.
* **Síntesi d'àudio des de zero:** Generació d'ones sonores amb harmònics, vibrato, percussió, cascavells i reverberació per interpretar una nadala a 168 BPM sense cap fitxer d'àudio extern.
* **Sincronització musical:** Totes les entrades instrumentals, el compte enrere i els moviments dels personatges estan calculats matemàticament a partir del *tempo* i els compassos.
* **Exportació directe a vídeo:** Renderitzat i codificació en format MP4 (H.264 + AAC) preparat per a xarxes socials.

## Tecnologies i llibreries utilitzades

* **Python 3**
* **NumPy & SciPy:** Síntesi digital d'àudio, processament de senyal i convolució de reverberació.
* **OpenCV (`cv2`) & Pillow (`PIL`):** Generació de gràfics vectorials, capes de transparència i tipografia.
* **FFmpeg (`imageio-ffmpeg`):** Codificació i muntatge final de vídeo i àudio.

## Com executar-lo

1. Instal·la les dependències necessàries:
   ```bash
   pip install numpy scipy opencv-python pillow imageio-ffmpeg
   ```
2. Executa l'script principal per generar el vídeo (`xaranga-nadal-ig.mp4`):
   ```bash
   python xaranga_nadal.py
   ```
   *(També admet el paràmetre `test` seguit de segons concrets, com ara `python xaranga_nadal.py test 5.0 10.0 18.0`, per generar una previsualització en imatge `test.jpg`).*
