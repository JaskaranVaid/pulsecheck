# Test fixtures

## face.jpg

- Subject: official NASA portrait of astronaut Michael A. Baker (NASA image S85-41895, taken 1985-10-09).
- Commons file page: https://commons.wikimedia.org/wiki/File:Official_Portrait_-_Astronaut_Michael_A._Baker.jpg
- Original image: https://upload.wikimedia.org/wikipedia/commons/8/80/Official_Portrait_-_Astronaut_Michael_A._Baker.jpg (2400 x 3000 px)
- Source: NASA Image and Video Library, https://images.nasa.gov/details-s85-41895
- Author: NASA
- License: Public domain — work of NASA / US federal government (Commons template `PD-USGov-NASA`, verified on the file page).
- Processing: resized to 640 px wide with OpenCV (`cv2.resize`, `INTER_AREA`, aspect ratio kept), saved as JPEG quality 90.
- Image size: (height, width) = (800, 640)

## landmarks_face.npy

`(478, 2) float32` MediaPipe Face Landmarker landmarks of `face.jpg` in pixel (x, y) coordinates.
Regenerate with `python scripts/make_landmark_fixture.py`.
