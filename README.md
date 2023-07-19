# Artificial Vision Project

## Project

A series of Python scripts have been programmed to solve artificial vision problems using the OpenCV library.

## Design

Each designed script is commented in a Jupyter notebook file. These files contain design and implementation information, as well as examples and results of applying the scripts.

### Notebooks

| Exercise | Description |
|---:|---|
| Calibration | Uses images of known size to perform image calibration |
| Activity | Detects and evaluates activity zones in images |
| Color | Detects objects using ROI curves |
| Filters | Applies different filters to images |
| Rectif | Rectifies distance measurements using image analysis and distances |
| Pano | Performs image perspective analysis to create a homographic image panorama |
| RA | Generates a 3D object on a surface with known reference points |

### Environment and Installation

This project has been developed using the PyCharm program from the IntelliJ package. The configuration is done by installing the requirements file _requirements.txt_. The necessary libraries for program execution are installed.

> **Requirements**: jupyter notebook, numpy, opencv-python

```shell
# python<version> -m venv <virtual_env_name>
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Sources of Information

[VIA UMU](https://dis.um.es/profesores/alberto/vision.html)

[VIA Repository](https://github.com/albertoruiz/umucv)

Citations:
[1] https://opencv.org
[2] https://opencv.org/about/
[3] https://www.geeksforgeeks.org/opencv-overview/
[4] https://viso.ai/computer-vision/opencv/
[5] https://viso.ai/computer-vision/the-most-popular-computer-vision-tools/
[6] https://www.superannotate.com/blog/computer-vision-libraries
