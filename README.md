# Face Extraction using Python pre-built libraries

- Install required packages
```sh
# install dependencies
pip install -r requirements.txt
```
- Used cvlib (which is dlib caffe model from opencv), mtcnn and dsfd pre-trained models built in MTCNN and face_detection libraries
- You can run 

```sh
# install dependencies
python face-extractor.py -i "./data/Benchmark_1.mp4" -o "results/" -m "mtcnn"
```

```text
-i for input video directory
-o for output directory
-m for choosing detector model - dlib, mtcnn and dsfd are available
```
