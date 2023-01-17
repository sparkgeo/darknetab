# Sparkgeo Kaggle Training Data

Subdirectory to train darknet on Kaggle car/swimming pool data.

## Getting Started

This directory has been set up to make it easier to train darknet on the Kaggle dataset.

### Prerequisites

This assumes you have a working compiler on your system.  If you want to use CUDA, you will need to install
gcc-10 in order for nvcc to work.  I would also recommend installing OpenCV via brew or whatever
package manage you use if you are on a Macbook.  Also install OpenMP as that allows the code to 
unroll loops to run things in parallel and what not.

The first thing will be to modify the top of the Makefile where you see these lines:

```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=1
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```

In the above, I have set it up to compile on my desktop that has an nVidia RTX video card.  If you are on
a Macbook, you should set GPU, CUDNN, and CUDNN_HALF to 0.

On your Mac, you can run this command to see if it supports the AVX instruction set:
```
sysctl machdep.cpu | grep AVX
```

If you see output, then you can set AVX in the Makefile to 1, otherwise leave it at zero.  Also set
OpenMP depending on if you installed OpenMP via brew or another package manager on your Mac or Linux.

### Installing

The first thing you will need to do is compile darknet based on the README in the root directory.  If you have 
the pre-requisites installed and have configured the Makefile it should be a simple 

```
make
```

Next will be to downlaod the Kaggle training data.  Go to [aws.sparkgeo.com](https://aws.sparkgeo.com) and click on 
world bank.  From their, click on Management Console and go to the S3 section.  Click on the spk-wb-data bucket and then
the training data object.  Then click on kaggle_archive.zip to download it to your computer.

Inside kaggle_archive.zip you will find a data directory and then test/ and training/ directories with images in them.
Extract this archive to the darknetab/kaggle_yolov4_tiny directory so that a directory listing looks like this:

```
bmaddox@sdf1:~/src/worldbank/darknetab/kaggle_yolov4_tiny$ dir
total 19356
drwxrwxr-x 4 bmaddox bmaddox     4096 Jan  5 15:34 data
-rw-rw-r-- 1 bmaddox bmaddox     3558 Jan 17 11:55 kaggle-readme.md
-rw-rw-r-- 1 bmaddox bmaddox     3154 Jan  5 15:18 kaggle-yolov4-tiny-custom.cfg
-rw-rw-r-- 1 bmaddox bmaddox      166 Jan  5 15:35 obj.data
-rw-rw-r-- 1 bmaddox bmaddox       17 Jan  5 15:19 obj.names
-rw-rw-r-- 1 bmaddox bmaddox      802 Jan  5 15:33 process.py
-rw-rw-r-- 1 bmaddox bmaddox 19789716 Dec  8  2021 yolov4-tiny.conv.29
```

Now inside this directory make a backup directory so that it now looks like this:

```
bmaddox@sdf1:~/src/worldbank/darknetab/kaggle_yolov4_tiny$ dir
total 19356
drwxrwxr-x 2 bmaddox bmaddox     4096 Jan  5 16:27 backup
drwxrwxr-x 4 bmaddox bmaddox     4096 Jan  5 15:34 data
-rw-rw-r-- 1 bmaddox bmaddox     3558 Jan 17 11:55 kaggle-readme.md
-rw-rw-r-- 1 bmaddox bmaddox     3154 Jan  5 15:18 kaggle-yolov4-tiny-custom.cfg
-rw-rw-r-- 1 bmaddox bmaddox      166 Jan  5 15:35 obj.data
-rw-rw-r-- 1 bmaddox bmaddox       17 Jan  5 15:19 obj.names
-rw-rw-r-- 1 bmaddox bmaddox      802 Jan  5 15:33 process.py
-rw-rw-r-- 1 bmaddox bmaddox 19789716 Dec  8  2021 yolov4-tiny.conv.29
```

Next you will need to generate a train.txt and a test.txt file inside the data directory.  These tell darknet what images
to use for training and testing.  To do this, open up a terminal window inside the kaggle_yolov4_tiny directory and then run
the command process.py like this:

```commandline
python3 process.py
```

If it runs successfully, you should now have a train.text and test.txt files inside the data directory.  The top few lines of
each file should look like this:

```
kaggle_yolov4_tiny/data/train/000000749.jpg
kaggle_yolov4_tiny/data/train/000003512.jpg
kaggle_yolov4_tiny/data/train/000002623.jpg
kaggle_yolov4_tiny/data/train/000001695.jpg
```

Note the numbers will likely be different on yours as this is randomized.

## Training

Now you should be ready to train the model.  To do this, go to the main darknetab direcotry and run:

```commandline
./darknet detector train kaggle_yolov4_tiny/obj.data kaggle_yolov4_tiny/kaggle-yolov4-tiny-custom.cfg kaggle_yolov4_tiny/yolov4-tiny.conv.29 -dont_show -map
```

This will take a while and you will see a lot of things fly by on the screen.  When it's done, you should have files like this in your kaggle_yolov4_tiny/backup
directory:

```commandline
bmaddox@sdf1:~/src/worldbank/darknetab/kaggle_yolov4_tiny$ dir backup/
total 206820
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 15:46 kaggle-yolov4-tiny-custom_1000.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 15:54 kaggle-yolov4-tiny-custom_2000.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:02 kaggle-yolov4-tiny-custom_3000.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:11 kaggle-yolov4-tiny-custom_4000.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:19 kaggle-yolov4-tiny-custom_5000.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:27 kaggle-yolov4-tiny-custom_6000.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:26 kaggle-yolov4-tiny-custom_best.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:27 kaggle-yolov4-tiny-custom_final.weights
-rw-rw-r-- 1 bmaddox bmaddox 23530556 Jan  5 16:27 kaggle-yolov4-tiny-custom_last.weights
```

The main file you will be interested in here is kaggle-yolov4-tiny-custom_final.weights.

## Testing

To test your new model, there are files inside the darknetab/test_images directory.  

From the darknetab directory, you would run a test like this:

```commandline
./darknet detector test kaggle_yolov4_tiny/obj.data kaggle_yolov4_tiny/kaggle-yolov4-tiny-custom.cfg kaggle_yolov4_tiny/backup/kaggle-yolov4-tiny-custom_final.weights test_images/test3.png
```

This will pop up an output window showing the detections as well as create a predictions.jpg file in the main darknetab/ directory.