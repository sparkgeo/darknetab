# Sparkgeo SIMD Training Data

Subdirectory to train darknet on SIMD vehicles data.

## Getting Started

This directory has been set up to make it easier to train darknet on the SIMD dataset.

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

Next will be to downlaod the SIMD training data.  Go to [aws.sparkgeo.com](https://aws.sparkgeo.com) and click on 
world bank.  From their, click on Management Console and go to the S3 section.  Click on the spk-wb-data bucket and then
the training data object.  Then click on go into the SIMD directory and click on training.gzip.tar and validation.gzip.tar
to download them to your computer.

Extract these archives to the darknetab/simd directory so that a directory listing looks like this:

```
bmaddox@sdf1:~/src/worldbank/darknetab/simd$ dir full-dataset/
total 616
drwx------ 5 bmaddox bmaddox   4096 Jan 16 12:17 annotations
-rw-rw-r-- 1 bmaddox bmaddox  73999 Jan 16 12:21 testing.txt
drwxrwxr-x 2 bmaddox bmaddox 188416 Apr 25  2020 training
-rw-rw-r-- 1 bmaddox bmaddox 242350 Jan 16 12:22 training.txt
drwxrwxr-x 2 bmaddox bmaddox  53248 Apr 25  2020 validation
-rw-rw-r-- 1 bmaddox bmaddox  55100 Jan 16 12:22 validation.txt
```

Note that the annotations directory is optional.

You should also copy all of the files from validation into the training directory.  The mine.testing.data
file specifies which files are used for training, testing, and validation.  However, there is a small bug
in the SIMD repo where files in testing.txt are in both training and the validation directories.  Copying
them all to the training directory is just easier.

Now inside the darknetab/simd directory, make a backup directory so that it now looks like this:

```
bmaddox@sdf1:~/src/worldbank/darknetab/simd$ dir
total 8496
drwxrwxr-x 2 bmaddox bmaddox    4096 Jan 17 14:17 backup
drwxrwxr-x 2 bmaddox bmaddox    4096 Jan 17 13:18 cfg
drwxrwxr-x 5 bmaddox bmaddox    4096 Jan 16 12:18 full-dataset
drwxrwxr-x 2 bmaddox bmaddox    4096 Jan  3 13:30 images
-rw-rw-r-- 1 bmaddox bmaddox      87 Jan  3 13:30 map_init-4l.sh
-rw-rw-r-- 1 bmaddox bmaddox      94 Jan  3 13:30 map_proposed.sh
-rw-rw-r-- 1 bmaddox bmaddox      91 Jan  3 13:30 map_tiny-3l.sh
-rw-rw-r-- 1 bmaddox bmaddox      90 Jan  3 13:30 map_yolov2.sh
-rw-rw-r-- 1 bmaddox bmaddox      86 Jan  3 13:30 map_yolov3.sh
-rw-rw-r-- 1 bmaddox bmaddox      86 Jan  3 13:30 map_yolt.sh
-rw-rw-r-- 1 bmaddox bmaddox    1613 Jan  3 13:30 README.md
drwxrwxr-x 2 bmaddox bmaddox    4096 Jan  3 13:30 simd
-rw-rw-r-- 1 bmaddox bmaddox     121 Jan  3 13:30 simd-classes.txt
-rw-rw-r-- 1 bmaddox bmaddox    5880 Jan 20 11:59 simd-readme.md
-rw-rw-r-- 1 bmaddox bmaddox 8627374 Jan  3 13:30 simd-sample-images.pdf
-rw-rw-r-- 1 bmaddox bmaddox     899 Jan  3 13:30 training-mini.txt.txt
-rw-rw-r-- 1 bmaddox bmaddox     359 Jan  3 13:30 validation-mini.txt.txt
```

## Training

Now you should be ready to train the model.  To do this, go to the main darknetab direcotry and run:

```commandline
./darknet detector train simd/cfg/mine.testing.data simd/cfg/kaggle-yolov4-tiny-custom.cfg -dont_show -map
```

In this case we are using the kaggle yolov4 tiny config file.  This will take a while and you will see a 
lot of things fly by on the screen.  When it's done, you should have files like this in your simd/backup
directory:

```commandline
bmaddox@sdf1:~/src/worldbank/darknetab/simd$ dir backup/
total 1364732
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 13:27 kaggle-yolov4-tiny-custom_1000.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 13:37 kaggle-yolov4-tiny-custom_2000.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 13:47 kaggle-yolov4-tiny-custom_3000.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 13:57 kaggle-yolov4-tiny-custom_4000.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 14:07 kaggle-yolov4-tiny-custom_5000.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 14:17 kaggle-yolov4-tiny-custom_6000.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 14:17 kaggle-yolov4-tiny-custom_final.weights
-rw-rw-r-- 1 bmaddox bmaddox  23650676 Jan 17 14:17 kaggle-yolov4-tiny-custom_last.weights
```

The main file you will be interested in here is kaggle-yolov4-tiny-custom_best.weights.

## Testing

To test your new model, there are files inside the darknetab/test_images directory.  

From the darknetab directory, you would run a test like this:

```commandline
./darknet detector test simd/cfg/mine.testing.data simd/cfg/kaggle-yolov4-tiny-custom.cfg simd/backup/kaggle-yolov4-tiny-custom_best.weights test_images/test3.png```
```
This will pop up an output window showing the detections as well as create a predictions.jpg file in the main darknetab/ directory.