# Low-frame-reconstruction

Low frame reconstruction is an offline reconstruction system based on RGB-D input data. The whole system is constituted by three main components:
feature detection, feature correspondence and a novel global optimization.

After optimization, you will get a precise camera trajectory even though it is faced with very low frame rate situation. Our experiment demostrate our method achieved an impressive result and more robust than state-of-the-art SLAM and reconstruction system. 

If you want to get a triangle mesh model, you can use TSDF to integrate all of pointclouds and then use marching cube to meshing. For TSDF we provide source code and execute program. For marching cube we only provide execute program, it is a CUDA program which comes from PCL.  

# 1. Prerequisites

We have tested the libraries in Windows10 x64 operation system and Microsoft Visual Studio 2013. And it should be easy to compile in other OS like linux, Mac OS etc. We include all of dependencies in 3rdParty file folder. You can easily install them. 

The following install tutorials which we provide are only tested in Windows10 X64 OS and Microsoft Visual Studio 2013.

OpenCV2.7.13
--------------------
1. Install 3rdParty/opencv-2.4.13.exe

2. Modify your machine's environment variable and add X:\opencv2.4.13\opencv\build\x86\vc12\bin and X:\opencv2.4.13\opencv\build\x64\vc12\bin

3. Modify 


