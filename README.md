Completed Volume Local Binary Count Pattern (cuda version)
=================================

Viewing a gray-scale video as a 3D volume, its VLBC and CVLBC features are extracted for dynamic texture recognition or 2D face spoofing detection. For more information, please refer to the paper http://ieeexplore.ieee.org/abstract/document/8030131/.

Complilation
============

This project is compiled with g++ and cuda-toolbox-8.0.


Usage
=====

In the file cvlbc.cu, an example of extracting features from the videos in the Print-Attack is presented. Firstly, the frames of a given gray-scale video are converted to gray-scale pgm files and are named consecutively, e.g., 1.pgm, 2.pgm.... Then, these pgm images are loaded one-by-one, resulting in a 2D array. Finally, features are extracted from the array.


Reference
=========

1, The portable graymap format (PGM) parser (pgm.h, pgm.cpp) is from Andrea Vedaldi.

2, If you use this code in your work, please cite the following paper in your publication:

	@article{8030131, 
		author={Xiaochao Zhao and Yaping Lin and Janne Heikkilä}, 
		journal={IEEE Transactions on Multimedia}, 
		title={Dynamic Texture Recognition Using Volume Local Binary Count Patterns with an Application to 2D Face Spoofing Detection}, 
		year={2017}, 
		doi={10.1109/TMM.2017.2750415}, 
		ISSN={1520-9210}
	}
