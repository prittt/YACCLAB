## YACCLAB: Yet Another Connected Components Labeling Benchmark

### Introduction
<p align="justify"> 
YACCLAB is an open source C++ project which runs and tests CCL algorithms on a collection of datasets described below. Beside running a CCL algorithm and testing its correctness, YACCLAB performs three more kinds of test: average run-time test, density test and size test, in which the performance of the algorithms are evaluated with images of increasing density and size.
<br/>
To check the correctness of an implementation, the output of an algorithm is compared with that of the OpenCV 'cv::connectedComponents function' available from 3.0 release. Notice that 8-connectivity is always used. A colorized version of the input images can also be produced,to visually check the output and investigate possible labeling errors.
</p>

=======

### Datasets
 
<p align="justify">YACCLAB dataset includes both synthetic and real images. All images are provided in 1 bit per pixel PNG format, with 0 (black) being background and 1 (white) being foreground. Images are organized by folders and as follows: </p>

- <p align="justify"><b>test_random:</b> a set of synthetic random noise images who contain black and white random noise with 9 different foreground densities (10\% up to 90\%), from a low resolution of 32x32 pixels to a maximum resolution of 4096x4096 pixels, allowing to test the scalability and the effectiveness of different approaches when the number of labels gets high. For every combination of size and density, 10 images are provided for a total of 720 images. The resulting subset allows to evaluate performance both in terms of scalability on the number of pixels and on the number of labels (density).</p>

- <p align="justify"><b>mirflickr:</b> Otsu-binarized version of the MIRflickr dataset, publicly available under a Creative Commons License. It contains 25,000 standard resolution images taken from Flickr. These images have an average resolution of 0.17 megapixels, there are few connected components (495 on average) and are generally composed of not too complex patterns, so the labeling is quite easy and fast.</p>

- <p align="justify"><b>hamlet:</b> a set of 104 images scanned from a version of the Hamlet found on Project Gutenberg (URL http://www.gutenberg.org).</p>

- <p align="justify"><b>tobacco800:</b> a set of 1290 document images. It is a realistic database for document image analysis research as these documents were collected and scanned using a wide variety of equipment over time. Resolutions of documents in Tobacco800 vary significantly from 150 to 300 DPI and the dimensions of images range from 1200 by 1600 to 2500 by 3200 pixels. Since CCL is one of the initial preprocessing steps in most layout analysis or OCR algorithms, hamlet and tobacco800 allow to test the algorithm performance in such scenarios. </p>

- <p align="justify"><b>3dpes:</b> it comes from 3DPeS (3D People Surveillance Dataset), a surveillance dataset designed mainly for people re-identification in multi camera systems with non-overlapped fields of view. 3DPeS can be also exploited to test many other tasks, such as people detection, tracking, action analysis and trajectory analysis. The background models for all cameras are provided, so a very basic technique of motion segmentation has been applied to generate the foreground binary masks, i.e.,  background subtraction and fixed thresholding. The analysis of the foreground masks to remove small connected components and for nearest neighbor matching is a common application for CCL. </p>
=======
###Tests

Average run-time tests execute an algorithm on every image of a dataset. The process can be repeated more times in a single test, to get the minimum execution time for each image: this allows to get more reproducible results and overlook delays produced by other running processes. It is also possible to compare the execution
speed of different algorithms on the same dataset: in this case, selected algorithms are executed sequentially on every image of the dataset. Results are presented in
three different formats: a plain text file, histogram charts, either in color or in gray-scale, and a LaTeX table, which can be directly included in research papers.

Finally, density and size tests check the performance of different CCL algorithms when they are executed on images with varying foreground density and size.
To this aim, a list of algorithms selected by the user is run sequentially on every image of the test\_random dataset. As for run-time tests, it is possible to repeat this test for more than one run. The output is presented as both plain text and charts. For a density test, the mean execution time of each algorithm is reported for densities ranging from 10\% up to 90\%, while for a size test the same is reported for resolutions ranging from $32\times 32$ up to $4096 \times 4096$. A showcase will be presented in Section~\ref{sec:results}.

A configuration file placed in the installation folder lets the user specify which kind of test should be performed, on which datasets and on which algorithms. 
A complete description of all configuration parameters is reported in Table~\ref{tab:flags}.


###Requirements

To correctly install and run YACCLAB following packages, libraries and utility are needed: <br />

- CMake 2.4.0 or higher (https://cmake.org/download/),
- OpenCV 3.0 or higher (http://opencv.org/downloads.html),
- Gnuplot (http://www.gnuplot.info/). 

Note: you must add gnuplot to system path if you want YACCLAN  

=======

###Installation

- Clone the GitHub repository (HTTPS clone URL: https://github.com/prittt/YACCLAB.git) or simply download the full master branch zip file.
- With CMake 

=======

###Configuration File

A configuration file placed in the installation folder lets you to specify which kind of test should be performed, on which datasets and on which
algorithms. A complete description of all configuration parameters is reported in the table below.

#####Parameter name Description
<table>
<tr><td>input path</td><td>folder on which datasets are placed</td></tr>
<tr><td>output path</td><td>folder on which result are stored</td></tr>
<tr><td>write n labels</td><td>whether to report the number of Connected Components in output files</td></tr>
</table>

#####Correctness tests
<table>
<tr><td>check 8connectivity</td><td>whether to perform correctness tests</td></tr>
<tr><td>check list</td><td>list of datasets on which CCL algorithms should be checked</td></tr>
</table>

#####Density and size tests
<table>
<tr><td>ds perform</td><td>whether to perform density and size tests</td></tr>
<tr><td>ds colorLabels</td><td>whether to output a colorized version of input images</td></tr>
<tr><td>ds testsNumber</td><td>number of runs</td></tr>
<tr><td>ds saveMiddleTests</td><td>whether to save the output of single runs, or only a summary of the whole test</td></tr>
</table>

#####Average execution time tests
<table>
<tr><td>at perform</td><td>whether to perform average execution time tests</td></tr>
<tr><td>at colorLabels</td><td>whether to output a colorized version of input images</td></tr>
<tr><td>at testsNumber</td><td>number of runs</td></tr>
<tr><td>at saveMiddleTests</td><td>whether to save the output of single runs, or only a summary of the whole test</td></tr>
<tr><td>averages tests</td><td>list of algorithms on which average execution time tests should be run</td></tr>
</table>

#####Algorithm configuration
<table>
<tr><td>CCLAlgorithmsFunc</td><td>list of available algorithms (function names)</td></tr>
<tr><td>CCLAlgorithmsName</td><td>list of available algorithms (display names for charts)</td></tr>
</table>

=======

###How Add New Algorithms

YACCLAB has been designed with extensibility in mind, so that new resources can be easily integrated into the project. A CCL algorithm is coded with
a .h header file, which declares a function implementing the algorithm, and a .cpp source file which defines the function itself. The function must
follow a standard signature: its first parameter should be a const reference to an OpenCV Mat1b matrix, containing the input image, and its second 
parameter should be a reference to a Mat1i matrix, which shall be populated with computed labels. The function must also return an integer specifying
the

=======
