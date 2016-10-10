## YACCLAB: Yet Another Connected Components Labeling Benchmark

### Introduction
<p align="justify"> 
YACCLAB is an open source C++ project which runs and tests CCL algorithms on a collection of datasets described below. Beside running a CCL algorithm and testing its correctness, YACCLAB performs four more kinds of test: average run-time test, density and size test in which the performance of the algorithms are evaluated with images of increasing density and size, and memory test (see <a href="#tests">Tests</a> section for more details).
<br/><br/>
To check the correctness of an implementation, the output of an algorithm is compared with that of the Scan Array Union Find Algorithm<sup><a href="#SAUF">6</a></sup>. Notice that 8-connectivity is always used. A colorized version of the input images can also be produced, to visually check the output and investigate possible labeling errors. A more detailed description of the project/benchmark can be found in <a href="#YACCLAB">[17]</a>.
</p>

=======

### Datasets
 
<p align="justify">YACCLAB dataset includes both synthetic and real images. All images are provided in 1 bit per pixel PNG format, with 0 (black) being background and 1 (white) being foreground. The dataset will be automatically downloaded by CMake during the installation process. Images are organized by folders as follows: </p>

- <b>Random:<sup><a href="#BBDT">4</a></sup></b><p align="justify"> A set of synthetic random noise images who contain black and white random noise with 9 different foreground densities (10% up to 90%), from a low resolution of 32x32 pixels to a maximum resolution of 4096x4096 pixels, allowing to test the scalability and the effectiveness of different approaches when the number of labels gets high. For every combination of size and density, 10 images are provided for a total of 720 images. The resulting subset allows to evaluate performance both in terms of scalability on the number of pixels and on the number of labels (density). </p>

- <b>MIRflickr:<sup><a href="#MIRFLICKR">10</a></sup></b><p align="justify"> Otsu-binarized version of the MIRflickr dataset, publicly available under a Creative Commons License. It contains 25,000 standard resolution images taken from Flickr. These images have an average resolution of 0.17 megapixels, there are few connected components (495 on average) and are generally composed of not too complex patterns, so the labeling is quite easy and fast.</p>

- <b>Hamlet:</b><p align="justify"> A set of 104 images scanned from a version of the Hamlet found on Project Gutenberg (http://www.gutenberg.org). Images have an average amount of 2.71 million of pixels to analyze and 1447 components to label, with an average foreground density of 0.0789. </p>

- <b>Tobacco800:<sup><a href="#TOBACCO1">11</a>,<a href="#TOBACCO2">12</a>,<a href="#TOBACCO3">13</a></sup></b><p align="justify"> A set of 1290 document images. It is a realistic database for document image analysis research as these documents were collected and scanned using a wide variety of equipment over time. Resolutions of documents in Tobacco800 vary significantly from 150 to 300 DPI and the dimensions of images range from 1200 by 1600 to 2500 by 3200 pixels. Since CCL is one of the initial preprocessing steps in most layout analysis or OCR algorithms, hamlet and tobacco800 allow to test the algorithm performance in such scenarios. </p>

- <b>3DPeS:<sup><a href="#3DPES">14</a></sup></b> <p align="justify"> It comes from 3DPeS (3D People Surveillance Dataset), a surveillance dataset designed mainly for people re-identification in multi camera systems with non-overlapped fields of view. 3DPeS can be also exploited to test many other tasks, such as people detection, tracking, action analysis and trajectory analysis. The background models for all cameras are provided, so a very basic technique of motion segmentation has been applied to generate the foreground binary masks, i.e.,  background subtraction and fixed thresholding. The analysis of the foreground masks to remove small connected components and for nearest neighbor matching is a common application for CCL. </p>

- <b>Medical:<sup><a href="#MEDICAL">15</a></sup></b><p align="justify"> This dataset is composed by histological images and allow us to cover this fundamental medical field. The process used for nuclei segmentation and binarization is described in  <a href="#MEDICAL">[12]</a>. The resulting dataset is a collection of 343 binary histological images with an average amount of 1.21 million of pixels to analyze and 484 components to label. </p> 

- <b>Fingerprints:<sup><a href="#FINGERPRINTS">16</a></sup></b><p align="justify"> This dataset counts 960 fingerprint images collected by using low-cost optical sensors or synthetically generated. These images were taken from the three Verification Competitions FCV2000, FCV2002 and FCV2004. In order to fit CCL application, fingerprints have been binarized using an adaptive threshold and then negated in order to have foreground pixel with value 255. Most of the original images have a resolution of 500 DPI and their dimensions range from 240 by 320 up to 640 by 480 pixels. </p> 

=======
<a name="tests"></a>
###Tests

- <b>Average run-time tests:</b> <p align="justify"> execute an algorithm on every image of a dataset. The process can be repeated more times in a single test, to get the minimum execution time for each image: this allows to get more reproducible results and overlook delays produced by other running processes. It is also possible to compare the execution speed of different algorithms on the same dataset: in this case, selected algorithms (see <a href="#conf">Configuration File</a> for more details) are executed sequentially on every image of the dataset. Results are presented in three different formats: a plain text file, histogram charts (.pdf/.ps), either in color or in gray-scale, and a LaTeX table, which can be directly included in research papers.</p>

- <b>Average density and size tests:</b> <p align="justify"> check the performance of different CCL algorithms when they are executed on images with varying foreground density and size. To this aim, a list of algorithms selected by the user is run sequentially on every image of the test_random dataset. As for run-time tests, it is possible to repeat this test for more than one run. The output is presented as both plain text and charts(.pdf/.ps). For a density test, the mean execution time of each algorithm is reported for densities ranging from 10% up to 90%, while for a size test the same is reported for resolutions ranging from 32x32 up to 4096x4096.</p>

- <b>Memory tests:</b> <p align="justify"> are useful to understand the reason for the good performances of an algorithm or in general to explain its behavior. Memory tests compute the average number of accesses to the label image (i.e the image used to store the provisional and then the final labels for the connected components), the average number of accesses to the binary image to be labeled, and, finally, the average number of accesses to data structures used to solve the equivalences between label classes. Moreover, if an algorithm requires extra data, memory tests summarize them as ``other'' accesses and return the average. Furthermore, all average contributions of an algorithm and dataset are summed together in order to show the total amount of memory accesses. Since counting the number of memory accesses imposes additional computations, functions implementing memory access tests are different from those implementing run-time and density tests, to keep run-time tests as objective as possible.</p>


=======

###Requirements

<p align="justify">To correctly install and run YACCLAB following packages, libraries and utility are needed:</p>

- CMake 2.4.0 or higher (https://cmake.org/download/),
- OpenCV 3.0 or higher (http://opencv.org/downloads.html),
- Gnuplot (http://www.gnuplot.info/), 
- One of your favourite IDE/compiler: Visual Studio 2013 or higher, Xcode 5.0.1, gcc 4.7 or higher, .. (with C++11 support)

Note for gnuplot:
- on Windows system: be sure add gnuplot to system path if you want YACCLAB automatically generates charts.
- on MacOS system: 'pdf terminal' seems to be not available due to old version of cairo, 'postscript' one is used.

=======

###Installation

- <p align="justify">Clone the GitHub repository (HTTPS clone URL: https://github.com/prittt/YACCLAB.git) or simply download the full master branch zip file and extract it (e.g YACCLAB folder).</p>
- <p align="justify">Install software in YACCLAB/build subfolder (suggested) or wherever you want using CMake. Note that CMake should automatically find OpenCV path (if installed), download YACCLAB Dataset and create a project for the selected IDE/compiler.</p>
- <p align="justify">Set <a href="#conf">configuration file</a> in order to execute desired tests, open the project created at the previous point, compile and run it: the work is done. </p>

=======
<a name="conf"></a>
###Configuration File

<p align="justify">A configuration file placed in the installation folder lets you to specify which kind of test should be performed, on which datasets and on which algorithms. A complete description of all configuration parameters is reported in the table below.</p>

#####Parameter name Description
<table>
<tr><td>input_path</td><td>folder on which datasets are placed</td></tr>
<tr><td>output_path</td><td>folder on which result are stored</td></tr>
<tr><td>write_n_labels</td><td>whether to report the number of Connected Components in output files</td></tr>
</table>

#####Correctness tests
<table>
<tr><td>check_8connectivity</td><td>whether to perform correctness tests</td></tr>
<tr><td>check_list</td><td>list of datasets on which CCL algorithms should be checked</td></tr>
</table>

#####Density and size tests
<table>
<tr><td>ds_perform</td><td>whether to perform density and size tests</td></tr>
<tr><td>ds_colorLabels</td><td>whether to output a colorized version of input images</td></tr>
<tr><td>ds_testsNumber</td><td>number of runs</td></tr>
<tr><td>ds_saveMiddleTests</td><td>whether to save the output of single runs, or only a summary of the whole test</td></tr>
</table>

#####Average execution time tests
<table>
<tr><td>at_perform</td><td>whether to perform average execution time tests</td></tr>
<tr><td>at_colorLabels</td><td>whether to output a colorized version of input images</td></tr>
<tr><td>at_testsNumber</td><td>number of runs</td></tr>
<tr><td>at_saveMiddleTests</td><td>whether to save the output of single runs, or only a summary of the whole test</td></tr>
<tr><td>averages tests</td><td>list of algorithms on which average execution time tests should be run</td></tr>
</table>

#####Memory tests
<table>
<tr><td>mt_perform</td><td>whether to perform memory tests</td></tr>
<tr><td>memory_tests</td><td>list of datasets on which memory tests should be run</td></tr>
</table>

#####Algorithm configuration
<table>
<tr><td>CCLAlgoFunc</td><td>list of algorithms (function names) on which apply correctness/average/density-size tests</td></tr>
<tr><td>CCLAlgoName</td><td>list of algorithms (display names for charts) on which apply correctness/average/density-size tests</td></tr>
<tr><td> CCLMemAlgoFunc</td><td>list of available algorithms (function names) on which apply memory tests</td></tr>
<tr><td>CCLMemAlgoName</td><td>list of available algorithms (display names for charts) on which apply memory tests</td></tr>
</table>

=======

###How Add New Algorithms to YACCLAB

<p align="justify">YACCLAB has been designed with extensibility in mind, so that new resources can be easily integrated into the project. A CCL algorithm is coded with a <tt>.h</tt> header file, which declares a function implementing the algorithm, and a <tt>.cpp</tt> source file which defines the function itself. The function must follow a standard signature: its first parameter should be a const reference to an OpenCV Mat1b matrix, containing the input image, and its second parameter should be a reference to a Mat1i matrix, which shall be populated with computed labels. The function must also return an integer specifying the number of labels found in the image, included background's one. For example:</p>
```c++
int MyLabelingAlgorithm(const cv::Mat1b& img,cv::Mat1i &imgLabels);
```
<p align="justify">Making YACCLAB aware of a new algorithm requires two more steps. The new header file has to be included in <tt>labelingAlgorithms.h</tt>, which is in charge of collecting all the algorithms in YACCLAB. This file also defines a C++ map with function pointers to all implemented algorithms, which has also to be updated with the new function.</p>

<p align="justify">Once an algorithm has been added to YACCLAB, it is ready to be tested and compared to the others. To include the newly added algorithm in a test, it is sufficient to include its function name in the <tt>CCLAlgorithmsFunc</tt> <a href="#conf">parameter</a> and a display name in the <tt>CCLAlgorithmsName</tt> parameter. We look at YACCLAB as a growing effort towards better reproducibility of CCL algorithms, so implementations of new and existing labeling methods are welcome.</p>

=======

###Results ...

In this section we use  acronyms  to  refer  to  the  available  algorithms:  
- CT  is  the  Contour  Tracing  approach  by Fu  Chang et al.<sup>[1](#CT)</sup>; 
- CCIT  is  the  algorithm  by  Wan-Yu Chang et al. <sup>[2](#CCIT)</sup>; 
- DiStefano is the algorithm in <sup>[3](#DiStefano)</sup>; 
- BBDT is the  Block  Based  with  Decision  Trees  algorithm  by  Grana et al. <sup>[4](#BBDT)</sup>; 
- LSL STD  is  the  Light  Speed  Labeling  algorithm  by Lacassagne et al. <sup>[5](#LSL_STD)</sup>; 
- SAUF  is  the  Scan  Array  Union  Find algorithm by Wu et al. <sup>[6](#SAUF)</sup>; 
- CTB is the Configuration-Transition-Based algorithm by He et al. <sup>[7](#CTB)</sup>;  
- SBLA is the stripe-based algorithm by Zhao et al.<sup>[8](#SBLA)</sup>; 
- PRED is the Optimized Pixel Prediction by Grana et al. <sup>[9](#PRED)</sup>;
- NULL labeling is an algorithms that defines a lower bound limit for the execution time of CCL algorithms on a given machine and dataset. As the name suggests, this algorithm does not provide the correct connected components for a given image. It only checks the pixels of that image and sets almost randomly the value of the output. 

SAUF and BBDT are the algorithms currently included in OpenCV. 

####... on 04/21/2016

<p align="justify">To  make  a  first  performance  comparison  and  to  showcase automatically  generated  charts  we  have  run  each algorithm in YACCLAB on all datasets and in three different environments:  a  Windows  PC  with  a  i7-4790  CPU  @  3.60 GHz and Microsoft Visual Studio 2013, a Linux workstation with a Xeon CPU E5-2609 v2 @ 2.50GHz and GCC 5.2, and a Intel Core Duo @ 2.8 GHz running OS~X with XCode 7.2.1. Average run-time tests, as well as density and size tests, were repeated 10 times, and for each image the minimum execution time was considered.</p>

<table border="0">
<caption><h4>Average run-time tests on a i7-4790 CPU @ 3.60 GHz with Windows and Microsof Visual Studio 2013</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Hamlet</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/ilb14_averages_3dpes.png" alt="ilb14_3dpes" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/ilb14_averages_hamlet.png" alt="ilb14_hamlet" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/ilb14_averages_mirflickr.png" alt="ilb14_mirflickr" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/ilb14_averages_tobacco800.png" alt="ilb14_tobacco800" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Average run-time tests on a Xeon CPU E5-2609 v2 @ 2.50GHz with Linux and GCC 5.2</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Hamlet</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/softechict-nvidia_averages_3dpes.png" alt="softechict-nvidia_3dpes" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/softechict-nvidia_averages_hamlet.png" alt="softechict-nvidia_hamlet" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/softechict-nvidia_averages_mirflickr.png" alt="softechict-nvidia_mirflickr" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/softechict-nvidia_averages_tobacco800.png" alt="softechict-nvidia_tobacco800" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Average run-time tests on a Intel Core Duo @ 2.8GHz with OS~X and XCode 7.2.1.</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Hamlet</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_averages_3dpes.png" alt="pro_mid2009_3dpes" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_averages_hamlet.png" alt="pro_mid2009_hamlet" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_averages_mirflickr.png" alt="pro_mid2009_mirflickr" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_averages_tobacco800.png" alt="pro_mid2009_tobacco800" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Density-Size tests on a i7-4790 CPU @ 3.60 GHz with Windows and Microsof Visual Studio 2013</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/ilb14_density.png" alt="ilb14_density" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/ilb14_size.png" alt="ilb14_size" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Density-Size tests on a Xeon CPU E5-2609 v2 @ 2.50GHz with Linux and GCC 5.2</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/softechict-nvidia_density.png" alt="softechict-nvidia_density" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/softechict-nvidia_size.png" alt="softechict-nvidia_size" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Density-Size tests on a Intel Core Duo @ 2.8GHz with OS~X XCode 7.2.1.</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_density.png" alt="pro_mid2009_density" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_size.png" alt="pro_mid2009_size" height="260" width="415"></td>
</tr>
</table>

####... on 08/10/2016

<p align="justify">We performed tests on all datasets and algorithms on four different environments, in order to show newest features of YACCLAB: an Intel Core i7-4980HQ CPU @ 2.80GHz running OS~X with XCode 7.2.1, a Windows PC with i7-4770 CPU @ 3.40GHz and Microsoft Visual Studio 2013, a Linux workstation with a i5-6600 CPU @ 3.30GHz and gcc 5.3.1-14, and, finally, a Windows PC with i5-6600 CPU @ 3.30GHz and Microsoft Visual Studio 2013. All performance tests were repeated 10 times, and for each image the minimum execution time was considered. </p>

<table border="0">
<caption><h4>Average run-time tests on an Intel Core i7-4980HQ CPU @ 2.80GHz running OS~X with XCode 7.2.1</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Fingerprints</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/3dpesMAC.png" alt="3dpesMAC" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/fingerprintsMAC.png" alt="fingerprintsMAC" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>Hamlet</b></td>
 <td align="center"><b>Medical</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/hamletMAC.png" alt="hamletMAC" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/medicalMAC.png" alt="medicalMAC" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/mirflickrMAC.png" alt="mirflickrMAC" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/tobacco800MAC.png" alt="tobacco800MAC" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Average run-time tests on a i7-4770 CPU @ 3.40GHz with Windows and Microsoft Visual Studio 2013</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Fingerprints</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/3dpesWIN.png" alt="3dpesWIN" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/fingerprintsWIN.png" alt="fingerprintsWIN" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>Hamlet</b></td>
 <td align="center"><b>Medical</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/hamletWIN.png" alt="hamletWIN" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/medicalWIN.png" alt="medicalWIN" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/mirflickrWIN.png" alt="mirflickrWIN" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/tobacco800WIN.png" alt="tobacco800WIN" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Average run-time tests on a i5-6600 CPU @ 3.30GHz with Linux and GCC 5.3.1-14</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Fingerprints</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/3dpesLINUX.png" alt="3dpesLINUX" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/fingerprintsLINUX.png" alt="fingerprintsLINUX" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>Hamlet</b></td>
 <td align="center"><b>Medical</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/hamletLINUX.png" alt="hamletLINUX" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/medicalLINUX.png" alt="medicalLINUX" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/mirflickrLINUX.png" alt="mirflickrLINUX" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/tobacco800LINUX.png" alt="tobacco800LINUX" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Average run-time tests on a i5-6600 CPU @ 3.30GHz with Windows and Microsof Visual Studio 2013</h4></caption>
<tr>
 <td align="center"><b>3DPeS</b></td>
 <td align="center"><b>Fingerprints</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/3dpesWIN2.png" alt="3dpesWIN2" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/fingerprintsWIN2.png" alt="fingerprintsWIN2" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>Hamlet</b></td>
 <td align="center"><b>Medical</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/hamletWIN2.png" alt="hamletWIN2" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/medicalWIN2.png" alt="medicalWIN2" height="260" width="415"></td>
</tr>
</table>
<table border="0">
<tr>
 <td align="center"><b>MIRflickr</b></td>
 <td align="center"><b>Tobacco800</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/mirflickrWIN2.png" alt="mirflickrWIN2" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/tobacco800WIN2.png" alt="tobacco800WIN2" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Density-Size tests on an Intel Core i7-4980HQ CPU @ 2.80GHz running OS~X with XCode 7.2.1 (some algorithms have been omitted to make the charts more legible). </h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/densityMAC.png" alt="densityMAC" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/sizeMAC.png" alt="sizeMAC" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Density-Size tests on a i7-4770 CPU @ 3.40GHz with Windows and Microsoft Visual Studio 2013 (some algorithms have been omitted to make the charts more legible).</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/densityWIN.png" alt="densityWIN" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/sizeWIN.png" alt="sizeWIN" height="260" width="415"></td>
</tr>
</table>

<table border="0">
<caption><h4>Density-Size tests on a i5-6600 CPU @ 3.30GHz with Linux and and GCC 5.3.1-14 (some algorithms have been omitted to make the charts more legible).</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/densityLINUX.png" alt="densityLINUX" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/sizeLINUX.png" alt="sizeLINUX" height="260" width="415"></td>
</tr>
</table>


<table border="0">
<caption><h4>Density-Size tests on a i5-6600 CPU @ 3.30GHz with Windows and Microsof Visual Studio 2013 (some algorithms have been omitted to make the charts more legible).</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/densityWIN2.png" alt="densityWIN2" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/sizeWIN2.png" alt="sizeWIN2" height="260" width="415"></td>
</tr>
</table>

<a name="CT">[1]</a><p align="justify"><em>F. Chang, C.-J. Chen, and C.-J. Lu, “A linear-time component-labeling algorithm using contour tracing technique,” Computer Vision and Image Understanding, vol. 93, no. 2, pp. 206–220, 2004.</em></p>

<a name="CCIT">[2]</a><p align="justify"><em>W.-Y.  Chang,  C.-C.  Chiu,  and  J.-H.  Yang,  “Block-based  connected-component  labeling  algorithm  using  binary  decision  trees,” Sensors, vol. 15, no. 9, pp. 23 763–23 787, 2015.</em></p>

<a name="DiStefano">[3]</a><p align="justify"><em>L.  Di  Stefano  and  A.  Bulgarelli,  “A  Simple  and  Efficient  Connected Components Labeling Algorithm,” in International Conference on Image Analysis and Processing. IEEE, 1999, pp. 322–327.</em></p>

<a name="BBDT">[4]</a><p align="justify"><em>C.  Grana,  D.  Borghesani,  and  R.  Cucchiara,  “Optimized  Block-based Connected Components Labeling with Decision Trees,” IEEE Transac-tions on Image Processing, vol. 19, no. 6, pp. 1596–1609, 2010.</em></p>

<a name="LSL_STD">[5]</a><p align="justify"><em>L. Lacassagne and B. Zavidovique, “Light speed labeling: efficient connected component labeling on risc architectures,” Journal of Real-Time Image Processing, vol. 6, no. 2, pp. 117–135, 2011</em>.</p>

<a name="SAUF">[6]</a><p align="justify"><em>K. Wu, E. Otoo, and K. Suzuki, Optimizing two-pass connected-component labeling algorithms,” Pattern Analysis and Applications, vol. 12, no. 2, pp. 117–135, 2009.</em></p>

<a name="CTB">[7]</a><p align="justify"><em>L.  He,  X.  Zhao,  Y.  Chao,  and  K.  Suzuki, Configuration-Transition-
Based  Connected-Component  Labeling, IEEE  Transactions  on  Image Processing, vol. 23, no. 2, pp. 943–951, 2014.</em></p>

<a name="SBLA">[8]</a><p align="justify"><em>H.  Zhao,  Y.  Fan,  T.  Zhang,  and  H.  Sang, Stripe-based  connected components  labelling, Electronics  letters,  vol.  46,  no.  21,  pp.  1434–1436, 2010.</em></p>

<a name="PRED">[9]</a><p align="justify"><em>C. Grana, L. Baraldi, and F. Bolelli, Optimized Connected Components Labeling  with  Pixel  Prediction, in Advanced  Concepts  for  Intelligent Vision Systems, 2016.</em></p>

<a name="MIRFLICKR">[10]</a><p align="justify"><em>M. J. Huiskes and M. S. Lew, “The MIR Flickr Retrieval Evaluation,” in MIR ’08: Proceedings of the 2008 ACM International Conference on Multimedia Information Retrieval. New York, NY, USA: ACM, 2008. [Online]. Available: http://press.liacs.nl/mirflickr/</em></p>

<a name="TOBACCO1">[11]</a><p align="justify"><em>G. Agam, S. Argamon, O. Frieder, D. Grossman, and D. Lewis, “The Complex Document Image Processing (CDIP) Test Collection Project,” Illinois Institute of Technology, 2006. [Online]. Available: http://ir.iit.edu/projects/CDIP.html</em></p>

<a name="TOBACCO2">[12]</a><p align="justify"><em>D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and J. Heard, “Building a test collection for complex document information processing,” in Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 2006, pp. 665–666.</em></p>

<a name="TOBACCO3">[13]</a><p align="justify"><em>“The Legacy Tobacco Document Library (LTDL),” University of California, San Francisco, 2007. [Online]. Available: http://legacy. library.ucsf.edu/</em></p>

<a name="3DPES">[14]</a><p align="justify"><em>D. Baltieri, R. Vezzani, and R. Cucchiara, “3DPeS: 3D People Dataset for Surveillance and Forensics,” in Proceedings of the 2011 joint ACM workshop on Human gesture and behavior understanding. ACM, 2011, pp. 59–64.</em></p>

<a name="MEDICAL">[15]</a><p align="justify"><em>F. Dong, H. Irshad, E.-Y. Oh, M. F. Lerwill, E. F. Brachtel, N. C. Jones, N. W. Knoblauch, L. Montaser-Kouhsari, N. B. Johnson, L. K. Rao et al., “Computational Pathology to Discriminate Benign from Malignant Intraductal Proliferations of the Breast,” PloS one, vol. 9, no. 12, p. e114885, 2014.</em></p>

<a name="FINGERPRINTS">[16]</a><p align="justify"><em>D. Maltoni, D. Maio, A. Jain, and S. Prabhakar, Handbook of fingerprint
recognition. Springer Science & Business Media, 2009.</em></p>

<a name="YACCLAB">[17]</a><p align="justify"><em>C.Grana, F.Bolelli, L.Baraldi, and R.Vezzani, YACCLAB - Yet Another Connected Components Labeling Benchmark, Proceedings of the 23rd International Conference on Pattern Recognition, Cancun, Mexico, 4-8 Dec 2016, 2016</em></p>

