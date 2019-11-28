# YACCLAB: Yet Another Connected Components Labeling Benchmark

|         OS            | Build |          Compiler           | OpenCV | CMake | GPU |                                              Travis CI                                                          |  GitHub Actions |
|-----------------------|-------|-----------------------------|--------|-------|-----|-----------------------------------------------------------------------------------------------------------------|-----------------|
| Ubuntu 16.04.6 LTS    |  x64  |           5.4.0             |  3.1.0 | 3.13  | NO  | [![Build Status](https://travis-ci.org/prittt/YACCLAB.svg?branch=master)](https://travis-ci.org/prittt/YACCLAB) |                 |
| MacOS (Darwin 17.7.0) |  x64  | AppleClang 10 (Xcode-10.1)  |  3.1.0 | 3.13  | NO  | [![Build Status](https://travis-ci.org/prittt/YACCLAB.svg?branch=master)](https://travis-ci.org/prittt/YACCLAB) |                 |


<p align="justify">Please include the following references when citing the YACCLAB project/dataset:</p>

- <p align="justify"> Bolelli, Federico; Cancilla, Michele; Baraldi, Lorenzo; Grana, Costantino "Towards Reliable Experiments on the Performance of Connected Components Labeling Algorithms" Journal of Real-Time Image Processing, 2018. <a title="BibTex" href="http://imagelab.ing.unimore.it/files2/yacclab/YACCLAB_JRTIP2018_BibTex.html">BibTex</a>. <a title="Download" href="http://imagelab.ing.unimore.it/imagelab/pubblicazioni/2016-icpr-yacclab.pdf"><img src="https://raw.githubusercontent.com/prittt/YACCLAB/master/doc/pdf_logo.png" alt="Download." /></a></p>

- <p align="justify"> Grana, Costantino; Bolelli, Federico; Baraldi, Lorenzo; Vezzani, Roberto "YACCLAB - Yet Another Connected Components Labeling Benchmark" Proceedings of the 23rd International Conference on Pattern Recognition, Cancun, Mexico, 4-8 Dec 2016. <a title="BibTex" href="http://imagelab.ing.unimore.it/files2/yacclab/YACCLAB_ICPR2016_BibTex.html">BibTex</a>. <a title="Download" href="http://imagelab.ing.unimore.it/imagelab/pubblicazioni/2016-icpr-yacclab.pdf"><img src="https://raw.githubusercontent.com/prittt/YACCLAB/master/doc/pdf_logo.png" alt="Download." /></a></p>

<p align="justify">
YACCLAB is an open source <i>C++</i> project that enables researchers to test CCL algorithms under extremely variable points of view, running and testing algorithms on a collection of datasets described below. The benchmark performs the following tests which will be described later in this readme: <i>correctness</i>, average run-time (<i>average</i>), average run-time with steps (<i>average_ws</i>), <i>density</i>, <i>size</i>, <i>granularity</i> and memory accesses (<i>memory</i>).

Notice that 8-connectivity is always used in the project.
</p>

## Requirements

<p align="justify">
To correctly install and run YACCLAB following packages, libraries and utility are needed:

- CMake 3.8.2 or higher (https://cmake.org),
- OpenCV 3.0 or higher (http://opencv.org),
- Gnuplot (http://www.gnuplot.info/),
- One of your favourite IDE/compiler with C++14 support

GPU algorithms also require:
- CUDA Toolkit 9.2 or higher (https://developer.nvidia.com/cuda-toolkit)

Notes for gnuplot:
- on Windows system: be sure add gnuplot to system path if you want YACCLAB automatically generates charts.
- on MacOS system: 'pdf terminal' seems to be not available due to old version of cairo, 'postscript' is used instead.

</p>

## Installation (refer to the image below)

- <p align="justify">Clone the GitHub repository (HTTPS clone URL: https://github.com/prittt/YACCLAB.git) or simply download the full master branch zip file and extract it (e.g YACCLAB folder).</p>
- <p align="justify">Install software in YACCLAB/bin subfolder (suggested) or wherever you want using CMake (point 2 of the example image). Note that CMake should automatically find the OpenCV path whether correctly installed on your OS (3), download the YACCLAB Dataset (be sure to check the box if you want to download it (4a) and (4b) or to select the correct path if the dataset is already on your file system (7)), and create a C++ project for the selected IDE/compiler (9-10). Moreover, if you want to test 3D or GPU algorithms tick the corresponding boxes (5) and (6). </p>

![Cmake](doc/readme_github.png)

- <p align="justify">Set the <a href="#conf">configuration file (config.yaml)</a> placed in the installation folder (bin in this example) in order to select desired tests.</p>

- <p align="justify">Open the project, compile and run it: the work is done!</p>

## How to include a YACCLAB algorithm into your own project?

<p align="justify">If your project requires a Connected Components Labeling algorithm and you are not interested in the whole YACCLAB benchmark you can use the <i>connectedComponent</i> function of the OpenCV library which implements the BBDT and SAUF algorithms since version 3.2.</p>
<p align="justify">Anyway, when the <i>connectedComponents</i> function is called, lot of additional code will be executed together with the core function. If your project requires the best performance you can include an algorithm implemented in YACCLAB adding the following files to your project:</p>
<ol>
  <li><i>labeling_algorithms.h</i> and <i>labeling_algorithms.cc</i> which define the base class from which every algorithm derives from.</li>
  <li><i>label_solver.h</i> and <i>label_solver.cc</i> which cointain the implementation of labels solving algorithms.</li>
  <li><i>memory_tester.h</i> and <i>performance_evaluator.h</i> just to make things work without changing the code.</li>
  <li><i>headers</i> and <i>sources</i> files of the required algorithm/s. The association between algorithms and headers/sources files is reported in the table below.</li>
</ol>  
 <table>
  <tr>
    <th></th>
    <th>Algorithm Name</th>
    <th width="130">Authors</th>
    <th>Year</th>
    <th>Acronym</th>
    <th>Required Files</th>
    <th>Templated on Labels Solver</th>
  </tr>	
	
  <tr>
    <td align="center" rowspan="11">CPU</td>
    <td align="center">-</td>
    <td align="center">L. Di Stefano,<br>A. Bulgarelli <sup><a href="#DiStefano">[3]</a></sup></td>
    <td align="center">1999</td>
    <td align="center">DiStefano</td>
    <td align="center"><i>labeling_distefano_1999.h</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Contour Tracing</td>
    <td align="center">F. Chang,</br>C.J. Chen,</br>C.J. Lu <sup><a href="#CT">[1]</a></sup></td>
    <td align="center">1999</td>
    <td align="center">CT</td>
    <td align="center"><i>labeling_fchang_2003.h</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Configuration Transition Based</td>
    <td align="center">L. He,</br>X. Zhao,</br>Y. Chao,</br>K. Suzuki <sup><a href="#CTB">[7]</a></sup></td>
    <td align="center">1999</td>
    <td align="center">CTB</td>
    <td align="center"><i>labeling_he_2014.h</i>, <i>labeling_he_2014_graph.inc</i>
    <td align="center">YES</td>
  </tr>
  <tr>
    <td align="center">Scan Array-based with Union Find</td>
    <td align="center">K. Wu,</br>E. Otoo,</br>K. Suzuki <sup><a href="#SAUF">[6]</a></sup></td>
    <td align="center">2009</td>
    <td align="center">SAUF</td>
    <td align="center"><i>labeling_wu_2009.h</i>, <i>labeling_wu_2009_tree.inc</i></td>
    <td align="center">YES</td>
  </tr>
    <tr>
    <td align="center">Stripe-Based Labeling Algorithm</td>
    <td align="center">H.L. Zhao,</br>Y.B. Fan,</br>T.X. Zhang,</br>H.S. Sang <sup><a href="#SBLA">[8]</a></sup></td>
    <td align="center">2010</td>
    <td align="center">SBLA</td>
    <td align="center"><i>labeling_zhao_2010.h</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Block-Based with Decision Tree</td>
    <td align="center">C. Grana,</br>D. Borghesani,</br>R. Cucchiara <sup><a href="#BBDT">[4]</a></sup></td>
    <td align="center">2010</td>
    <td align="center">BBDT</td>
    <td align="center"><i>labeling_grana_2010.h</i>, <i>labeling_grana_2010_tree.inc</i></td>
    <td align="center">YES</td>
  </tr>
  <tr>
    <td align="center">Block-Based with Binary Decision Trees</td>
    <td align="center">W.Y. Chang,</br>C.C. Chiu,</br>J.H. Yang <sup><a href="#CCIT">[2]</a></sup></td>
    <td align="center">2015</td>
    <td align="center">CCIT</td>
    <td align="center"><i>labeling_wychang_2015.h</i>, <i>labeling_wychang_2015_tree.inc</i>, <i>labeling_wychang_2015_tree_0.inc</i></td>
    <td align="center">YES</td>
  </tr>
  <tr>
    <td align="center">Light Speed Labeling</td>
    <td align="center">L. Cabaret,</br>L. Lacassagne,</br>D. Etiemble <sup><a href="#LSL_STD">[5]</a></sup></td>
    <td align="center">2016</td>
    <td align="center">LSL_STD<small><sup>I</sup></small></br>LSL_STDZ<small><sup>II</sup></small></br>LSL_RLE<small><sup>III</sup></small></td>
    <td align="center"><i>labeling_lacassagne_2016.h</i>, <i>labeling_lacassagne_2016_code.inc</i></td>
    <td align="center">YES<small><sup>IV</sup></small></td>
  </tr>
  <tr>
    <td align="center">Pixel Prediction</td>
    <td align="center">C.Grana,</br>L. Baraldi,</br>F. Bolelli <sup><a href="#PRED">[9]</a></sup></td>
    <td align="center">2016</td>
    <td align="center">PRED</td>
    <td align="center"><i>labeling_grana_2016.h</i>, <i>labeling_grana_2016_forest.inc</i>, <i>labeling_grana_2016_forest_0.inc</i>
    <td align="center">YES</td>
  </tr>
  <tr>
    <td align="center">Directed Rooted Acyclic Graph</td>
    <td align="center">F. Bolelli,</br>L. Baraldi,</br>M. Cancilla,</br>C. Grana <sup><a href="#DRAG">[23]</a></sup></td>
    <td align="center">2018</td>
    <td align="center">DRAG</td>
    <td align="center"><i>labeling_bolelli_2018.h</i>, <i>labeling_grana_2018_drag.inc</i></td>
    <td align="center">YES</td>
  </tr>
  <tr>
    <td align="center">Spaghetti Labeling</td>
    <td align="center">F. Bolelli,</br>S. Allegretti,</br>L. Baraldi,</br>C. Grana <sup><a href="#Spaghetti">[20]</a></sup></td>
    <td align="center">2019</td>
    <td align="center">Spaghetti</td>
    <td align="center"><i>labeling_bolelli_2019.h</i>, <i>labeling_bolelli_2019_forest.inc</i>, <i>labeling_bolelli_2019_forest_firstline.inc</i>, <i>labeling_bolelli_2019_forest_lastline.inc</i>, <i>labeling_bolelli_2019_forest_singleline.inc</i></td>
    <td align="center">YES</td>
  </tr>
  <tr>
    <td align="center">Null Labeling</td>
    <td align="center">F. Bolelli,</br>M. Cancilla,</br>L. Baraldi,</br>C. Grana</td>
    <td align="center">-</td>
    <td align="center">NULL<small><sup>V</sup></small></td>
    <td align="center"><i>labeling_null.h</i></td>
    <td align="center">NO</td>
  </tr>


  <tr>
    <td align="center" rowspan="9">GPU</td>
    <td align="center">Union Find</td>
    <td align="center">V. Oliveira,</br>R. Lotufo<sup><a href="#UF">[18]</a></sup></td>
    <td align="center">2010</td>
    <td align="center">UF</td>
    <td align="center"><i>labeling_CUDA_UF.cu</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Optimized</br>Label Equivalence</td>
    <td align="center">O. Kalentev,</br>A. Rai,</br>S. Kemnitz,</br>R. Schneider<sup><a href="#OLE">[19]</a></sup></td>
    <td align="center">2011</td>
    <td align="center">OLE</td>
    <td align="center"><i>labeling_CUDA_OLE.cu</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Block Equivalence</td>
    <td align="center">S. Zavalishin,</br>I. Safonov,</br>Y. Bekhtin,</br>I. Kurilin<sup><a href="#BE">[20]</a></sup></td>
    <td align="center">2016</td>
    <td align="center">BE</td>
    <td align="center"><i>labeling_CUDA_BE.cu</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Distanceless</br>Label Propagation</td>
    <td align="center">L. Cabaret,</br>L. Lacassagne,</br>D. Etiemble<sup><a href="#DLP">[21]</a></sup></td>
    <td align="center">2017</td>
    <td align="center">DLP</td>
    <td align="center"><i>labeling_CUDA_DLP.cu</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">CUDA SAUF</td>
    <td align="center">S. Allegretti,</br>F. Bolelli,</br>M. Cancilla,</br>C. Grana</td>
    <td align="center">-</td>
    <td align="center">C-SAUF</td>
    <td align="center"><i>labeling_CUDA_SAUF.cu</i>,</br><i>labeling_wu_2009_tree.inc</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">CUDA BBDT</td>
    <td align="center">S. Allegretti,</br>F. Bolelli,</br>M. Cancilla,</br>C. Grana</td>
    <td align="center">-</td>
    <td align="center">C-BBDT</td>
    <td align="center"><i>labeling_CUDA_BBDT.cu</i>, <i>labeling_grana_2010_tree.inc</i></td>
    <td align="center">NO</td>
  </tr>
    <tr>
    <td align="center">CUDA DRAG</td>
    <td align="center">S. Allegretti,</br>F. Bolelli,</br>M. Cancilla,</br>C. Grana</td>
    <td align="center">-</td>
    <td align="center">C-DRAG</td>
    <td align="center"><i>labeling_CUDA_DRAG.cu</i></td>
    <td align="center">NO</td>
  </tr>
  <tr>
    <td align="center">Block-based Union Find</td>
    <td align="center">S. Allegretti,</br>F. Bolelli,</br>C. Grana<sup><a href="#BUF_BKE">[24]</a></sup></td>
    <td align="center">2019</td>
    <td align="center">BUF</td>
    <td align="center"><i>labeling_CUDA_BUF.cu</i></td>
    <td align="center">NO</td>
  </tr>
    <tr>
    <td align="center">Block-based Komura Equivalence</td>
    <td align="center">S. Allegretti,</br>F. Bolelli,</br>C. Grana<sup><a href="#BUF_BKE">[24]</a></sup></td>
    <td align="center">2019</td>
    <td align="center">BKE</td>
    <td align="center"><i>labeling_CUDA_BKE.cu</i></td>
    <td align="center">NO</td>
  </tr>


</table>

(<small>I</small>) standard version </br>
(<small>II</small>) with zero-offset optimization </br>
(<small>III</small>) with RLE compression </br>
(<small>IV</small>) only on TTA and UF </br>
(<small>V</small>) it only copies the pixels from the input image to the output one simply defining a lower bound limit for the execution time of CCL algorithms on a given machine and dataset.

### Example of Algorithm Usage Outside the Benchmark

```c++
#include "labels_solver.h"
#include "labeling_algorithms.h"
#include "labeling_grana_2010.h" // To include the algorithm code (BBDT in this example)

#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    BBDT<UFPC> BBDT_UFPC; // To create an object of the desired algorithm (BBDT in this example)
                          // templated on the labels solving strategy. See the README for the
                          // complete list of the available labels solvers, available algorithms
                          // (N.B. non all the algorithms are templated on the solver) and their
                          // acronyms.

    BBDT_UFPC.img_ = imread("test_image.png", IMREAD_GRAYSCALE); // To load into the CCL object
                                                                 // the BINARY image to be labeled

    threshold(BBDT_UFPC.img_, BBDT_UFPC.img_, 100, 1, THRESH_BINARY); // Just to be sure that the
                                                                      // loaded image is binary

    BBDT_UFPC.PerformLabeling(); // To perform Connected Components Labeling!

    Mat1i output = BBDT_UFPC.img_labels_; // To get the output labeled image  
    unsigned n_labels = BBDT_UFPC.n_labels_; // To get the number of labels found in the input img

    return EXIT_SUCCESS;
}
```

<a name="conf"></a>
## Configuration File
<p align="justify">A <tt>YAML</tt> configuration file placed in the installation folder lets you specify which kinds of tests should be performed, on which datasets and on which algorithms.
Four categories of algorithms are supported: 2D CPU, 2D GPU, 3D CPU and 3D GPU. For each of them, the configuration parameters are reported below. </p>

- <i>execute</i> - boolean value which specifies whether the current category of algorithms will be tested: 
```yaml
execute:    true
```

- <i>perform</i> - dictionary which specifies the <a href="#conf">kind of tests</a> to perform: 
```yaml
perform:
  correctness:        false
  average:            true
  average_with_steps: false
  density:            false
  granularity:        false
  memory:             false
```

- <i>correctness_tests</i> - dictionary indicating the kind of correctness tests to perform:
```yaml
correctness_tests:
  eight_connectivity_standard: true
  eight_connectivity_steps:    true
  eight_connectivity_memory:   true
```

- <i>tests_number</i> - dictionary which sets the number of runs for each test available:
```yaml
tests_number:
  average:            10
  average_with_steps: 10
  density:            10
  granularity:        10
```

- <i>algorithms</i> - list of algorithms on which apply the chosen tests:
```yaml
algorithms:
  - SAUF_RemSP
  - SAUF_TTA
  - BBDT_RemSP
  - BBDT_UFPC
  - CT
  - labeling_NULL
```

- <i>check_datasets</i>, <i>average_datasets</i>, <i>average_ws_datasets</i> and <i>memory_datasets</i> - lists of <a href="#conf">datasets</a> on which, respectively, correctness, average, average_ws and memory tests should be run:
<!--
- <i>check_datasets:</i> list of datasets on which CCL algorithms should be checked.
- <i>average_datasets:</i> list of datasets on which average test should be run.
- <i>average_ws_datasets:</i> list of datasets on which <i>average_ws</i> test should be run.
- <i>memory_datasets:</i> list of datasets on which memory test should be run.
-->
```yaml
...
average_datasets: ["3dpes", "fingerprints", "hamlet", "medical", "mirflickr", "tobacco800", "xdocs"]
...
```

<p style=text-align: justify;>Finally, the following configuration parameters are common to all categories.</p>

- <i>paths</i> - dictionary with both input (datasets) and output (results) paths. It is automatically filled by Cmake during the creation of the project:
```yaml
paths: {input: "<datasets_path>", output: "<output_results_path>"}
```

- <i>write_n_labels</i> - whether to report the number of connected components in the output files:
```yaml
write_n_labels: false
```

- <i>color_labels</i> - whether to output a colored version of labeled images during tests:
```yaml
color_labels: {average: false, density: false}
```

- <i>save_middle_tests</i> - dictionary specifying, separately for every test, whether to save the output of single runs, or only a summary of the whole test:
```yaml
save_middle_tests: {average: false, average_with_steps: false, density: false, granularity: false}
```

<!-- <table>
<tr><td width=200><b>Parameter<b></td><td width=300><b>Description</b></td></tr>
<tr><td>perform</td><td><p align="justify">Dictionary which specifies the kind of tests to perform (<i>correctness</i>, <i>average</i>, <i>average_ws</i>, <i>density and size</i>, <i>granularity</i> and <i>memory</i>).</p></td></tr>
<tr><td>correcteness_tests</td><td><p align="justify">Dictionary indicating the kind of correctness tests to perform.</p></td></tr>
<tr><td>tests_number</td><td><p align="justify">Dictionary which sets the number of runs for each test available.</p></td></tr>
<tr bgcolor=gray align=center><td colspan="2"><i>Algorithms</i></td></tr>
<tr><td>algorithms</td><td><p align="justify">List of algorithms on which apply the chosen tests.</p></td></tr>
<tr align=center><td colspan="2"><i>Datasets</i></td></tr>
<tr><td>check_datasets</td><td><p align="justify">List of datasets on which CCL algorithms should be checked.</p></td></tr>
<tr><td>average_datasets</td><td><p align="justify">List of datasets on which average test should be run.</p></td></tr>
<tr><td>average_ws_datasets</td><td><p align="justify">List of datasets on which <i>average_ws</i> test should be run.</p></td></tr>
<tr><td>memory_datasets</td><td><p align="justify">List of datasets on which memory test should be run.</p></td></tr>
<tr align=center><td colspan="2"><i>Utilities</i></td></tr>
<tr><td>paths</td><td><p align="justify">Dictionary with both input (datasets) and output (results) paths.</p></td></tr>
<tr><td>write_n_labels</td><td><p align="justify">Whether to report the number of connected components in the output files.</p></td></tr>
<tr><td>color_labels</td><td><p align="justify">Whether to output a colored version of labeled images during tests.</p></td></tr>
<tr><td>save_middle_tests</td><td><p align="justify">Dictionary specifying, separately for every test, whether to save the output of single runs, or only a summary of the whole test.</p></td></tr>
</table>
-->

## How to Extend YACCLAB with New Algorithms

Work in progress.
<!-- <p align="justify">YACCLAB has been designed with extensibility in mind, so that new resources can be easily integrated into the project. A CCL algorithm is coded with a <tt>.h</tt> header file, which declares a function implementing the algorithm, and a <tt>.cpp</tt> source file which defines the function itself. The function must follow a standard signature: its first parameter should be a const reference to an OpenCV Mat1b matrix, containing the input image, and its second parameter should be a reference to a Mat1i matrix, which shall be populated with computed labels. The function must also return an integer specifying the number of labels found in the image, included background's one. For example:</p>
```c++
int MyLabelingAlgorithm(const cv::Mat1b& img,cv::Mat1i &imgLabels);
```
<p align="justify">Making YACCLAB aware of a new algorithm requires two more steps. The new header file has to be included in <tt>labelingAlgorithms.h</tt>, which is in charge of collecting all the algorithms in YACCLAB. This file also defines a C++ map with function pointers to all implemented algorithms, which has also to be updated with the new function.</p>

<p align="justify">Once an algorithm has been added to YACCLAB, it is ready to be tested and compared to the others. To include the newly added algorithm in a test, it is sufficient to include its function name in the <tt>CCLAlgorithmsFunc</tt> <a href="#conf">parameter</a> and a display name in the <tt>CCLAlgorithmsName</tt> parameter. We look at YACCLAB as a growing effort towards better reproducibility of CCL algorithms, so implementations of new and existing labeling methods are welcome.</p>
-->

<a name="datasets"></a>
## The YACCLAB Dataset
<p align="justify">The YACCLAB dataset includes both synthetic and real images and it is suitable for a wide range of applications, ranging from document processing to surveillance, and features a significant variability in terms of resolution, image density, variance of density, and number of components. All images are provided in 1 bit per pixel PNG format, with 0 (black) being background and 1 (white) being foreground. The dataset will be automatically downloaded by CMake during the installation process as described in the <a href="#inst">installation</a> paragraph, or it can be found at http://imagelab.ing.unimore.it/yacclab. Images are organized by folders as follows: </p>

- <b>Synthetic Images</b>:
	- <b>Classical:<sup><a href="#BBDT">4</a></sup></b><p align="justify"> A set of synthetic random noise images who contain black and white random noise with 9 different foreground densities (10% up to 90%), from a low resolution of 32x32 pixels to a maximum resolution of 4096x4096 pixels, allowing to test the scalability and the effectiveness of different approaches when the number of labels gets high. For every combination of size and density, 10 images are provided for a total of 720 images. The resulting subset allows to evaluate performance both in terms of scalability on the number of pixels and on the number of labels (density). </p>
	- <b>Granularity:<sup><a href="#LSL">4</a></sup></b><p align="justify"> This dataset allows to test algorithms varying not only the pixels density but also their granularity <i>g</i> (<i>i.e.</i>, dimension of minimum foreground block), underlying the behaviour of different proposals when the number of provisional labels changes. All the images have a resolution of 2048x2048 and are generated with the Mersenne Twister MT19937 random number generator implemented in the <i>C++</i> standard and starting with a "seed" equal to zero. Density of the images ranges from 0% to 100% with step of 1% and for every density value 16 images with pixels blocks of <i>gxg</i> with <i>g</i> ∈ [1,16] are generated. Moreover, the procedure has been repeated 10 times for every couple of density-granularity for a total of 16160 images.</p>

- <b>MIRflickr:<sup><a href="#MIRFLICKR">10</a></sup></b><p align="justify"> Otsu-binarized version of the MIRflickr dataset, publicly available under a Creative Commons License. It contains 25,000 standard resolution images taken from Flickr. These images have an average resolution of 0.17 megapixels, there are few connected components (495 on average) and are generally composed of not too complex patterns, so the labeling is quite easy and fast.</p>

- <b>Hamlet:</b><p align="justify"> A set of 104 images scanned from a version of the Hamlet found on Project Gutenberg (http://www.gutenberg.org). Images have an average amount of 2.71 million of pixels to analyze and 1447 components to label, with an average foreground density of 0.0789. </p>

- <b>Tobacco800:<sup><a href="#TOBACCO1">11</a>,<a href="#TOBACCO2">12</a>,<a href="#TOBACCO3">13</a></sup></b><p align="justify"> A set of 1290 document images. It is a realistic database for document image analysis research as these documents were collected and scanned using a wide variety of equipment over time. Resolutions of documents in Tobacco800 vary significantly from 150 to 300 DPI and the dimensions of images range from 1200 by 1600 to 2500 by 3200 pixels. Since CCL is one of the initial preprocessing steps in most layout analysis or OCR algorithms, hamlet and tobacco800 allow to test the algorithm performance in such scenarios. </p>

- <b>3DPeS:<sup><a href="#3DPES">14</a></sup></b> <p align="justify"> It comes from 3DPeS (3D People Surveillance Dataset), a surveillance dataset designed mainly for people re-identification in multi camera systems with non-overlapped fields of view. 3DPeS can be also exploited to test many other tasks, such as people detection, tracking, action analysis and trajectory analysis. The background models for all cameras are provided, so a very basic technique of motion segmentation has been applied to generate the foreground binary masks, i.e.,  background subtraction and fixed thresholding. The analysis of the foreground masks to remove small connected components and for nearest neighbor matching is a common application for CCL. </p>

- <b>Medical:<sup><a href="#MEDICAL">15</a></sup></b><p align="justify"> This dataset is composed by histological images and allow us to cover this fundamental medical field. The process used for nuclei segmentation and binarization is described in  <a href="#MEDICAL">[12]</a>. The resulting dataset is a collection of 343 binary histological images with an average amount of 1.21 million of pixels to analyze and 484 components to label. </p>

- <b>Fingerprints:<sup><a href="#FINGERPRINTS">16</a></sup></b><p align="justify"> This dataset counts 960 fingerprint images collected by using low-cost optical sensors or synthetically generated. These images were taken from the three Verification Competitions FCV2000, FCV2002 and FCV2004. In order to fit CCL application, fingerprints have been binarized using an adaptive threshold and then negated in order to have foreground pixel with value 255. Most of the original images have a resolution of 500 DPI and their dimensions range from 240 by 320 up to 640 by 480 pixels. </p>

<a name="tests"></a>
## Available Tests

- <b>Average run-time tests:</b> <p align="justify"> execute an algorithm on every image of a dataset. The process can be repeated more times in a single test, to get the minimum execution time for each image: this allows to get more reproducible results and overlook delays produced by other running processes. It is also possible to compare the execution speed of different algorithms on the same dataset: in this case, selected algorithms (see <a href="#conf">Configuration File</a> for more details) are executed sequentially on every image of the dataset. Results are presented in three different formats: a plain text file, histogram charts (.pdf/.ps), either in color or in gray-scale, and a LaTeX table, which can be directly included in research papers.</p>

- <b>Density and size tests:</b> <p align="justify"> check the performance of different CCL algorithms when they are executed on images with varying foreground density and size. To this aim, a list of algorithms selected by the user is run sequentially on every image of the test_random dataset. As for run-time tests, it is possible to repeat this test for more than one run. The output is presented as both plain text and charts(.pdf/.ps). For a density test, the mean execution time of each algorithm is reported for densities ranging from 10% up to 90%, while for a size test the same is reported for resolutions ranging from 32x32 up to 4096x4096.</p>

- <b>Memory tests:</b> <p align="justify"> are useful to understand the reason for the good performances of an algorithm or in general to explain its behavior. Memory tests compute the average number of accesses to the label image (i.e the image used to store the provisional and then the final labels for the connected components), the average number of accesses to the binary image to be labeled, and, finally, the average number of accesses to data structures used to solve the equivalences between label classes. Moreover, if an algorithm requires extra data, memory tests summarize them as ``other'' accesses and return the average. Furthermore, all average contributions of an algorithm and dataset are summed together in order to show the total amount of memory accesses. Since counting the number of memory accesses imposes additional computations, functions implementing memory access tests are different from those implementing run-time and density tests, to keep run-time tests as objective as possible.</p>

## Examples of YACCLAB Output Results
Work in progress.
<!--
## Results ...

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
- NULL labeling is an algorithm that defines a lower bound limit for the execution time of CCL algorithms on a given machine and dataset. As the name suggests, this algorithm does not provide the correct connected components for a given image. It only checks the pixels of that image and sets almost randomly the value of the output.

SAUF and BBDT are the algorithms currently included in OpenCV.

### ... on 04/21/2016

<p align="justify">To  make  a  first  performance  comparison  and  to  showcase automatically  generated  charts  we  have  run  each algorithm in YACCLAB on all datasets and in three different environments:  a  Windows  PC  with  a  i7-4790  CPU  @  3.60 GHz and Microsoft Visual Studio 2013, a Linux workstation with a Xeon CPU E5-2609 v2 @ 2.50GHz and GCC 5.2, and a Intel Core Duo @ 2.8 GHz running OS~X with XCode 7.2.1. Average run-time tests, as well as density and size tests, were repeated 10 times, and for each image the minimum execution time was considered.</p>

<table border="0">
<caption><h4>Average run-time tests on a i7-4790 CPU @ 3.60 GHz with Windows and Microsoft Visual Studio 2013 (lower is better)</h4></caption>
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
<caption><h4>Average run-time tests on a Xeon CPU E5-2609 v2 @ 2.50GHz with Linux and GCC 5.2 (lower is better)</h4></caption>
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
<caption><h4>Average run-time tests on a Intel Core Duo @ 2.8GHz with OS~X and XCode 7.2.1 (lower is better)</h4></caption>
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
<caption><h4>Density-Size tests on a i7-4790 CPU @ 3.60 GHz with Windows and Microsoft Visual Studio 2013 (lower is better)</h4></caption>
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
<caption><h4>Density-Size tests on a Xeon CPU E5-2609 v2 @ 2.50GHz with Linux and GCC 5.2 (lower is better)</h4></caption>
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
<caption><h4>Density-Size tests on a Intel Core Duo @ 2.8GHz with OS~X XCode 7.2.1 (lower is better)</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_density.png" alt="pro_mid2009_density" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/pro_mid2009_size.png" alt="pro_mid2009_size" height="260" width="415"></td>
</tr>
</table>

### ... on 08/10/2016

<p align="justify">We performed tests on all datasets and algorithms on four different environments, in order to show newest features of YACCLAB: an Intel Core i7-4980HQ CPU @ 2.80GHz running OS~X with XCode 7.2.1, a Windows PC with i7-4770 CPU @ 3.40GHz and Microsoft Visual Studio 2013, a Linux workstation with a i5-6600 CPU @ 3.30GHz and gcc 5.3.1-14, and, finally, a Windows PC with i5-6600 CPU @ 3.30GHz and Microsoft Visual Studio 2013. All performance tests were repeated 10 times, and for each image the minimum execution time was considered. </p>

<table border="0">
<caption><h4>Average run-time tests on an Intel Core i7-4980HQ CPU @ 2.80GHz running OS~X with XCode 7.2.1 (lower is better)</h4></caption>
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
<caption><h4>Average run-time tests on a i7-4770 CPU @ 3.40GHz with Windows and Microsoft Visual Studio 2013 (lower is better)</h4></caption>
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
<caption><h4>Average run-time tests on a i5-6600 CPU @ 3.30GHz with Linux and GCC 5.3.1-14 (lower is better)</h4></caption>
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
<caption><h4>Average run-time tests on a i5-6600 CPU @ 3.30GHz with Windows and Microsoft Visual Studio 2013 (lower is better)</h4></caption>
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
<caption><h4>Density-Size tests on an Intel Core i7-4980HQ CPU @ 2.80GHz running OS~X with XCode 7.2.1 (some algorithms have been omitted to make the charts more legible, lower is better). </h4></caption>
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
<caption><h4>Density-Size tests on a i7-4770 CPU @ 3.40GHz with Windows and Microsoft Visual Studio 2013 (some algorithms have been omitted to make the charts more legible, lower is better).</h4></caption>
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
<caption><h4>Density-Size tests on a i5-6600 CPU @ 3.30GHz with Linux and and GCC 5.3.1-14 (some algorithms have been omitted to make the charts more legible, lower is better).</h4></caption>
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
<caption><h4>Density-Size tests on a i5-6600 CPU @ 3.30GHz with Windows and Microsoft Visual Studio 2013 (some algorithms have been omitted to make the charts more legible, lower is better).</h4></caption>
<tr>
 <td align="center"><b>Density</b></td>
 <td align="center"><b>Size</b></td>
</tr>
<tr>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/densityWIN2.png" alt="densityWIN2" height="260" width="415"></td>
 <td><img src="http://imagelab.ing.unimore.it/files2/yacclab/sizeWIN2.png" alt="sizeWIN2" height="260" width="415"></td>
</tr>
</table>

<table>
<caption><h4>Analysis of memory accesses required by connected components computation for 'Random' dataset. The numbers are given in millions of accesses</h4></caption>
	<tr>
   <td align="center">Algorithm           </td>
   <td align="center">Total Accesses      </td>
   <td align="center">Binary Image        </td>
   <td align="center">Label Image         </td>
   <td align="center">Equivalence Vector/s</td>
   <td align="center">Other               </td>
	</tr>
 <tr>
   <td align="center">SAUF  </td>
   <td align="center">20.963</td>
   <td align="center">5.475 </td>
   <td align="center">11.288</td>
   <td align="center">4.200 </td>
   <td align="center">-     </td>
	</tr>
	<tr>
   <td align="center">DiStefano</td>
   <td align="center">187.512  </td>
   <td align="center">2.796    </td>
   <td align="center">12.580   </td>
   <td align="center">171.972  </td>
   <td align="center">0.164    </td>
	</tr>
	<tr>
   <td align="center">BBDT  </td>
   <td align="center">12.237</td>
   <td align="center">6.012 </td>
   <td align="center">4.757	</td>
   <td align="center">1.468	</td>
   <td align="center">-     </td>
	</tr>
	<tr>
   <td align="center">LSL_STD</td>
   <td align="center">27.792  </td>
   <td align="center">2.796	  </td>
   <td align="center">2.796	  </td>
   <td align="center">1.616	  </td>
   <td align="center">20.584  </td>
	</tr>
 	<tr>
   <td align="center">PRED  </td>
   <td align="center">20.194</td>
   <td align="center">4.706	</td>
   <td align="center">11.288</td>
   <td align="center">4.200	</td>
   <td align="center">-     </td>
	</tr>
 	<tr>
   <td align="center">NULL </td>
   <td align="center">5.592</td>
   <td align="center">2.796</td>
   <td align="center">2.796</td>
   <td align="center">-    </td>
   <td align="center">-    </td>
	</tr>
</table>

<table>
<caption><h4>Analysis of memory accesses required by connected components computation for 'Tobacco800' dataset. The numbers are given in millions of accesses</h4></caption>
	<tr>
   <td align="center">Algorithm           </td>
   <td align="center">Total Accesses      </td>
   <td align="center">Binary Image        </td>
   <td align="center">Label Image         </td>
   <td align="center">Equivalence Vector/s</td>
   <td align="center">Other               </td>
	</tr>
  <tr>
   <td align="center">SAUF  </td>
   <td align="center">23.874</td>
   <td align="center">4.935 </td>
   <td align="center">14.286</td>
   <td align="center">4.653 </td>
   <td align="center">-     </td>
	</tr>
  <tr>
   <td align="center">DiStefano</td>
   <td align="center">17.041   </td>
   <td align="center">4.604    </td>
   <td align="center">10.393   </td>
   <td align="center">2.037    </td>
   <td align="center">0.007    </td>
	</tr>
  <tr>
   <td align="center">BBDT  </td>
   <td align="center">12.046</td>
   <td align="center">4.942 </td>
   <td align="center">6.982 </td>
   <td align="center">0.122 </td>
   <td align="center">-     </td>
	</tr>
   <tr>
   <td align="center">LSL_STD</td>
   <td align="center">38.267 </td>
   <td align="center">4.604  </td>
   <td align="center">4.604  </td>
   <td align="center">1.189  </td>
   <td align="center">27.870 </td>
	</tr>
  <tr>
   <td align="center">PRED  </td>
   <td align="center">23.799</td>
   <td align="center">4.860 </td>
   <td align="center">14.286</td>
   <td align="center">4.653 </td>
   <td align="center">-     </td>
	</tr>
  <tr>
   <td align="center">NULL </td>
   <td align="center">9.208</td>
   <td align="center">4.604</td>
   <td align="center">4.604</td>
   <td align="center">-    </td>
   <td align="center">-    </td>
	</tr>
</table>
-->

## References

<p align="justify"><em><a name="CT">[1]</a> F. Chang, C.-J. Chen, and C.-J. Lu, “A linear-time component-labeling algorithm using contour tracing technique,” Computer Vision and Image Understanding, vol. 93, no. 2, pp. 206–220, 2004.</em></p>
<p align="justify"><em><a name="CCIT">[2]</a> W.-Y.  Chang,  C.-C.  Chiu,  and  J.-H.  Yang,  “Block-based  connected-component  labeling  algorithm  using  binary  decision  trees,” Sensors, vol. 15, no. 9, pp. 23 763–23 787, 2015.</em></p>
<p align="justify"><em><a name="DiStefano">[3]</a> L.  Di  Stefano  and  A.  Bulgarelli,  “A  Simple  and  Efficient  Connected Components Labeling Algorithm,” in International Conference on Image Analysis and Processing. IEEE, 1999, pp. 322–327.</em></p>
<p align="justify"><em><a name="BBDT">[4]</a> C.  Grana,  D.  Borghesani,  and  R.  Cucchiara,  “Optimized  Block-based Connected Components Labeling with Decision Trees,” IEEE Transac-tions on Image Processing, vol. 19, no. 6, pp. 1596–1609, 2010.</em></p>
<p align="justify"><em><a name="LSL_STD">[5]</a> L. Lacassagne and B. Zavidovique, “Light speed labeling: efficient connected component labeling on risc architectures,” Journal of Real-Time Image Processing, vol. 6, no. 2, pp. 117–135, 2011</em>.</p>
<p align="justify"><em><a name="SAUF">[6]</a> K. Wu, E. Otoo, and K. Suzuki, Optimizing two-pass connected-component labeling algorithms,” Pattern Analysis and Applications, vol. 12, no. 2, pp. 117–135, 2009.</em></p>
<p align="justify"><em><a name="CTB">[7]</a> L.  He,  X.  Zhao,  Y.  Chao,  and  K.  Suzuki, Configuration-Transition-
Based  Connected-Component  Labeling, IEEE  Transactions  on  Image Processing, vol. 23, no. 2, pp. 943–951, 2014.</em></p>
<p align="justify"><em><a name="SBLA">[8]</a> H.  Zhao,  Y.  Fan,  T.  Zhang,  and  H.  Sang, Stripe-based  connected components  labelling, Electronics  letters,  vol.  46,  no.  21,  pp.  1434–1436, 2010.</em></p>
<p align="justify"><em><a name="PRED">[9]</a> C. Grana, L. Baraldi, and F. Bolelli, Optimized Connected Components Labeling  with  Pixel  Prediction, in Advanced  Concepts  for  Intelligent Vision Systems, 2016.</em></p>
<p align="justify"><em><a name="MIRFLICKR">[10]</a> M. J. Huiskes and M. S. Lew, “The MIR Flickr Retrieval Evaluation,” in MIR ’08: Proceedings of the 2008 ACM International Conference on Multimedia Information Retrieval. New York, NY, USA: ACM, 2008. [Online]. Available: http://press.liacs.nl/mirflickr/</em></p>
<p align="justify"><em><a name="TOBACCO1">[11]</a> G. Agam, S. Argamon, O. Frieder, D. Grossman, and D. Lewis, “The Complex Document Image Processing (CDIP) Test Collection Project,” Illinois Institute of Technology, 2006. [Online]. Available: http://ir.iit.edu/projects/CDIP.html</em></p>
<p align="justify"><em><a name="TOBACCO2">[12]</a> D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and J. Heard, “Building a test collection for complex document information processing,” in Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 2006, pp. 665–666.</em></p>
<p align="justify"><em><a name="TOBACCO3">[13]</a> “The Legacy Tobacco Document Library (LTDL),” University of California, San Francisco, 2007. [Online]. Available: http://legacy. library.ucsf.edu/</em></p>
<p align="justify"><em><a name="3DPES">[14]</a> D. Baltieri, R. Vezzani, and R. Cucchiara, “3DPeS: 3D People Dataset for Surveillance and Forensics,” in Proceedings of the 2011 joint ACM workshop on Human gesture and behavior understanding. ACM, 2011, pp. 59–64.</em></p>
<p align="justify"><em><a name="MEDICAL">[15]</a> F. Dong, H. Irshad, E.-Y. Oh, M. F. Lerwill, E. F. Brachtel, N. C. Jones, N. W. Knoblauch, L. Montaser-Kouhsari, N. B. Johnson, L. K. Rao et al., “Computational Pathology to Discriminate Benign from Malignant Intraductal Proliferations of the Breast,” PloS one, vol. 9, no. 12, p. e114885, 2014.</em></p>
<p align="justify"><em><a name="FINGERPRINTS">[16]</a> D. Maltoni, D. Maio, A. Jain, and S. Prabhakar, Handbook of fingerprint
recognition. Springer Science & Business Media, 2009.</em></p>
<p align="justify"><em><a name="YACCLAB">[17]</a> C.Grana, F.Bolelli, L.Baraldi, and R.Vezzani, YACCLAB - Yet Another Connected Components Labeling Benchmark, Proceedings of the 23rd International Conference on Pattern Recognition, Cancun, Mexico, 4-8 Dec 2016, 2016</em></p>
<p align="justify"><em><a name="UF">[18]</a> V. Oliveira and R. Lotufo, A study on connected components labeling algorithms using GPUs, in SIBGRAPI. vol. 3, p. 4, 2010.</em></p>
<p align="justify"><em><a name="OLE">[19]</a> O. Kalentev, A. Rai, S. Kemnitz, R. Schneider, Connected component labeling on a 2D grid using CUDA, in Journal of Parallel and Distributed Computing 71(4), 615–620, 2011.</em></p>
<p align="justify"><em><a name="BE">[20]</a> S. Zavalishin, I. Safonov, Y. Bekhtin, I. Kurilin, Block Equivalence Algorithm for Labeling 2D and 3D Images on GPU, in Electronic Imaging 2016(2), 1–7, 2016.</em></p>
<p align="justify"><em><a name="DLP">[21]</a> L. Cabaret, L. Lacassagne, D. Etiemble, Distanceless Label Propagation: an Efficient Direct Connected Component Labeling Algorithm for GPUs, in Seventh
International Conference on Image Processing Theory, Tools and Applications, IPTA, 2017.</em></p>
<p align="justify"><em><a name="KE">[22]</a> S. Allegretti, F. Bolelli, M. Cancilla, C. Grana, Optimizing GPU-Based Connected Components Labeling Algorithms, in Third IEEE International Conference
on Image Processing, Applications and Systems, IPAS, 2018.</em></p>
<p align="justify"><em><a name="DRAG">[23]</a> F. Bolelli, L. Baraldi, M. Cancilla, C. Grana, Connected Components Labeling
on DRAGs, in International Conference on Pattern Recognition, 2018.</em></p>
<p align="justify"><em><a name="BUF_BKE">[24]</a> S. Allegretti, F. Bolelli, C. Grana, Optimized Block-Based Algorithms to Label Connected Components on GPUs, in IEEE Transactions on Parallel and Distributed Systems, 2019.</em></p>
<p align="justify"><em><a name="YACCLAB_JRTIP">[25]</a> Bolelli, Federico; Cancilla, Michele; Baraldi, Lorenzo; Grana, Costantino "Towards Reliable Experiments on the Performance of Connected Components Labeling Algorithms" Journal of Real-Time Image Processing, 2018.</em></p>
<p align="justify"><em><a name="Spaghetti">[26]</a> Bolelli, Federico; Allegretti Stefano; Baraldi, Lorenzo; Grana, Costantino "Spaghetti Labeling: Directed Acyclic Graphs for Block-Based Connected Components Labeling" IEEE Transactions on Image Processing, 2019.</em></p>
