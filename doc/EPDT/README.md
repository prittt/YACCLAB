## [RLPR - Reproducible Label in Pattern Recognition](https://github.com/RLPR)

<p align="justify">
This README is intended to describe how to configure YACCLAB to reproduce the experimental results reportod in the paper <a href="https://prittt.github.io/pub_files/2021icpr_labeling.pdf">A Heuristic-Based Decision Tree for Connected Components Labeling of 3D Volumes</a>
presented @ <a href="https://www.micc.unifi.it/icpr2020/">ICPR2020</a>.
</p>

<p align="justify">
Results that can be reproduced are the ones reported in TABLE II, TABLE III, and Fig. 6. Experimental results reported in TABLE II and Fig. 6 can be affected by the environment employed for testing the algorithms. In particular, cache size and RAM speed can change absolute results while preserving relative performance. Operative System and compiler are likely to heavily influence the outcome. Numbers reported in TABLE III, instead, should be totally independent from the chosen environment.
After installing the  YACCLAB benchmark (installation guide is reported in the main <a href="https://github.com/prittt/YACCLAB">README.md</a> of this repo), the following additional instructions must be taken into account in order to reproduce the experiments in the aforementioned paper:  /
</p>

<p align="justify">
When configuring the project through CMake the flags <code>YACCLAB_ENABLE_3D</code>, <code>YACCLAB_ENABLE_EPDT_19C</code>, <code>YACCLAB_ENABLE_EPDT_22C</code>, <code>YACCLAB_ENABLE_EPDT_26C</code>, and <code>YACCLAB_FORCE_CONFIG_GENERATION</code> must be enabled in order to set-up the benchmark for 3D algorithms and to include EPDT implementations. The CMake file should automatically find the OpenCV installation path, otherwise, it must be manually specified through the <code>OpenCV_DIR</code> CMake parameter. <code>YACCLAB_DOWNLOAD_DATASET_3D</code> flag must be enabled if the user wants CMake to automatically download the YACCLAB 3D dataset (required to test the algorithms). CMake will automatically generate the <em>C++</em> project for the selected compler.
</p>

<p align="justify">
The <code>config.yaml</code> file must be set as follows (excerpt) to reproduce the experimental results reported in TABLE II. 
<br/>
<b><u>Hint</u></b>: disable `CPU 2D 8-way connectivity` to avoid useless tests.
</p>

```yaml
[...]
CPU 3D 26-way connectivity:
  execute: true
  perform: 
    correctness: false
    average: true
    average_with_steps: false
    density: false
    granularity: false
    memory: false

  algorithms: 
    - EPDT_3D_19c_RemSP
    - EPDT_3D_19c_TTA
    - EPDT_3D_19c_UF
    - EPDT_3D_19c_UFPC
    - EPDT_3D_22c_RemSP
    - EPDT_3D_22c_TTA
    - EPDT_3D_22c_UF
    - EPDT_3D_22c_UFPC
    - EPDT_3D_26c_RemSP
    - EPDT_3D_26c_TTA
    - EPDT_3D_26c_UF
    - EPDT_3D_26c_UFPC
    - LEB_3D_TTA
    - RBTS_3D_TTA
[...]
```

<p align="justify">
The <code>config.yaml</code> file must be set as follows (excerpt) to reproduce the experimental results reported in TABLE III. 
<br/>
<b><u>Hint</u></b>: disable `CPU 2D 8-way connectivity` to avoid useless tests.
</p>

```yaml
[...]
CPU 3D 26-way connectivity:
  execute: true
  perform: 
    correctness: false
    average: false
    average_with_steps: false
    density: false
    granularity: false
    memory: true

  algorithms: 
    - LEB_3D_TTA
    - EPDT_3D_19c_RemSP
    - EPDT_3D_22c_RemSP
    - EPDT_3D_26c_RemSP
[...]
  memory_datasets: ["oasis"]
[...]
```

<p align="justify">
The <code>config.yaml</code> file must be set as follows (excerpt) to reproduce the experimental results reported in Fig. 6. 
<br/>
<b><u>Hint</u></b>: disable `CPU 2D 8-way connectivity` to avoid useless tests.
</p>

```yaml
[...]
CPU 3D 26-way connectivity:
  execute: true
  perform: 
    correctness: false
    average: false
    average_with_steps: true
    density: false
    granularity: false
    memory: false

  algorithms: 
    - EPDT_3D_19c_RemSP
    - EPDT_3D_22c_RemSP
    - EPDT_3D_26c_RemSP
    - LEB_3D_TTA
    - RBTS_3D_TTA
[...]
```
