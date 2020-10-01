# SPH Fluid Simulation
This repository contains my implementation of the paper [Position Based Fluids](http://mmacklin.com/pbf_sig_preprint.pdf) by Macklin et al. To achieve real time performance of ~80,000 particles I parallelized the algorithm using Cuda along with a spatial acceleration structure. 

## Results
Scene with 40,000 particles running at 60 fps on my laptop (GTX 1650). The particles are spawned in a random distribtution throughout the bounding box.

<img src="gifs/Standard.gif" width="640" height="360" />

Dam break scenario with 80,000 particles running at 60 fps on my laptop

<img src="gifs/DamBreak.gif" width="640" height="360" />

This clip demonstrates an interesting phenomena that occurs when you spawn all 80,000 points within a narrow radius (0.5 scene units). The density constraint blows up resulting in an explosion effect. Notice how the system still tends towards equillibrium even after blowing up.

<img src="gifs/Blowup.gif" width="640" height="360" />

## Quick Code Walkthrough
```main.cpp``` contains OpenGL setup, the rendering loop and also calls the FluidSimulator class.

```FluidSimulator.cu``` is where the main simulation kernels are implemented. ```computeDensity()``` and ```projectDensityConstraint()``` are the two main SPH functions. ```computeDensity()``` implements equations (9) and (11) in the paper while ```projectDensityConstraint()``` implements (12), both in parallel.

```LoadShaders.cpp``` and ```camera.h``` are both utility files with appropriate credit given to their sources at the top of each file.

## Building
You will need GLEW, GLFW, GLM, and at least Cuda 10.

In the root folder run

```mkdir build```

```cd build```

```cmake ../```

```make```