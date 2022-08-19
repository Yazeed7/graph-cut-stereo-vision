# Graph-Cut Stereo Vision

This project implements the Stereo Vision Graph Cut algorithm as described in [Boykov et. al.'s graph cut method](https://doi.org/10.1109/34.969114). It infers the depth of objects by computing the shift in pixels between images from slightly different perspectives. Below are the results:

![cv_proj](https://user-images.githubusercontent.com/61038456/185650236-820ec669-83c3-4f83-ba75-6d840831db93.png)

To test the code of this project, in an anaconda terminal, navigate to the project’s main directory ‘final_project_stereo’ and run the follow command to create the required environment:

> conda env create -f cv_proj.yml

Activate the environment:

> conda activate cv_proj

Run the test code:

> python testproject.py

Note that this will roughly take 45 minutes to compute. Once the code has finished running, you will find the output plots as well as all recorded statistics in the results folder.
