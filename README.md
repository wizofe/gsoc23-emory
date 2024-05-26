# Google Summer of Code 2023 - Progress Log

Hello there! My name is Ioannis Valasakis, and I am thrilled to be a participant in Google Summer of Code 2023 for EMORY. This repository houses my work-in-progress, a GUI-based PyTorch model trainer for different dataset, in order to provide a workflow which is transparent and easy to train models. In this README, I'm going to walk you through my journey over the past weeks, the progress I've made, the challenges I've faced, and the lessons I've learned.

Here's the [User Documentation](user-doc.md) and the final [report](final-report.pdf) of the project!!


## Week 1 - Beginning of June 2023

This week was primarily about setting the foundation for my project. I started with setting up the Anaconda environment and installing PyQt5, a fantastic library that allows one to create GUIs in Python. This was a new territory for me, but the process was surprisingly smooth.

After setting up the environment, I set my sights on creating the basic structure of my project. I decided to use PyTorch for defining and training my machine learning models. I started simple, creating a straightforward neural network with a single hidden layer.

I also laid the groundwork for GUI design, including creating the main window and controls. At this point, the controls didn't do anything just yet, but it was a start!

## Week 2 - Second week of June 2023

The second week saw significant progress. I fleshed out the PyTorch model, adding more functionality and testing out the training on the MNIST dataset. The training process was a bit slow at first, but after some adjustments to the learning rate and batch size, it started to perform better.

On the GUI side, I implemented the logic behind the controls. Now, the GUI can load the dataset, select the model to train, and initiate the training process! However, the training process was still blocking the main GUI thread, which was next on my list to address.

## Week 3 - Third week of June 2023

This week was all about improving the user experience. I managed to offload the training process to a separate thread, which now allows the GUI to remain responsive during the training. This was a bit tricky due to some intricacies of threading in Python and PyQt, but I managed to get it working.

I also added the functionality to monitor the training progress from the GUI. Now, when the model is training, the GUI shows the loss and accuracy values updating in real-time!

## Week 4 - Last week of June 2023

This week I delved into experimentation with 1D time series data. I used a different dataset for this purpose, and started by adapting the existing model to work with the 1D data. This was a challenge in itself as working with 1D time series data requires a different approach compared to image data.

I explored various model architectures including LSTM and GRU, which are more suitable for time series data. I added these models to the application and made necessary changes to the data loading and processing code.

The results have been promising, but there's still more to explore and refine.

# Progress log report

## First Week of June 2023

- Initial setup of the project
- Created a GUI with PyQt5
- Integrated basic model training and evaluation functionality with PyTorch

## Second Week of June 2023

- Added a feature to choose between different neural network models (Net, ConvNet)
- Implemented data loading functionality to load time-series datasets from an online repository

## Third Week of June 2023

- Introduced new models and experimented with 1D time-series datasets
- Enhanced the training functionality with real-time progress updates on the GUI

## Fourth Week of June 2023

- Improved application's robustness by adding test cases
- Started working on parallel training functionality for better performance

## Commencing First Week of July 2023

- Testing the parallel training functionality using PyTorch's DataParallel
- Introduced model persistence to save and load trained models
- Included functionality to visualize model performance and results

## Key Features

- User-Friendly GUI: Interactive interface to manage the entire workflow.
- Multiple Model Support: Choose from different neural network models like Net and ConvNet.
- Real-time Updates (testing): See the training progress in real-time.
- Model Persistence (draft - to test): Save and load models for future use.
- Data Visualization (testing): Plot and analyze model performance.
- Parallel Training (draft): Efficiently utilize all available GPU resources.

## Moving Forward

As I am moving into the second month of Google Summer of Code 2023, my plans are to continue refining the application, working on improvements, handling edge cases, and of course, squashing any bugs that pop up. I will also be spending more time understanding and refining the performance of time series models.

This journey so far has been a blend of challenges and learning, and I am super excited to see where the next month will take me. Stay tuned for more updates!

Cheers,
Ioannis Valasakis

Note: This is a work-in-progress as part of Google Summer of Code 2023. The code and features are still under development and may change as the project progresses.
