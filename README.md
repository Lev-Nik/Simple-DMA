# Simple-DMA
a simple Dual Memory Architecture for classifications.

based on the paper Dual-Memory Deep Learning Architectures for
Lifelong Learning of Everyday Human Behaviors written by Sang-Woo Lee1,
Chung-Yeon Lee1, Dong Hyun Kwak2, Jiwon Kim3, Jeonghee Kim3, and Byoung-Tak Zhang1,2.

The Dual Memory is build from a simple CNN for the deep memory and Linear Regression fro the fast Memory

CNN architecture:


![CNN Architecture](https://user-images.githubusercontent.com/70317719/151132651-87b96e45-6fda-4f66-a4c5-5e704a4728f2.jpg)


DMA architecture: 

![DMA](https://user-images.githubusercontent.com/70317719/151132923-90b2d56f-9fdb-4cf7-b43e-3d5678a99607.png)


# The Code:
    main.py - the Dual Memory architecture model training and running.
    CNN-full_dataset.py - running the CNN on the entire dataset inorder to see how well it works.
    CNN_retrainning_model.py - running the CNN on 10 split datasets to see the catastrophic forgetting phenomenon.
    cnn.py - the class for creating a cnn.
    model.py - the class for creating the Dual Memory model.
    Bqueue.py - the class for managing the long term memory.
    Plotter.py - the class for plotting the graphs of the scores from the DMA snd CNN. 

# The libraries we used:
	Tensorflow.keras - for the nueral networks
	sklearn - for the preprocessing of the data, shuffling of the data, splitting the data to train and test
	  	  and for the linear model.
	logging - for writing to the output window
	numpy - for processing the data
	copy - for copying the model
	shutil - for getting the terminal columns.
	tqdm - for printing the process bar in the output window.
	matplotlib.pyplot - for plotting the graphs.
  
  
  # RUN
    In order to run the DMA just open it in pycharm, make sure you have all the libraries installed.

    with the scores of the CNN_retrainning_model.py that you recieve after running it you plot the graph of the catastrophic forgetting
    that we have presented in class - ithe plot happenes automatically.

    with the scores of the main.py that you recieve after running it you plot the graph of the final results that we have presented in class 
    and the scores of the DMA vs the scores of the CNN - the plot happenes automatically.
