CSCI 4052U  
Machine Learning II  
Final Project

# Project Sign-up

The final project is a collaborative team exercise.

* Teams with a maximum size of 3. Larger teams require explicit approval by the instructor. You need to sign up (for 5% of the credit for the project).  
* Submission URL: [https://forms.gle/7j65dgVhywf184mt6](https://forms.gle/7j65dgVhywf184mt6)

# Project Submission

* Submission URL: [https://forms.gle/7XnYLVP7hqdaXft7A](https://forms.gle/7XnYLVP7hqdaXft7A)  
  * Project ID (provided to you by TA).  
  * Github repository (public repo only)  
  * **PDF of presentation** * A demonstration YouTube video of at most 10 minutes. The video can be a slide show accompanied by a demonstration of the overall AI application. Elements of the video should include the description of the AI problem, elements of the neural network based components, and the deployment of the overall AI system.

# Project Description

This project focuses on neural computation and embedded neural networks in an end-to-end AI application. This means that students are asked to solve problems that used to be very hard (if not impossible) without sophisticated neural networks. Due to the requirement on training data and the training computing requirements, for the purpose of this project, we encourage students to consider pretrained models (e.g. [https://huggingface.co/models](https://huggingface.co/models)).

**The Problem**

You must clearly describe the problem you wish to solve in the project with emphasis on the following:

* Which part of the problem was particularly challenging for traditional approaches.  
* What is the neural network approach to solve this part of the problem?

**The Neural Network Component(s)**

Describe the theoretical design of the neural network component(s)

1. Describe neural network architecture in terms of the elements of ML we have discussed.  
2. Describe how the model weights are obtained. If you are training the neural network, describe the training data and the training loop. If the model is pretrained, describe the training data used for the model. Cite the original research of the network architecture and the training data used in training.  
3. Describe if any fine tuning is to be done on the pretrained model.

**End-to-end Application Pipeline**

Describe the overall software architecture which solves the problem in its entirety.

* Describe how the application data is converted to and from tensor encoding.  
* Describe how the application code interfaces with the neural network components. 

**Deployment**

You are expected to develop fully functional applications. Consider the following deployment runtime environments:

* Local deployment on your own GPU machine. This is possible if your machine has a GPU with >10GB VRAM. In this case, you are free to utilize any application development stack.  
* Google Colab offered (time constrained) T4 GPU runtime with 16GB VRAM, which is large enough to run many models (YOLO, LLM, whisper). However, the runtime environment is limited to a Google colab notebook. Consider using Gradio ([https://www.gradio.app/](https://www.gradio.app/)) or Mesop ([https://github.com/mesop-dev/mesop](https://github.com/mesop-dev/mesop)).

**Github Repository**

The github repository must have the following:

* Complete source code for all your implementations.  
* README.md must contain the following:  
  * Setup instructions to run the application (e.g. link to Google Colab)  
  * Theoretical and software engineering discussion on the design of the neural network and software components.  
  * Screenshots/video on the running of the application

**Presentation**

You must prepare three forms of presentation:

1. A public github repository containing the source code for your entire application.    
2. A presentation (in PDF format) on the project.  
3. A YouTube video (max 10 minutes) based on the presentation but with additional narrative and demo of the application. IMPORTANT: make sure you showcase the application and the github repo in your video.

**Submission**

You are to submit your work via the following Google Form.

You are required to enter:

1. Project ID (provided to you)  
2. Source code (in a github repository)  
3. Upload presentation slides (PDF export)  
4. Specify the presentation video URL (youtube or Google drive)

The grading rubric is:

| Signup on time | 5% |
| :---- | ----: |
| Problem formulation | 15% |
| Neural network models | 20% |
| AI integration | 20% |
| Github repo quality | 20% |
| Presentation slide quality | 10% |
| Video presentation quality | 10% |