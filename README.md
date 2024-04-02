# Intel-Hackathon 
Team Name: Turnings   

Team Leader Email:vishalkumarp211102@gmail.com

Team Members
- Dev vidit
- Bikash Baruah
  
Empowering Efficiency: Real-Time Building Estimation with Computer Vision
```bash
give link of hosted repo on Hugging Face hub
```

# Problem Statement
The task at hand involves creating an innovative, AI-powered solution using computer vision to track energy consumption within buildings in real time, with a focus on monitoring consumption on a floor-by-floor basis. This solution aims to utilize advanced algorithms and machine learning techniques to analyze video frames captured by cameras installed within buildings. By leveraging computer vision capabilities, the system will be able to identify and quantify energy consumption patterns within the camera frame, providing insights into the usage of electricity, heating, and cooling systems across different floors of the building. The proposed solution will enable facility managers and building operators to gain a comprehensive understanding of energy usage within their facilities, allowing timely interventions to optimize energy efficiency and reduce costs. Additionally, by providing real-time feedback on energy consumption, the system can help identify potential anomalies or inefficiencies, enabling proactive measures to be taken to address them promptly. This solution has the potential to revolutionize energy management practices in buildings, enabling efficient resource allocation and contributing to sustainability efforts.

# Intel One API AIAnalytics toolkit-Boon for Developers ![all text](https://github-production-user-asset-6210df.s3.amazonaws.com/75485469/288444454-c4da56ab-906a-4aa3-b3cd-47f93e3f7b59.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240402T075950Z&X-Amz-Expires=300&X-Amz-Signature=06f10f92f6e4c49c91d701662dfa30ed14e44eef6281372e82c588a15a1ac2c6&X-Amz-SignedHeaders=host&actor_id=124500092&key_id=0&repo_id=726910194)
The main goal of our project is to detect different electrical appliances by our model and give us about power consumption in building and to optimize power consumption by giving suggestions.During training process, we were impressed by the efficiency of Intel Developer Cloud, particularly the impactful performance of AI Analytics Toolkit. The optimization of image processing speed on Intel GPU significantly spped up our training process.

# Description
We collect data from [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) and cleaning it and make annotation from [CVAT](https://www.cvat.ai/) in xml format and build our Faster R-CNN model from scratch in which we use ResNet-50 as convolutional backbone.Extensive training was conducted using a substantial image dataset gathere from Open Images Dataset V7. This exhaustive training equipped the model with comprehensive detection of "ON" and "OFF" electrical appliances like Laptops, PC/Monitor, AC. 

We can use pretrained ResNet-50 by following python code.
``` python
model = torchvision.models.resnet50(pretrained=True)
```

# Intel AI analytics toolkit in Data preprocessing
The synthetically collected dataset have to extensively preprocessed before sending to our model for training.Here we have used Intel Optimized Numpy to handle image matrix. This have improved the speed to multifold, made even CPU computation so powerful .It made the program utilize all the cores in CPU instead of leaving them idle. just change in a line have improved our efficiency multifold.

# Intel AI analytics toolkit in training 
![all text](https://www.intel.com/content/dam/www/central-libraries/us/en/images/2023-03/boxshot-base-toolkit-oneapi-rwd.png.rendition.intel.web.864.486.png)

In training process, we utilized the same foundational model, enriching it with  comprehensive detection of "ON" and "OFF" electrical appliances like Laptops, PC/Monitor, AC.We implement our model using idc and colab attached. We implement our deep learning model using optimized Intel PyTorch significantly enhanced code optimization. We train our model on Intel GPU. Despite the unfortunate loss of our trained model due to a system crash at IDC, the evident reduction in training loss underscored the success of our efforts.

# Usecase of Intel Developer cloud
The Intel Developer Cloud proves to be an excellent platform, offering access to powerful CPUs and high-speed internet, thereby facilitating a remarkably swift process. This challanges the misconception that training with images takes long time as Intel GPU optimizes images batches efficiently, ensuring faster processing and significantly reducing training times. The experimentation phase demonstrated that faster inferencing and training are achievable.

# Final Output
Our model is proficient in detecting "ON" and "OFF" electrical appliances and calculate power consumption in a office building and suggesting how to optimize power consumption in bilding.As we train our model on image data gathered by us we use Intel® Max Series GPU which very fast, it optimizes images batches efficiently, ensuring faster processing and significantly reducing training times. We show you some of our result detected by our model.

![model_performance](https://github.com/Vishalkumar158/Intel_Ai_Hackathon/assets/124500092/69436ccc-c6f5-458c-9fd9-4f9eb4c48fac)

![image](https://github.com/Vishalkumar158/Intel_Ai_Hackathon/assets/124500092/7ec3bf63-f197-4dff-bfeb-ac4cf4a46507)

![output_power_consumption](https://github.com/Vishalkumar158/Intel_Ai_Hackathon/assets/124500092/8bbfbd16-e0e7-4f01-9c00-4606a4fc7565)

We use table of power consumption as given on [altE](https://www.altestore.com/diy-solar-resources/power-ratings-typical-for-common-appliances/), an organization for making renewable resources.

We can also download [weight](https://drive.google.com/drive/folders/1jZIjqxenq5VNlMTWzD_O7enLqFdOWyl8?usp=sharing) file here. 

# Future Scope
- To enhance performance through increased computational capacity, we aim to construct an expansive dataset for the aforementioned use cases. This augmented dataset will serve as the foundation for retraining a more robust detection model, enabling superior capabilities. The intention is to leverage advanced computing power to refine and elevate the model's proficiency in generating more accurate real time inference.
- We embarked on retraining the model using Intel frameworks and incorporated quantization with NNCF. This approach yielded improved results, showcasing the model's enhanced performance. However, a setback occurred as our session expired, preventing us from saving the valuable progress made during this training endeavor. Despite this challenge, the discernible advancements in model performance underscored the effectiveness of the adopted methodologies.
- We deploy our model in AI-enabled camera for better real time inference as suugested in this research paper [A Computer Vision-Based Occupancy and Equipment UsageDetection Approach for Reducing Building Energy Demand](https://www.mdpi.com/1996-1073/14/1/156)
- We enhance our approach by integrating it with the [GCN Depth model](https://arxiv.org/pdf/2111.01715), enabling more accurate distance calculation between individuals and electrical appliances. This integration allows us to optimize our decision-making process, determining which appliances should be turned 'ON' or 'OFF' more effectively.
- We integrate our model with IOT so that it could optimize energy usage by itself.

# Learning and Insights
- Specialzed CV focus:- Expanding expertise in Computer Vision(CV), specifically in detction, tracking object, and estimating depth.
- End-to-End Model Bulding:- Our end-to-end model building process involves gathering, annotating, and cleaning our own dataset. We acquire the necessary skills to preprocess the dataset and construct the model from scratch, leveraging research papers such as [Faster R-CNN](https://arxiv.org/abs/1506.01497).
- Intel Technologies Integration:- Acquiring knowledge about Intel technologies, specifically IDC and one API.  - Experimenting with features like quantization with GPU and Intel-optimized fine-tuning.

# Future Application Enhancement for Intel:
- Recognizing the prospective benefits of integrating Intel's features in future iterations.
- Envisaging heightened end-to-end application performance through the strategic application of recently acquired insights and technologies.

# Tech Stack used 
-------------------------IDC Jupyter Notebook Intel AI analytic ToolKit------------------------

# Conclusion
In conclusion, our model represents a significant advancement in energy optimization within office buildings. By leveraging Intel's optimized PyTorch framework and Intel GPU, we have developed a highly efficient and scalable solution. Through meticulous data collection, annotation, and cleaning processes, coupled with the utilization of cutting-edge technologies such as Faster R-CNN, our model effectively identifies and classifies electrical appliances. This enables precise decision-making regarding which appliances should remain 'ON' or be switched 'OFF', leading to substantial energy savings. The integration of Intel's optimized hardware and software solutions ensures that our model operates with maximum efficiency and performance, making it a valuable tool for enhancing energy management practices in office environments.

# Qucik Steps
Required Intsallation 
``` bash
pip install -r requirements.txt
```
# References:- 
- [A Computer Vision-Based Occupancy and Equipment UsageDetection Approach for Reducing Building Energy Demand](https://www.mdpi.com/1996-1073/14/1/156)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [GCN Depth model](https://arxiv.org/pdf/2111.01715)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- [GCNDepth: Self-supervised monocular depth estimation based on graph convolutional network](https://www.sciencedirect.com/science/article/pii/S0925231222013601?ref=pdf_download&fr=RR-2&rr=86df8f09dd3a9365)


