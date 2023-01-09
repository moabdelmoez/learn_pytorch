> *Note: This notebook is a takeaway from the amazing course https://learning.edx.org/course/course-v1:LinuxFoundationX+LFS116x+3T2022/home

# Chapter 1

PyTorch is an open-source deep learning framework developed by Meta AI in 2016. It was designed to be flexible and modular, thus allowing the rapid prototyping and experimentation required by cutting-edge research, while providing the stability and support required by the industry.

PyTorch, a deep learning framework, can be used to automate and optimize processes through the development and deployment of state-of-the-art AI applications.

why getting the right data should be the top priority for any AI project.

All of a sudden, tasks that could only be performed by humans, like visually inspecting items in a production line for signs of imperfections or defects, can be more easily automated. There was no need for a research team and five years’ time anymore. The main challenge, under the new paradigm, is collecting enough high-quality data to use in the training process and correctly labeling it.

![image](https://user-images.githubusercontent.com/37369603/211324920-11a9c437-2c7c-431a-8f9f-06778fa27ecb.png)

> References
* https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-executives-ai-playbook

* https://www.mckinsey.com/capabilities/quantumblack/our-insights/global-survey-the-state-of-ai-in-2021

* Data never sleeps https://www.domo.com/learn/infographic/data-never-sleeps-9

Generative models, like LaMDA, are trained to, as their name says, generate data that resembles the data they were trained on when prompted. To put it in an oversimplified way, these models learn how to "fill in the blanks" in sequences of words. In many other cases, though, raw data alone is not enough, and you need the corresponding labels too.

**Software 2.0**

![image](https://user-images.githubusercontent.com/37369603/211325099-5c21b113-d8fc-40b6-8465-99fee0d6751b.png)

![image](https://user-images.githubusercontent.com/37369603/211325122-8b3ab878-355a-4ef1-b1cf-b54bd0657742.png)


**Deployment**

* TorchScript and PyTorchJIT
PyTorch’s celebrated "eager mode", responsible for its ease of use and quick prototyping and experimentation, was also its Achilles’ heel when it came to deployment in production.

* TorchServe
TorchServe, a collaboration between AWS and Meta, is a lightweight server capable of high-performance inference with low latency. In a nutshell, TorchServe wraps a PyTorch model in a set of REST APIs so that the client application can easily interact with and send requests to it.

**Edge Deployment**

Edge deployment, or edge computing, is defined as "part of a distributed computing topology where information processing is located close to the edge, where things and people produce or consume that information" by Gartner. The general idea is to bring data storage, and computation, closer to the devices.
The benefits are many: increased data privacy, lower latency, and lower costs (both centralized computing power and bandwidth). But it also comes with several challenges: decentralized applications are harder to maintain and monitor, and devices on the edge have limited computing power and are notoriously insecure.

PyTorch Mobile, introduced in 2019, provides an end-to-end workflow to develop and deploy models for mobile devices (it is available for iOS, Android, and Linux). It leverages the power of TorchScript to optimize and compile a model before porting it to either an Android or iOS application.

 PlayTorch app to rapidly create mobile AI applications. It supports many models, including some of the computer vision models for image classification

You're Not Locked In!
ONNX
Ivy

**What about TensorFlow?**

On the one hand, PyTorch has a more well-designed interface (API) for development, while TensorFlow’s API is still considered somewhat confusing. On the other hand, TensorFlow is more mature than PyTorch when it comes to deployment in production. For this reason, and given the interoperability between frameworks provided by tools such as ONNX and Ivy, we’re briefly discussing TensorFlow’s alternatives for deployment: TensorFlow Serving and TensorFlow Lite.
>* *TensorFlow Serving is the counterpart to TorchServe*
>* *TensorFlow Lite is the counterpart to PyTorch Mobile*

# Chapter 2

Models are only as good as their input data: **"garbage in, garbage out"**, as the expression goes.

https://www.gartner.com/smarterwithgartner/how-to-improve-your-data-quality

If your data is bad, your machine learning tools are useless.

High-quality data is connected, accurate, relevant, and enough to work with. The first two attributes, connected and accurate, describe the quality of the data at collection/acquisition time. The other two attributes, relevant and enough to work with, describe the quality of the data as it is used in conjunction with a model.

![image](https://user-images.githubusercontent.com/37369603/211325645-edc68b80-2147-4eb0-87a8-bbd6f4d43adc.png)


Labeling: Annotating Images
image classification 
object detection
semantic segmentation
pose estimation

There are several approaches to the labeling/annotating process:
* manual annotation, where the annotator has to draw regions and assign labels to each image
* programmatic or semi-automated annotation, where tools are used to automate the selection of regions/pixels which can be corrected by the annotator if needed
* synthetic labeling, where images corresponding to the required labels are synthetically generated (e.g. generative models)

Data labeling is a time-consuming and expensive activity because, in most cases, it requires "human-in-the-loop" (HITL) participation.

Data labeling is an integral part of application development since it is required to train most models: we need to provide them with both input data and expected output so that they can learn the underlying association rules by themselves. That’s "Software 2.0" in a nutshell.

Data augmentation is a clever technique used to incorporate more data in a dataset used to train a model without having to actually collect more data. 

# Chapter 2

Complex models may fail to deliver on other requirements such as latency, explainability, and interpretability.

Andrew Ng recommends that the first AI project of a company should have five important traits, in order to gain momentum and firsthand knowledge:

	1. the project should be a quick win (6-12 months to completion)
	2. the project shouldn't be too trivial (otherwise it won't have an impact) or too ambitious (otherwise it may fail for taking too long or being too expensive)
	3. the project is industry-specific, so internal stakeholders are more invested in it and can clearly see the value it delivers
	4. the project leverages the expertise of credible external partners to speed-up development
	5. the project actually creates value, either by reducing costs or increasing revenue

To address the divide between the promises of AI applications, and the disappointing results experienced by many decision-makers that expected to fix business processes using AI:

	1. The AI application must match the desired business-related outcomes
	2. Use external data to amplify the business impact
	3. Conquer complexity by breaking down the AI model into its smallest parts
	4. Machine learning should help in making concrete business decisions
	5. Avoid machine learning outcomes that seem accurate but may not prove useful

# Chapter 3

Transfer learning is one of the main drivers of the commoditization of deep learning models. It leverages the power of pre-trained models to deliver better performance from the get-go (when compared to training from scratch) and higher performance potential.

Train-Validation-Test Split
![image](https://user-images.githubusercontent.com/37369603/211325892-828a1a21-85d8-4203-8075-6b9566abd1b6.png)


Once the model is deployed to production, it’s important to keep monitoring it and evaluating its performance over time. Performance is expected to decay, and models may need to be retrained or fine-tuned regularly. 

The evaluation of a model or AI application should not only use the proper metric but also consider the many potential costs and impacts associated with the delivery of wrong predictions after deployment.

Apart from the difference in performance, white- and black-box models also differ greatly in two key aspects: explainability and interpretability.
![image](https://user-images.githubusercontent.com/37369603/211325942-3e5d0065-4d6a-4344-90bd-970a810d9d16.png)


**Interpretability** answers the questions of "why" and "how" a model is producing a given output. Interpretability means that the cause and effect can be determined (the "why"), and it is possible to understand how exactly the model went from one to the other (the "how"). White-box models, since they’re built from first principles, are more interpretable than black-box models.

"Interpretability is the degree to which a human can understand the cause of a decision."  

**Explainability**, on the other hand, only answers "how" a model is producing a given output. Once again, white-box models are more easily explainable, but even black-box models can be explained if you use some clever techniques to peek inside them. It is important to notice that, even if you can explain "how" a black-box model arrives at a given conclusion (e.g., subject A is likely to default), there’s rarely (if ever) an indication of "why" it did it.

![image](https://user-images.githubusercontent.com/37369603/211326029-300ca389-0195-4aa0-b8bd-c8106f53a08b.png)

Unfortunately, interpretability comes at the cost of performance. The figure below illustrates well the trade-off: linear regression, a traditional statistical model, is the "gold standard" of interpretability, but it is also likely outperformed by more sophisticated (and increasingly less-interpretable) machine and deep learning algorithms.

model-agnostic methods of interpretability;
![image](https://user-images.githubusercontent.com/37369603/211326079-7c56c6f3-f5c0-4673-86ff-bf70aea1f98b.png)


Captum, comprehension in Latin, is a library that can be used to understand "how" a PyTorch model is arriving at a given prediction.
https://towardsdatascience.com/what-is-responsible-ai-548743369729

# Chapter 4

The idea behind federated learning is to enable edge devices such as mobile phones to collaboratively learn (train) a model while keeping all the training data on the device.
This is different from edge deployment, where an already trained model was deployed to a mobile phone so predictions were made directly on it. Federated learning brings the training process to the edge as well: the data on the device is used to only slightly improve the current state of the model, and only these improvements (but not the data) are encrypted and sent to a centralized location, where they are averaged with the improvements sent by other users before being stored.

https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

PySyft, developed and maintained by OpenMined, is a library that allows data scientists to work with data hosted on someone else’s server, effectively performing remote data science. It decouples private data from model training using the techniques: federated learning, differential privacy, and encrypted computation.

Model development is not software development. A model is never truly finished. Once it is deployed as a real-world application, its performance immediately starts to degrade, the rate of decay depending on the nature of the problem it was trained to address.

Data and concept drift may be addressed through careful monitoring of the models deployed to production. 

The detection of an anomalous input could trigger an "I don’t know" response from the application before the input is even used to make a prediction using the underlying model. This can be achieved through a second model trained to detect anomalous inputs, such as a variational autoencoder (VAE). An autoencoder is a special kind of model trained to compress an input (e.g., an image) to a sequence of a few numbers, and then use those numbers to reconstruct the input. If the reconstruction is too far off, the input is likely anomalous.

![image](https://user-images.githubusercontent.com/37369603/211326181-cb99214a-a266-4f31-bc78-f614b45774ef.png)


**Monitoring your Model**
"An ounce of prevention is worth a pound of cure." (Benjamin Franklin)
In our case, prevention is done through monitoring and detecting early signs of degraded performance. Unlike typical software-as-a-service deployments, monitoring is not only about the service’s health/uptime status. It goes much deeper than that, as it requires monitoring data and model health as well.

![image](https://user-images.githubusercontent.com/37369603/211326256-21dabb88-f0f1-4054-be8d-3814bddca7de.png)

 Assessing the ongoing performance of a deployed model isn’t a nice to have, but a critical part of the development cycle.
