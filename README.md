# AdaQSPR-II
AdaQSPR-II is a QSPR model for epoxy resins, which can quickly and accurately predict their thermal, mechanical, and electrical properties. AdaQSPR-II is an improved version of AdaQSPR (https://github.com/luo-junyu/AdaQSPR), with an expanded dataset and optimized model architecture and algorithm.


**Project Introduction**

AdaQSPR-II is a transfer learning-based 3D molecular representation learning framework for predicting the comprehensive properties of epoxy resins. This framework can efficiently predict key performance indicators such as glass transition temperature (Tg), initial thermal decomposition temperature (Td5%), tensile strength (TS), relative permittivity (ε), dielectric loss (tanδ), and electrical breakdown strength (Eb).


**Project Structure**

- unimol\_tools is the folder containing the environment packages required to run the program

- AdaQSPR-II\_domain is the domain-specific adaptation component of the model<br>
    The dataset folder contains the organic molecular structure-property datasets required for domain-specific adaptation<br>
    main.py is the program code used for domain-specific adaptation training

- AdaQSPR-II\_task is the task-specific adaptation component of the model<br>
    The `domain\_weight` file contains the weights for the domain-specific adaptation component. These are transferred as the initial weights for task-specific adaptation and must be downloaded from the provided link<br>
    The `exp` file contains the weights obtained from task-specific adaptation training and can be downloaded from the provided link<br>
    The `train\_dataset` folder contains the epoxy resin structure-property dataset used for training<br>
    The three `predict\_results\_...`  folders contain the model’s performance prediction results for the training and validation sets, the test set, and 534 candidate dynamic disulfide-bonded epoxy resins<br>
    `train.py` is the program code used for model training<br>
    The three `pred\_... .py` files are the example codes used for model prediction<br>


**Datasets**

This project contains two datasets:<br>
Organic molecular structure-property dataset: Contains property data of over 150,000 organic molecules<br>
Epoxy resin structure-property dataset: Contains experimental data of over 2800 epoxy resin macroscopic properties<br>


**Model Framework**

The AdaQSPR-II model is based on transfer learning and domain-specific adaptation design, mainly consisting of two parts:<br>
- Domain-specific Adaptation: Based on the Uni-Mol framework, pretrained on the organic molecular structure-property dataset<br>
- Task-specific Adaptation: Fine-tuned on the epoxy resin structure-property dataset to predict six macroscopic properties of epoxy resins. Compared to AdaQSPR, AdaQSPR-II introduces graph models in this part, enhancing the representation and learning of cross-linked structures of epoxy resins.<br>


**Application Case**

The research team used the AdaQSPR-II model to screen 534 different dynamic disulfide epoxy vitrimers and ultimately selected 2 promising candidates for experimental verification. The experimental results proved that these 2 disulfide epoxy vitrimers have excellent comprehensive properties, including:

- Glass transition temperature Tg > 105 °C<br>
- Initial thermal decomposition temperature Td5% > 280 °C<br>
- Tensile strength TS ≥ 60 MPa<br>
- relative permittivity ε < 5.10<br>
- dielectric loss tanδ < 0.0136<br>
- Electrical breakdown strength Eb > 33.5 kV/mm<br>
- Good repairability and degradability<br>


**Weights**

We open-source the weights of the AdaQSPR-II model for reproducibility.

Link: 

Please download the weights of Uni-Mol2 from the origin Repo Uni-Mol2.


Usage/Installation

- Step 1: Install the environment and Python packages required for Uni-Mol2 and unimol\_tools
See https://github.com/deepmodeling/Uni-Mol/tree/main/unimol\_tools

- Step 2: Install unimol\_tools
See https://github.com/deepmodeling/Uni-Mol/tree/main/unimol\_tools

- Step 3: Replace the unimol\_tools folder installed in your environment in Step 2 with the unimol\_tools folder from this project, and download the Uni-Mol2 weights to the weights folder within the replaced unimol\_tools folder. Path: unimol\_tools/weights

- Step 4: Train the AdaQSPR-II model (using AdaQSPR-II\_task/train.py); Or directly download the already trained AdaQSPR-II weights, and refer to the prediction program examples (AdaQSPR-II\_task/pred\_train\&val.py, AdaQSPR-II\_task/pred\_test.py, or pred\_disulfide.py) to complete the epoxy resin properties prediction task.


**Acknowledgements**

We would like to thank the following projects for their contributions to this work:
Uni-Mol
https://github.com/deepmodeling/Uni-Mol


**Contact**

If you have any questions, please contact us at: cheng-handsome@stu.xjtu.edu.cn


**Citation**

If our work has been helpful to you, please consider citing it.<br>

Development of Disulfide Epoxy Vitrimers via Transfer Learning: Bridging the Gap between Excellent Comprehensive Properties and Dynamic Cross-Linking Networks.<br>

published in Macromolecules 2025. <br>
https://doi.org/10.1021/acs.macromol.5c02077

