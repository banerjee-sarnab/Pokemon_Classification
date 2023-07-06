# Pokemon Classification using ResNet-50

This project aims to classify different Pokémon species using the ResNet-50 model. The goal is to achieve high accuracy and F1-score in the classification task by leveraging various techniques such as data preprocessing, data augmentation, feature extraction with ResNet-50, and fine-tuning.

## Project Overview

The main components and achievements of this project are as follows:

1. **Data Preprocessing and Data Augmentation**: The project starts with a small dataset of Pokemons consisting of 4 labels. To enhance the model's ability to generalize, data preprocessing techniques are employed to normalize and standardize the data. Additionally, data augmentation techniques are utilized to artificially expand the dataset by applying transformations such as rotation, scaling, and flipping to the existing images.

2. **Feature Extraction with ResNet-50**: The ResNet-50 model, which is a deep convolutional neural network architecture, is utilized as a base model for feature extraction. By leveraging the pre-trained weights of ResNet-50, the model can extract relevant and discriminative features from the input images.

3. **Fine-Tuning**: Fine-tuning is performed on the ResNet-50 model by training the last few layers of the network on the dataset specific to the Pokémon classification task. This process allows the model to adapt and specialize its features for the specific classification problem.

4. **Achievements**: Through the implemented methodologies, the project achieved a validation accuracy of 0.9643, indicating the model's ability to accurately classify Pokémon species. Furthermore, the project achieved an F1-score of 0.89, which is a measure of the model's precision and recall in classification.

## Usage

To reproduce or build upon this project, follow these steps:

1. **Dataset**: Prepare a dataset of Pokémon images with labeled categories. Ensure that the dataset includes at least 4 different labels.

2. **Data Preprocessing**: Apply data preprocessing techniques such as normalization and standardization to the dataset to prepare it for training.

3. **Data Augmentation**: Utilize data augmentation techniques such as rotation, scaling, and flipping to artificially expand the dataset and improve the model's ability to generalize.

4. **Feature Extraction**: Use the pre-trained ResNet-50 model as a feature extractor. Pass the preprocessed and augmented dataset through the ResNet-50 network to obtain feature representations for each image.

5. **Fine-Tuning**: Perform fine-tuning by training the last few layers of the ResNet-50 model on the Pokémon dataset. Adjust the model's weights to specialize in the classification task.

6. **Evaluation**: Evaluate the model's performance by calculating metrics such as accuracy and F1-score on a validation dataset. Adjust the hyperparameters and fine-tuning process as necessary to improve the results.

## Dependencies

This project requires the following dependencies:

- Python 
- TensorFlow 
- Keras 
- NumPy 
- Pandas 
- Matplotlib 

Please ensure that these dependencies are installed before running the project.

## Cloning the Project

To clone this project and get started, follow these steps:

1. **Clone the Repository**: Open a terminal or command prompt and navigate to the directory where you want to clone the project. Then, run the following command to clone the repository:

   ```
   git clone https://github.com/your-username/pokemon-classification.git
   ```

2. **Navigate to the Project Directory**: Once the cloning process is complete, navigate into the project directory using the following command:

   ```
   cd pokemon-classification
   ```

3. **Install Dependencies**: Before running the project, make sure to install the required dependencies. 

4. **Dataset Preparation**: Prepare your Pokémon dataset by organizing the images into appropriate directories based on their labels. Ensure that you have a sufficient number of images for each label to enable effective training.

5. **Configure Project Settings**: Open the project files and adjust any necessary configurations or hyperparameters to suit your dataset and requirements. This may include modifying data preprocessing techniques, augmentation parameters, or fine-tuning strategies.

6. **Train the Model**: Run the training script to train the Pokémon classification model on your dataset. Depending on the size of your dataset and the complexity of the classification task, training may take some time.

7. **Evaluation and Testing**: Once training is complete, evaluate the model's performance using the validation dataset and calculate metrics such as accuracy and F1-score. You can also test the trained model on unseen Pokémon images to assess its classification capabilities.

You are now ready to explore and utilize the Pokémon classification project! Feel free to make modifications, experiment with different techniques, and adapt the code to suit your specific needs and goals.

## Conclusion

By following the steps outlined above, you can clone this project, set it up on your local machine, and start experimenting with Pokémon classification using the ResNet-50 model. Enjoy exploring the fascinating world of Pokémon and deep learning!
