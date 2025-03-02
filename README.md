# Knowledge Distillation Application
This codebase is for the barebones demonstration of distilling knowledge between two neural networks. Utilizes concepts and equations from **Distilling the Knowledge in a Neural Network** (Hinton, Vinyals, Dean). Also using Samson Zhang's youtube channel for the neural network structure.

## Running the demo
1. Running **BigModel.py** will train the cumbersome model, and store the values into the **teacher.txt** file. After training you can input integers to choose specific training cases to test.

2. Afterwards running **KnowledgeDistill.py** will extract those teacher values from the text file to use to train the student model. You can also input integers to select test cases  after training.

3. (Optional) Running the **MyModel.py** will train a model with the same complexity of the student, but without distilling from the teacher and is trained normally. You can use this to compare performance against the distilled model.

## Reference Materials
- Distilling the Knowledge in a Neural Network, Hinton, Vinyals, Dean
- Samson Zhang's tutorial: https://www.youtube.com/watch?v=w8yWXqWQYmU
- **train.csv** is a MNIST data set found on kaggle.com