# Financial Product Fraud Detection Tool
This is a simple financial product fraud detector that uses self-organizing maps, a type of artificial neural network that is trained using unsupervised learning.

## Why unsupervised learning?

Let's assume you are a deep learning scientist working in the fraud detection department, 
you are given a data set that contains information of customers that applyed for a loan. Sensitive information has been anonymized, the number of customers is big and the dataset contains a bunch of linear and non-linear varibles. We won't be able to create a model to predict if each customer potentially cheated and we don't have previous information about customers having or having-not cheated. So, this means no labels to teach the machine about expected results , no binary dependent variable, and  categorical and continuous independent variables. All this would make this problem very complex for a human and unsupervised learning seems perfect for this type of task.

We'll provide the machine a lot of high dimensional data and it will draw conclusions from the non-linear relationships it discovers in the data. One of this patters will be potential fraud. 



## Intuition behind SOM for fraud detection


### First let us define fraud. 

According to the Cambridge Dictionary, "Fraud" is defined as "something that is not what it appears to be and is deliberately used to deceive people, especially to get money."

So... basically we are spotting customers that are claiming to be someone that they're not.


### How to classify the customers?

 All these customers from our dataset and they're answers or variables (1 identificator and 14 other attributes) are the inputs of our neural network.

These input points are going to be mapped to a new output space. As a result we will have a neural network composed of neurons where each neuro is being initialized as a vector of weights that is the same size as the vector of a customer (a 15 element vector).

We will the have one neuron or winning node for the closest vector or customer. The winning node is the most similar neuron to the customer, this is practical as we can use a neighborhood function to update the weight of the neighbors of the winning node to reduce the distance between them and separate them further from other winning nodes. We'll do this with all the observations or customers and we will repeat this over and over many times in order to decrease the output space and reduce dimensions until the neighborhood and the output space stops decreasing. In the end we'll obtain 
our self organizing map in two dimensions with all the winning nodes that were identified

### Fraud identification

Once our two-dimensional SOM has classified all the observations we will be able to identify the outlying neurons, those that differ most from the rest. Why are we interested in the outliers? As we mentioned it previously fraud can be understood as the attempt to be something or someone that is not, therefore customers that are attempting to commit fraud will lie and provide false information deviating from the general rules or from the majority of honest customer profiles in their neighborhoods.

### Outline detection

In order to detect the outline neurons in the self organizing map we need to calculate the Mean Interneuron Distance.

This means that for each neuron in our SOM we're going to compute the mean of the Euclidean distance between a neuron and all the neurons in its neighborhood. We will then manually define the neighborhood (one neighborhood for each neuron) and we'll compute the mean of the Euclidean distance between this neuron that we picked and all the neurons in the neighborhood that we defined.

Doing this will allow us to detect outliers (thoae far from all the neurons in their neighborhoods).

### Potential fraudster identification

Finally, in order to identify which customers originally in the input space are outliers associated to this winning node, we'll use an inverse mapping function.


## Dependencies

Numpy
Matplotlib
Pandas
MiniSom


## Implementation 

Import data set (you can find it in the repository or download directly from(UCI Machine Learning Repository)[http://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29])

* Import the libraries

* Import the application status dataset

* Split the data sets into two subsets:

    The first one is the set that contains all the variables from customer ID to attribute 14 and we'll call this subset "X".

    The second one will be the class that is the variable that tells if the application of the customer was approved (0 for no and 1 for yes). This way we'll be able to identify from the SOM fraudulent customers with approved applications. Nevertheless, we will only use X to train our model since we are using unsupervised learning, therefore our dependent varibale y is used only in the end to figure out who of the potential fraudsters had his or her application approved.

    Since we are dealing with a high dimensional data set containing many non-linear relationships we need to use feature scaling before we can proceed to train our model. In this case we will use normalization in order to get all our features between 0 and 1.

In order to train our SOM we can either implement it from scratch or we can use available classes from other data scientists, which is what I'm using to save some time. [JustGlowing](https://github.com/JustGlowing/minisom) has an excellent numpy based implementation available in his repository under the Creative Commons 3.0 license.

* import MiniSom class (double check that you have the file "MiniSom.py" in the working directory folder where you have the dataset)

* create an object of this class and set the arguments (x and y are the dimensions of the SOM grid, the bigger x and y the more accurate the self-organizing map classification and identification of outliers will be, the second argument is the number of features in subset X). For the amount of observations in the dataset a grid of 10 by 10 has been chosen.

    dimensions of the grid
    input link = number of features of subset X
    Sigma = radius of the neighborhoods in the grid
    learning rate = increase for faster convergence
    Decay = to improve convergence if necessary

* Create object

* Initialize weights

* Train SOM

* Plot the SOM to visualize the results (see python file for specifications)

* Extract the potential fraud cases.

    I used a method available in minisom.py that allows us to obtain all the mappings from the winning nodes to then use the coordinates of the outliers we identified in the previous step (those with the highest MID)


* Inverse transform to reverse the scaling and you'll have the list of potential frauds including those who got their application accepted.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [JustGlowing](https://github.com/JustGlowing/minisom) for his excellent numpy based self-organizing map implementation.
