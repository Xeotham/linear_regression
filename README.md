# Linear Regression.  
Small 42 Project to make our first Linear Regression and Gradient Descent Algorithm.  
  
## Sources
- [An IBM article about what is the Gradient Descent and how it's useful.](https://www.ibm.com/think/topics/gradient-descent)  
- [A good french video to understand how the Gradient Descent work.](https://www.youtube.com/watch?v=rcl_YRyoLIY&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)  
- [A playlist to understand more about Machine Learning. (French)](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)

## Subject
### Objective
The aim of this project is to introduce you to the basic concept behind machine learning.
For this project, you will have to create a program that predicts the price of a car by
using a linear function train with a gradient descent algorithm.
### Mandatory part
You will implement a simple linear regression with a single feature - in this case, the
mileage of the car.
To do so, you need to create two programs :
• The first program will be used to predict the price of a car for a given mileage.
When you launch the program, it should prompt you for a mileage, and then give
you back the estimated price for that mileage. The program will use the following
hypothesis to predict the price :
``estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)``

Before the run of the training program, theta0 and theta1 will be set to 0.

• The second program will be used to train your model. It will read your dataset file
and perform a linear regression on the data.
Once the linear regression has completed, you will save the variables theta0 and
theta1 for use in the first program.

You will be using the following formulas :
``tmpθ0 = learningRate ∗ (1/m) m−1∑ i=0 (estimatePrice(mileage[i]) − price[i])``

``tmpθ1 = learningRate ∗ (1 m) m−1∑ i=0 (estimatePrice(mileage[i]) − price[i]) ∗ mileage[i]``

I let you guess what m is :)

Note that the estimatePrice is the same as in our first program, but here it uses your temporary, lastly computed theta0 and theta1.

Also, don’t forget to simultaneously update theta0 and theta1.

## Linear Regression
### Concept / Uses:
The **Linear Regression** is a **Machine Learning** use to predict outcome based on a DataSet.
It's one of the most **simple**, **easy to implement and to visualize** Machine Learning Algorithm.
The goal is to create the **Best-Fit Line**, which is a straight line that **minimizes** the error (the difference) between the **observed data** points and the **predicted values**.
This line helps us **predict** the dependent variable for **new data**.

![Lightbox](https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png)

## Project
### 1- The DataSet:
A **DataSet** is a set of data used to **train** the **Linear Regression**.

The provided DataSet is:
|km|price|
|--|--|
|240000|3650|  
|139800|3800|
|...|...|
|22899|7990|  
|61789|8290|

It group the **price** of a car based on it's **mileage**.

### 2- The Model:
The **model** is the function used to **predict** the **outcome** of the **Machine Learning Algorithm**.
In our case it's a **Linear Function** (`ax + b`), but it can be a **Polynomial function** (`ax^2 + bx + c`) for exemple.

The Provided model is a **linear function**:
``estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)``
Basically:
``f(x) = ax + b``
Where `ax` is `(θ1 ∗ mileage)` and `b` is `θ0`.

### 3- Cost Function:
The **cost function** is a funtion that serv to calculate the **error** between a **predicted value** and the **real value**.
The lower the **error** is, the **closer** the prediction is to the real value.

In our cas, we use the **Mean Squared Error**:
``J(a,b) = (1/2m) m
i=0∑(f(xi) - yi)^2``
Where `m` is the size of the DataSet, `f()` the previously defined **model**. and `yi` the true result of i on the DataSet.
### 4- Minimization Algorithm:
We use the **Gradient Descent**, like required in the subject.

A **Minimization Algorithm** is an algorithm that will find the `a` and `b` value of the **cost function**, where the result of the function is closer or equal to **0**.

## Gradient Descent
### Explanation:
The **Gradient Descent** is a **Minimization algorithm** which as for goal to find the "**steepest descent**" and find the lowest point or **local minimum**.

The algorithm, find the **steepest descent** or the **gradient** at a given point, and use the **learning rate** to step down on the previously found **steepest descent** and find the new **gradient**.

![undefined](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/1024px-Gradient_descent.svg.png)

By following those steps, the algorithm will eventually find the **local minimum**, which mean that in our case, when provided the **cost function** will find the case where the **error** is the **closer** to **0**.

The **mathematical formula** is:
``an+1 = an - αPn``
Where `an` is the actual point, `α` the **learning rate** and `Pn` is the derivative of a **multi-variable function**, in our case the **cost function**, to find the **gradient**.

### The Learning Rate:
The **learning rate** is an important part of the algorithm. It's what make the algorithm "**learn**".
It define the "step size" by which the algorithm will descend.

A too big **learning rate** make the steps to big so that the algorithm will always go around the **local minimum** without finding it.

A too small **learning rate** can make a greater precision, but the **computing** will be **too long**.

That's why, finding the right **learning rate** is very important.
![Setting the learning rate of your neural network.](https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png)

### The gradients:
The final step for the **gradient decent** is to find the **gradient** for `a` and `b`.
The **gradient** is found with the **cost function**.

So to find the **gradient** for `a`, you have:
``an+1 = an - αPn``
Where `Pn` is the derivative of the **cost function** `f`:
``Pn = f'(a)``

``f'(a) = (1/2m) m i=0∑(f(xi) - yi)^2``
``f'(a) = (1/2m) m i=0∑(ax + b - yi)^2``
So we do the derivative of `a`:
``f'() = (1/m) m i=0∑x(ax + b - yi)``

And we put it together:
``an + 1 = an - αPn``
``an + 1 = an - α * (1/m) m i=0∑x(ax + b - yi)``

Which was give on the subject as:
``tmpθ1 = learningRate ∗ (1/m) m−1∑ i=0 (estimatePrice(mileage[i]) − price[i]) ∗ mileage[i]``

Where `an+1` is `tmpθ1`,  `α` is `learningRate`, `x` is `mileage[i]` and `b` is `price[i]`.

So as we do the same for the derivative of `f` for `b`, we have the second formula provided in the subject:
``tmpθ0 = learningRate ∗ (1/m) m−1∑ i=0 (estimatePrice(mileage[i]) − price[i])``

We now have all we need to start the project.
