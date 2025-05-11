# README CNN_CIFAR

## question 1

Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)

Answer: had problem with getting tensorflow 2.2.0 on M4 MacBook.
I'm pretty sure my MacBook version (2.16.2) caused issue with the loss test returning a eagertensor which the loss tests did not like.
There was one workaround where if at the end of the loss function I could put .numpy() on the return of output, it fixs the loss test but breaks the code when actually tanning.
After some research, I found a fix and added this to the top of the code in assignment.py: tf.experimental.numpy.experimental_enable_numpy_behavior()
This has fixed the issue allowing the tests to pass and all the training of the models to be with the correct accuracies.

## question 2

Q2: Consider the three following 23x23 images of the digit 3. Which neural net is better suited for identifying the digit in each image: a convolutional neural net or a feed-forward (multi-layer linear+ReLU) neural network? Explain your reasoning. (2-3 sentences)
Answer: A feed-forward neural network will take the image, flatten it into a vector of 529 pixels, and learns weights to map this vector to an output. Since it treats the input as a unstructured list, it does not naturally learn the spatial arrangement of the pixels. So as the 3 moves positions, the pixel values move to different parts of the vector, the network then has to learn these separate patterns for each location, making it inefficient and hard to generalize. A convolutional neural network uses the convolutional layers with filters that slide over the image, detecting features of the 3, despite the position of the 3 with the help of weight-sharing and pooling for this translational invariance, making it more effient at detecting the pattern 3 across different positions as it does not need to relearn the pattern. Thus a convolutional neural network is better suited for identifying the digit in each image.

## question 3

Q3: Consider the dataset shown in this scatterplot:
The orange points are labeled with class label 0, and the blue points are labeled with class label 1. Write out a mathematical expression in terms of the inputs, using linear layers and ReLUs, that will correctly classify all of these points. We expect something like output = .. as many expressions/nested expressions as you need.. where an expression can include a literal number such as 3.4, calls to relu(...), x1, x2, and +, *, and > operators. 

For example, this expression does not work but does follow the expected format:
output = 2 * relu(x1) + relu(x1 + x2) > 1

Hint: Use https://tinyurl.com/y5gayl5b and with your mouse hover over the bias/weight edges.

Answer:

Input Layer:
x1
x2

Hidden Layer:
hi = ReLU(wi1 * x1 + wi2 * x2 + bi) for i = [1, 2, 3] (3 neurons)

Output Layer:
output = w01 * h1 + w02 * h2 + w03 * h3 + b0 > 0

## question 4

Q4: Read about this algorithm, which claims to predict “criminality” based on people’s faces, and was created by researchers in China. (If interested, you can click through to the arxiv link where the researchers publish a response to criticism & media about their original paper).

(a) What factors do the researchers claim contribute to “criminality?” (1-3 sentences)
Answer:

- Curvature of the upper lip
- Distance between the eyes
- Angle between the two lines from the tip of the nose to the corners of the mouth

(b) What’s one potential confounding variable/feature that their algorithm learned? What’s your evaluation of the “effectiveness” of this algorithm? (2-4 sentences)
Answer:

Confounding variable could potentially be the ethnicity or or socioeconomic background of the faces in the dataset.
Effectiveness evaluation while having an accuracy of 89.5%, this can be misleading as this might just be good at recognising faces within this particular group or dataset rather than criminality itself.

(c) If this algorithm were actually deployed, what are the consequences of this algorithm making a mistake (misclassification)? (1-3 sentences)
Answer:

If deployed this could be problematic as it could produce false positives, where a innocent is labelled as criminal and false negatives, where a criminal is labelled as innocent.
