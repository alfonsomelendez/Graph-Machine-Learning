# Explanation 5

## Introducing dynamic graphs

Dynamic graphs and temporal GNNs unlock a variety of new applications, such as transport and web
traffic forecasting, motion classification, epidemiological forecasting, link prediction, power system
forecasting, and so on. Time series forecasting is particularly popular with this kind of graph, as we
can use historical data to predict the system’s future behavior.
In this chapter, we focus on graphs with a temporal component. They can be divided into two categories:
• Static graphs with temporal signals: The underlying graph does not change, but features and
labels evolve over time.
• Dynamic graphs with temporal signals: The topology of the graph (the presence of nodes
and edges), features, and labels evolve over time.
In the first case, the graph’s topology is static. For example, it can represent a network of cities within
a country for traffic forecasting: features change over time, but the connections stay the same.
In the second option, nodes and/or connections are dynamic. It is useful to represent a social network
where links between users can appear or disappear over time. This variant is more general, but also
harder to learn how to implement.
we will see how to handle these two types of graphs with temporal signals
using PyTorch Geometric Temporal.

## Forecasting web traffic

we will predict the traffic of Wikipedia articles (as an example of a static graph with a
temporal signal) using a temporal GNN. This regression task has already been covered in Chapter 6,
Introducing Graph Convolutional Networks. However, in that version of the task, we performed traffic
forecasting using a static dataset without a temporal signal: our model did not have any information
about previous instances. This is an issue because it could not understand whether the traffic was
currently increasing or decreasing, for example. We can now improve this model to include information
about past instances.
We will first introduce the temporal GNN architecture with its two variants and then implement it
using PyTorch Geometric Temporal.

## Introducing EvolveGCN

For this task, we will use the EvolveGCN architecture. Introduced by Pareja et al. [1] in 2019, it proposes
a natural combination of GNNs and Recurrent Neural Networks (RNNs). Previous approaches,
such as graph convolutional recurrent networks, applied RNNs with graph convolution operators to
calculate node embeddings. By contrast, EvolveGCN applies RNNs to the GCN parameters themselves.
As the name implies, the GCN evolves over time to produce relevant temporal node embeddings. The
following figure illustrates a high-level view of this process.

![](Figure5-1.PNG)

This architecture has two variants:

1. EvolveGCN-H, where the recurrent neural network considers both the previous GCN parameters
and the current node embeddings.

3. EvolveGCN-O, where the recurrent neural network only considers the previous GCN parameters

## Implementing EvolveGCN


We want to forecast web traffic on a static graph with a temporal signal. The WikiMaths
dataset is comprised of 1,068 articles represented as nodes. Node features correspond to the past daily
number of visits (eight features by default). Edges are weighted, and weights represent the number of
links from the source page to the destination page. We want to predict the daily user visits to these
Wikipedia pages between March 16, 2019, and March 15, 2021, which results in 731 snapshots. Each
snapshot is a graph describing the state of the system at a certain time.

![](Figure5-2.PNG)


PyTorch Geometric does not natively support static or dynamic graphs with a temporal signal.
Fortunately, an extension called PyTorch Geometric Temporal [2] fixes this issue and even implements
various temporal GNN layers. The WikiMaths dataset was also made public during the development
of PyTorch Geometric Temporal. In this chapter, we will use this library to simplify the code and
focus on applications:

1. We need to install this library in an environment containing PyTorch Geometric:
pip install torch-geometric-temporal==0.54.0
2. We import the WikiMaths dataset, called WikiMathDatasetLoader, a temporal-aware
train-test split with temporal_signal_split, and our GNN layer, EvolveGCNH:

![](Figure5-3.PNG)

3. We load the WikiMaths dataset, which is a StaticGraphTemporalSignal object. In
this object, dataset[0] describes the graph (also called a snapshot in this context) at = 0
and dataset[500] at = 500 . We also create a train-test split with a ratio of 0.5. The
training set is composed of snapshots from the earlier time periods, while the test set regroups
snapshots from the later periods:


![](Figure5-4.PNG)




4. The graph is static, so the node and edge dimensions do not change. However, the values contained
in these tensors are different. It is difficult to visualize the values of each of the 1,068 nodes. To
better understand this dataset, we can calculate the mean and standard deviation values for each
snapshot instead. The moving average is also helpful in smoothing out short-term fluctuations.

![](Figure5-5.PNG)




5. We plot these time series with matplotlib to visualize our task:


![](Figure5-6.PNG)
![](Figure5-65.PNG)



This produces the following figure:

![](Figure5-7.PNG)


Our data presents periodic patterns that the temporal GNN can hopefully learn. We can now
implement it and see how it performs.
6. The temporal GNN takes two parameters as inputs: the number of nodes (node_count) and
the input dimension (dim_in). The GNN only has two layers: an EvolveGCN-H layer and a
linear layer that outputs a predicted value for each node:

![](Figure5-8.PNG)


7. The forward() function applies both layers to the input with a ReLU activation function:

![](Figure5-9.PNG)


8. We create an instance of TemporalGNN and give it the number of nodes and input dimension
9.  from the WikiMaths dataset. We will train it using the Adam optimizer:


![](Figure5-10.PNG)


9. We can print the model to observe the layers contained in EvolveGCNH:

![](Figure5-11.PNG)

We see three layers: TopKPooling, which summarizes the input matrix in eight columns;
GRU, which updates the GCN weight matrix; and GCNConv, which produces the new node
embedding. Finally, a linear layer outputs a predicted value for every node in the graph.

10. We create a training loop that trains the model on every snapshot from the training set. The
loss is backpropagated for every snapshot:

![](Figure5-12.PNG)

11. Likewise, we evaluate the model on the test set. The MSE is averaged on the entire test set to
produce the final score:

![](Figure5-13.PNG)


12. We obtain a loss value of 0.7559. Next, we will plot the mean values predicted by our model on
the previous graph to interpret it. The process is straightforward: we must average the predictions
and store them in a list. Then, we can add them to the previous plot:

![](Figure5-14.PNG)

We obtain the following:

![](Figure5-15.PNG)

We can see that the predicted values follow the general trend in the data. This is an excellent
result, considering the limited size of the dataset.

13. Finally, let’s create a scatter plot to show how predicted and ground truth values differ for a
single snapshot:

![](Figure5-16.PNG)

![](Figure5-17.PNG)

We observe a moderate positive correlation between predicted and real values. Our model is not
remarkably accurate, but the previous figure showed that it understands the periodic nature of the
data very well.





