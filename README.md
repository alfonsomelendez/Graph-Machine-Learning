# Explanation 1


we show an example using a particular embedding algorithm known as Node to Vector (Node2Vec).

![](Figure1.PNG)

In the preceding code, we have done the following:
1. We generated a barbell graph (described in the previous chapter).
2. The Node2Vec embedding algorithm is then used in order to map each node of the
graph in a vector of two dimensions.
3. Finally, the two-dimensional vectors generated by the embedding algorithm,
representing the nodes of the original graph, are plotted.

The result is shown in the following figure:

![](Figure1-2.PNG)

it is easy to see that nodes that have a similar structure are close to each
other and are distant from nodes that have dissimilar structures. It is also interesting
to observe how good Node2Vec is at discriminating group 1 from group 3. Since the
algorithm uses neighboring information of each node to generate the representation, the
clear discrimination of those two groups is possible.

Another example on the same graph can be performed using the Edge to Vector
(Edge2Vec) algorithm in order to generate a mapping for the edges for the same graph, G:


![](Figure1-3.PNG)



In the preceding code, we have done the following:
1. We generated a barbell graph (described in the previous chapter).
2. The HadamardEmbedder embedding algorithm is applied to the result of the
Node2Vec algorithm (keyed_vectors=model.wv) used in order to map each
edge of the graph in a vector of two dimensions.
3. Finally, the two-dimensional vectors generated by the embedding algorithm,
representing the nodes of the original graph, are plotted.
The results are shown in the following figure:

![](Figure1-4.PNG)

From the figure, it is easy to see that the edge embedding algorithm clearly
identifies similar edges. As expected, edges belonging to groups 1, 2, and 3 are clustered in
well-defined and well-grouped regions. Moreover, the (6,7) and (10,11) edges, belonging
to groups 4 and 5, respectively, are well clustered in specific groups.



Finally, we will provide an example of a Graph to Vector (Grap2Vec) embedding
algorithm. This algorithm maps a single graph in a vector. As for another example, we
will discuss this algorithm in more detail in the next chapter. In the following code block,
we provide a Python example showing how to use the Graph2Vec algorithm in order to
generate the embedding representation on a set of graphs:



![](Figure1-5.PNG)




In this example, the following has been done:
1. 20 Watts-Strogatz graphs (described in the previous chapter) have been generated
with random parameters.
2. We have then executed the graph embedding algorithm in order to generate a
two-dimensional vector representation of each graph.
3. Finally, the generated vectors are plotted in their Euclidean space.
The results of this example are the following:




![](Figure1-6.PNG)





graphs with a large Euclidean distance, such as graph 12
and graph 8, have a different structure. The former is generated with the nx.watts_
strogatz_graph(20,20,0.2857) parameter and the latter with the nx.watts_
strogatz_graph(13,6,0.8621) parameter. In contrast, a graph with a low
Euclidean distance, such as graph 14 and graph 8, has a similar structure. Graph 14 is
generated with the nx.watts_strogatz_graph(9,9,0.5091) command, while
graph 4 is generated with nx.watts_strogatz_graph(10,5,0.5659).




## Páginas Web

[GML an Overview](https://towardsdatascience.com/graph-machine-learning-an-overview-c996e53fab90)<br>

[Overview Hugging Face](https://huggingface.co/blog/intro-graphml)<br>

[Graphs Everywhere !!Para Mirar!!](https://engineering.rappi.com/graphs-everywhere-an-introduction-to-graph-ml-f0a3d5893cb8)

[A Gentle Introduction to graph neural networks](https://distill.pub/2021/gnn-intro/)<br>

[How to get started with GML](https://gordicaleksa.medium.com/how-to-get-started-with-graph-machine-learning-afa53f6f963a)<br>

[AI Trends in 2023: Graph Neural Networks](https://www.assemblyai.com/blog/ai-trends-graph-neural-networks/)

[Graph Neural Networks (GNNs) Study Guide GITHUB](https://github.com/dair-ai/GNNs-Recipe)<br>

[awesome compilation, to go further with research](https://github.com/GRAND-Lab/Awesome-Graph-Neural-Networks)<br>
[Demystifying Graph based Machine Learning](https://medium.com/mlearning-ai/demystifying-graph-based-machine-learning-ed6b6b7c4081)<br>

[Neptune Graph neural networks](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications)

## Cursos en Línea

[Mandarin 2023](https://www.youtube.com/playlist?list=PLH-G21fz8AE1XgwfH5YPMealh6td1LMuh)<br>


[Curso Completo  Graph neural Networks DeepFindR](https://www.youtube.com/watch?v=fOctJB4kVlM&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z)

[Udemy Graph neural Networks](https://www.udemy.com/course/graph-neural-network/)<br>

[mlabonne Github Course Graph neural networks](https://github.com/mlabonne/graph-neural-network-course)<br>

[Geometric Deep Learning Michael Bronstein ](https://www.youtube.com/playlist?list=PLn2-dEmQeTfSLXW8yXP4q_Ii58wFdxb3C)<br>

[Penn University Machine learning onGraphs](https://www.youtube.com/watch?v=90lWiGEHTn4&list=PL-BLJBpGQyLOBRhqEry2rsibv14hH5A2u)<br>


[Machine leaning with Graphs 2024 Stanford](https://online.stanford.edu/courses/xcs224w-machine-learning-graphs)<br>

[Jure Leskovec](https://www.youtube.com/watch?v=JAB_plj2rbA&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=2&t=2s)<br>

[Mcgills Graph Representation Learning](https://cs.mcgill.ca/~wlh/comp766/)<br>



## Videos


[ Hamilton Representation Learning](https://www.youtube.com/watch?v=fbRDfhNrCwo&t=2197s)<br>

[Excelente Video GNN Divulgativo](https://www.youtube.com/watch?v=GXhBEj1ZtE8)<br>

[Int to graph machine learning Oracle](https://www.youtube.com/watch?v=ZdDwN1cUEck)<br>

GNN for beginners:

[Video 1](https://www.youtube.com/watch?v=YdGN-J322y4)
[Video 2](https://www.youtube.com/watch?v=VDzrvhgyxsU&t=3145s)


[Neural Nine Simple Embedding karate Club](https://www.youtube.com/watch?v=uszt88Z-0Fc&t=22s)<br>

[Everything is connected Deep leraning on Graphs peter Lekovic](https://www.youtube.com/watch?v=5h6MbQ_65-o&t=1655s)<br>


[Petar Veličković's GNN video]( https://youtu.be/8owQBFAHw7E)<br>

[Michael Bronstein's Geometric Deep Learning keynote speech (beautiful!)](https://youtu.be/w6Pw4MOzMuo)<br>

[Xavier Bresson's Graph Convolutional Networks lecture](https://youtu.be/Iiv9R6BjxH)<br>

[3Blue1Brown’s series on Neural Networks](https://youtu.be/aircAruvnKk)<br>




## Blogs<br>

[Petar resources](https://goo.gle/3cO7gvb)<br>



[Michael Bronstein](https://towardsdatascience.com/geomet...) (a must-read)<br>

[aa](https://towardsdatascience.com/do-we-...)<br>

[Amal Menzli](https://neptune.ai/blog/graph-neural-...)<br>

[Eric J. Ma](https://ericmjl.github.io/essays-on-d... )<br>

[Rishabh Anand](https://medium.com/dair-ai/an-illustr...)<br>

[Distill 1](https://distill.pub/2021/gnn-intro/)<br>

[Distill 2](https://distill.pub/2021/understandin...)<br>


## Adicional Temporal Graph Networks

### Ejemplos curso GNN DeepFindR 

[Ejemplo temporal network Traffic PyTorch](https://www.youtube.com/watch?v=Rws9mf1aWUs&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=19)<br>

[Ejemplo temporal network Fraud Detection](https://www.youtube.com/watch?v=MZGuz-o7Fl0&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=20)<br>

[Ejemplo fake news detection](https://www.youtube.com/watch?v=QAIVFr24FrA&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=21)<br>

[Ejemplo Recommender Systems](https://www.youtube.com/watch?v=NyNqzDKcKG4&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=22)<br>


[Emmanuel Rossi Bronstein Blog Deep learning in Dynamic graphs](https://blog.twitter.com/engineering/en_us/topics/insights/2021/temporal-graph-networks)<br>

[TGN: Temporal graph networks (Libreria)](https://github.com/twitter-research/tgn)<br>
[Inductive representation Learning in Temporal Graphs](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)<br>
[Temporal networks in Gephi](https://www.youtube.com/watch?v=W6RzekieOgM&list=PLwbiwzlYiabrLw9zkfs55oD8J-rp-0IUu)

