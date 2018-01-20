package org.murdodepar.test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/*
 * 
https://blog.zenika.com/2017/11/14/tensorflow-le-nouveau-framework-pour-democratiser-le-deep-learning-13/#prettyPhoto

Placeholder, Const, Abort, RandomUniform, MaxPool, FractionalMaxPool

Operation groupes 	Operations
Maths				Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
Array				Concat, Slide, Split, Constant, Rank, Shape, Shuffle
Matrix				MatMul, MatrixInverse, MatrixDeterminant
Neuronal Network	SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool

Optimiseurs
GradientDescentOptimizer
AdagradOptimizer
AdamOptimizer
...

 */
public class GraphUtil implements AutoCloseable{
	private  Graph g;
	private  Session s;
	GraphUtil(){
		g = new Graph();
		s = new Session(g);
	}
	GraphUtil(Graph g){
		this.g = g;
	}
	public void close() {
		s.close();
		g.close();
	}
	public Graph getGraph(){
		return g;
	}
	public Session getSession() {
		return s;
	}
	public void saveGraphDef(String paths) {
	    try {
			Files.write(Paths.get(paths), g.toGraphDef(), StandardOpenOption.CREATE);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public void loadGraphDef(String paths) {
		SavedModelBundle load = SavedModelBundle.load(paths, "serve");
		s = load.session();
		g = load.graph();
		load.close();
	}
	public Output<OperationBuilder> setPlaceholder(String name, DataType type){
		return g.opBuilder("Placeholder", name)
				.setAttr("dtype", type)
				.build().output(0);
	}
	public Output<OperationBuilder> setPlaceholder(String name, Class<?> type){
		return g.opBuilder("Placeholder", name)
				.setAttr("dtype", DataType.fromClass(type))
				.build().output(0);
	}
	public Output<OperationBuilder> setVariable(String name, DataType type, Object obj){
		Tensor<?> t = Tensor.create(obj);
		return g.opBuilder("Variable", name)
				.setAttr("dtype", type).setAttr("value", t)
				.build().output(0);
	}
	public Output<OperationBuilder> setConst(String name, DataType type, Object obj){
		Tensor<?> t = Tensor.create(obj);
		return g.opBuilder("Const", name)
				.setAttr("dtype", type).setAttr("value", t)
				.build().output(0);
	}
	public Output<OperationBuilder> setMatMul(String name, Output<OperationBuilder> var1, Output<OperationBuilder> var2) {
    	return g.opBuilder("MatMul", name).addInput(var1).addInput(var2)
        .setAttr("transpose_a", true).setAttr("transpose_b", false)
        .build().output(0);
    	
    	/* https://github.com/tensorflow/tensorflow/issues/7149
    	 * GraphBuider.matMul(a, b).withTransposeB(true).softmax(logits)
    	 */
	}
	public Output<OperationBuilder> setAdd(String name, Output<OperationBuilder> var1, Output<OperationBuilder> var2) {
    	return g.opBuilder("Add", name).addInput(var1).addInput(var2)
        .setAttr("transpose_a", true).setAttr("transpose_b", false)
        .build().output(0);
	}
	public Output<OperationBuilder> setSub(String name, Output<OperationBuilder> var1, Output<OperationBuilder> var2) {
    	return g.opBuilder("Sub", name).addInput(var1).addInput(var2)
        .setAttr("transpose_a", true).setAttr("transpose_b", false)
        .build().output(0);
	}
	public void setMul(String name, Output<OperationBuilder> var1, Output<OperationBuilder> var2) {
    	g.opBuilder("Mul", name).addInput(var1).addInput(var2)
        .setAttr("transpose_a", true).setAttr("transpose_b", false)
        .build().output(0);
	}
	public Output<OperationBuilder> identity(Output input, String name) {
		Tensor<?> t = Tensor.create(input);
		return g.opBuilder("identity", name).setAttr("value", t)
				.build().output(0);
	}
	public Output<OperationBuilder> square(Output input, String name) {
		Tensor<?> t = Tensor.create(input);
		return g.opBuilder("square", name).setAttr("value", t)
				.build().output(0);
	}
	public Output reduceMean(String name, Output input) {
	    try (Tensor t = Tensor.create(new int[] {0})) {
	        Output reductionIndices = g.opBuilder("Const", "ReductionIndices")
	                .setAttr("dtype", t.dataType()).setAttr("value", t)
	                .build().output(0);
	        Output ret = g.opBuilder("Mean", name).setAttr("T", DataType.FLOAT)
	            .setAttr("Tidx", DataType.INT32)
	            .addInput(input).addInput(reductionIndices)
	            .build().output(0);
	        return ret;
	      }
	}
}
