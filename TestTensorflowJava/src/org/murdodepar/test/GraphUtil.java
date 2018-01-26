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
import org.tensorflow.Shape;
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
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc/gradients

REGISTER_NO_GRADIENT_OP("Const");
REGISTER_NO_GRADIENT_OP("StopGradient");
REGISTER_NO_GRADIENT_OP("ConcatOffset");
REGISTER_NO_GRADIENT_OP("EditDistance");
REGISTER_NO_GRADIENT_OP("ZerosLike");
REGISTER_NO_GRADIENT_OP("InvertPermutation");
REGISTER_NO_GRADIENT_OP("Shape");
REGISTER_NO_GRADIENT_OP("ShapeN");
REGISTER_NO_GRADIENT_OP("Rank");
REGISTER_NO_GRADIENT_OP("Size");
REGISTER_NO_GRADIENT_OP("BroadcastGradientArgs");
REGISTER_NO_GRADIENT_OP("OneHot");

REGISTER_GRADIENT_OP("Pack", PackGrad);
REGISTER_GRADIENT_OP("Unpack", UnpackGrad);
REGISTER_GRADIENT_OP("Identity", IdentityGrad);
REGISTER_GRADIENT_OP("RefIdentity", RefIdentityGrad);
REGISTER_GRADIENT_OP("QuantizeAndDequantize", QuantizeAndDequantizeGrad);
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV2", QuantizeAndDequantizeV2Grad);
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV3", QuantizeAndDequantizeV3Grad);
REGISTER_GRADIENT_OP("Split", SplitGrad);
REGISTER_GRADIENT_OP("Diag", DiagGrad);
REGISTER_GRADIENT_OP("DiagPart", DiagPartGrad);
REGISTER_GRADIENT_OP("MatrixDiag", MatrixDiagGrad);
REGISTER_GRADIENT_OP("MatrixBandPart", MatrixBandPartGrad);
REGISTER_GRADIENT_OP("GatherNd", GatherNdGrad);
REGISTER_GRADIENT_OP("CheckNumerics", CheckNumericsGrad);
REGISTER_GRADIENT_OP("Reshape", ReshapeGrad);
REGISTER_GRADIENT_OP("ExpandDims", ExpandDimsGrad);
REGISTER_GRADIENT_OP("Squeeze", SqueezeGrad)
REGISTER_GRADIENT_OP("Transpose", TransposeGrad);
REGISTER_GRADIENT_OP("ReverseSequence", ReverseSequenceGrad);
REGISTER_GRADIENT_OP("ReverseV2", ReverseGrad);
REGISTER_GRADIENT_OP("ScatterNd", ScatterNdGrad);
REGISTER_GRADIENT_OP("ScatterNdNonAliasingAdd", ScatterNdNonAliasingAddGrad);
REGISTER_GRADIENT_OP("Pad", PadGrad<false>);
REGISTER_GRADIENT_OP("PadV2", PadGrad<true>);
REGISTER_GRADIENT_OP("SpaceToBatch", SpaceToBatchGrad);
REGISTER_GRADIENT_OP("SpaceToBatchND", SpaceToBatchNDGrad);
REGISTER_GRADIENT_OP("BatchToSpace", BatchToSpaceGrad);
REGISTER_GRADIENT_OP("BatchToSpaceND", BatchToSpaceNDGrad);
REGISTER_GRADIENT_OP("SpaceToDepth", SpaceToDepthGrad);
REGISTER_GRADIENT_OP("DepthToSpace", DepthToSpaceGrad);
REGISTER_GRADIENT_OP("MirrorPad", MirrorPadGrad);
REGISTER_GRADIENT_OP("MirrorPadGrad", MirrorPadGradGrad);
REGISTER_GRADIENT_OP("DynamicPartition", DynamicPartitionGrad);
REGISTER_GRADIENT_OP("DynamicStitch", DynamicStitchGrad);
REGISTER_GRADIENT_OP("ParallelDynamicStitch", DynamicStitchGrad);

REGISTER_NO_GRADIENT_OP("Queue");
REGISTER_NO_GRADIENT_OP("QueueEnqueue");
REGISTER_NO_GRADIENT_OP("QueueEnqueueMany");
REGISTER_NO_GRADIENT_OP("QueueDequeue");
REGISTER_NO_GRADIENT_OP("QueueDequeueMany");
REGISTER_NO_GRADIENT_OP("QueueDequeueUpTo");
REGISTER_NO_GRADIENT_OP("QueueClose");
REGISTER_NO_GRADIENT_OP("QueueSize");
REGISTER_NO_GRADIENT_OP("Stack");
REGISTER_NO_GRADIENT_OP("StackPush");
REGISTER_NO_GRADIENT_OP("StackPop");
REGISTER_NO_GRADIENT_OP("StackClose");
REGISTER_NO_GRADIENT_OP("GetSessionHandle");
REGISTER_NO_GRADIENT_OP("GetSessionHandleV2");
REGISTER_NO_GRADIENT_OP("GetSessionTensor");
REGISTER_NO_GRADIENT_OP("DeleteSessionTensor");

// Logical operations have no gradients.
REGISTER_NO_GRADIENT_OP("Less");
REGISTER_NO_GRADIENT_OP("LessEqual");
REGISTER_NO_GRADIENT_OP("Greater");
REGISTER_NO_GRADIENT_OP("GreaterEqual");
REGISTER_NO_GRADIENT_OP("Equal");
REGISTER_NO_GRADIENT_OP("ApproximateEqual");
REGISTER_NO_GRADIENT_OP("NotEqual");
REGISTER_NO_GRADIENT_OP("LogicalAnd");
REGISTER_NO_GRADIENT_OP("LogicalOr");
REGISTER_NO_GRADIENT_OP("LogicalNot");
REGISTER_GRADIENT_OP("Abs", AbsGrad);
REGISTER_GRADIENT_OP("Neg", NegGrad);
REGISTER_GRADIENT_OP("Inv", InvGrad);
REGISTER_GRADIENT_OP("Reciprocal", InvGrad);
REGISTER_GRADIENT_OP("Square", SquareGrad);
REGISTER_GRADIENT_OP("Sqrt", SqrtGrad);
REGISTER_GRADIENT_OP("Rsqrt", RsqrtGrad);
REGISTER_GRADIENT_OP("Exp", ExpGrad);
REGISTER_GRADIENT_OP("Expm1", Expm1Grad);
REGISTER_GRADIENT_OP("Log", LogGrad);
REGISTER_GRADIENT_OP("Log1p", Log1pGrad);
REGISTER_GRADIENT_OP("Sinh", SinhGrad);
REGISTER_GRADIENT_OP("Cosh", CoshGrad);
REGISTER_GRADIENT_OP("Tanh", TanhGrad);
REGISTER_GRADIENT_OP("Asinh", AsinhGrad);
REGISTER_GRADIENT_OP("Acosh", AcoshGrad);
REGISTER_GRADIENT_OP("Atanh", AtanhGrad);
REGISTER_GRADIENT_OP("Sigmoid", SigmoidGrad);
REGISTER_GRADIENT_OP("Sign", SignGrad);
REGISTER_GRADIENT_OP("Sin", SinGrad);
REGISTER_GRADIENT_OP("Cos", CosGrad);
REGISTER_GRADIENT_OP("Asin", AsinGrad);
REGISTER_GRADIENT_OP("Acos", AcosGrad);
REGISTER_GRADIENT_OP("Tan", TanGrad);
REGISTER_GRADIENT_OP("Atan", AtanGrad);
REGISTER_GRADIENT_OP("Add", AddGrad);
REGISTER_GRADIENT_OP("Sub", SubGrad);
REGISTER_GRADIENT_OP("Mul", MulGrad);
REGISTER_GRADIENT_OP("Div", DivGrad);
REGISTER_GRADIENT_OP("RealDiv", RealDivGrad);
REGISTER_GRADIENT_OP("SquaredDifference", SquaredDifferenceGrad);
REGISTER_GRADIENT_OP("AddN", AddNGrad);
REGISTER_GRADIENT_OP("Pow", PowGrad);
REGISTER_GRADIENT_OP("Maximum", MaximumGrad);
REGISTER_GRADIENT_OP("Minimum", MinimumGrad);
REGISTER_GRADIENT_OP("Real", RealGrad);
REGISTER_GRADIENT_OP("Imag", ImagGrad);
REGISTER_GRADIENT_OP("Complex", ComplexGrad);
REGISTER_GRADIENT_OP("Angle", AngleGrad);
REGISTER_GRADIENT_OP("Conj", ConjGrad);
REGISTER_GRADIENT_OP("Sum", SumGrad);
REGISTER_GRADIENT_OP("Mean", MeanGrad);
REGISTER_GRADIENT_OP("Erf", ErfGrad);
REGISTER_GRADIENT_OP("Lgamma", LgammaGrad);
REGISTER_GRADIENT_OP("Min", MinOrMaxGrad);
REGISTER_GRADIENT_OP("Max", MinOrMaxGrad);
REGISTER_GRADIENT_OP("Prod", ProdGrad);
REGISTER_GRADIENT_OP("MatMul", MatMulGrad);
REGISTER_GRADIENT_OP("BatchMatMul", BatchMatMulGrad);

REGISTER_GRADIENT_OP("Softmax", SoftmaxGrad);
REGISTER_GRADIENT_OP("LogSoftmax", LogSoftmaxGrad);
REGISTER_GRADIENT_OP("Relu", ReluGradHelper);
REGISTER_GRADIENT_OP("Relu6", Relu6GradHelper);
REGISTER_GRADIENT_OP("Elu", EluGradHelper);
REGISTER_GRADIENT_OP("Selu", SeluGradHelper);
REGISTER_GRADIENT_OP("L2Loss", L2LossGrad);
REGISTER_GRADIENT_OP("BiasAdd", BiasAddGradHelper);
REGISTER_GRADIENT_OP("Conv2D", Conv2DGrad);
REGISTER_GRADIENT_OP("MaxPool", MaxPoolGradHelper);
REGISTER_GRADIENT_OP("MaxPoolV2", MaxPoolGradV2Helper);
REGISTER_GRADIENT_OP("LRN", LRNGradHelper)

 */
public class GraphUtil implements AutoCloseable{
	private  Graph g;
	private  Session s;
	public GraphUtil(){
		g = new Graph();
		s = new Session(g);
	}
	public GraphUtil(Graph g){
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
	public Output<OperationBuilder> setConst(String name, DataType type, Object obj){
		Tensor<?> t = Tensor.create(obj);
		return g.opBuilder("Const", name)
				.setAttr("dtype", type).setAttr("value", t)
				.build().output(0);
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
	/*TODO : KO  a revoir*/
	public Output<OperationBuilder> setVariable(String name, DataType type, Object obj){
		Tensor<?> t = Tensor.create(obj);
		return g.opBuilder("Variable", name)
				.setAttr("dtype", type)
				.setAttr("shape", t.shape())
				.build().output(0);
/*		
NodeDef mentions attr 'value' not in Op
<name=Variable; signature= -> ref:Ref(dtype); 
attr=shape:shape; attr=dtype:type; attr=container:string,default=""; 
attr=shared_name:string,default=""; is_stateful=true>; 
NodeDef: V1 = Variable[container="", dtype=DT_INT32, shared_name="", 
value=Tensor<type: int32 shape: [] values: 3>](). 

java.lang.IllegalArgumentException: AttrValue missing value with expected type 'shape'
	 for attr 'shape'; 
NodeDef: V1 = Variable[container="", dtype=DT_INT32, shape=[], shared_name=""](); 
Op<name=Variable; signature= -> 
ref:Ref(dtype); attr=shape:shape; attr=dtype:type; attr=container:string,default=""; 
attr=shared_name:string,default=""; is_stateful=true>
*/	
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
        .build().output(0);
/*    	
<name=Add; 
signature=x:T, y:T -> z:T; 
attr=T:type,
allowed=[DT_HALF, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128, DT_STRING]>; 
*/
	}
	public Output<OperationBuilder> setSub(String name, Output<OperationBuilder> var1, Output<OperationBuilder> var2) {
    	return g.opBuilder("Sub", name).addInput(var1).addInput(var2)
        .build().output(0);
	}
	public Output<OperationBuilder> setMul(String name, Output<OperationBuilder> var1, Output<OperationBuilder> var2) {
    	return g.opBuilder("Mul", name).addInput(var1).addInput(var2)
        .build().output(0);
	}
	public Output<OperationBuilder> setSquare(String name, Output<OperationBuilder> var1) {
    	return g.opBuilder("Square", name).addInput(var1)
        .build().output(0);
	}
	public Output<OperationBuilder> layers_dense(String name, Output input) {
	    try (Tensor t = Tensor.create(new int[] {0})) {
	        Output reductionIndices = g.opBuilder("Const", "layer")
	                .setAttr("dtype", t.dataType()).setAttr("value", t)
	                .build().output(0);
	        Output ret = g.opBuilder("layersdense", name).setAttr("T", DataType.FLOAT)
	            .setAttr("Tidx", DataType.INT32)
	            .addInput(input).addInput(reductionIndices)
	            .build().output(0);
	        return ret;
	      }
	}
	public Output<OperationBuilder> identity(Output input, String name) {
		Tensor<?> t = Tensor.create(input);
		return g.opBuilder("Identity", name).setAttr("value", t)
				.build().output(0);
	}
	/*
	 * https://github.com/tensorflow/tensorflow/issues/8280
	 */
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
	public Output GradientDescentOptimizer(String name, Float learning_rate) {
	    try (Tensor t = Tensor.create(learning_rate)) {
	        Output lr = g.opBuilder("Const", "learning_rate")
	                .setAttr("dtype", t.dataType()).setAttr("value", t)
	                .build().output(0);
	        Output ret = g.opBuilder("GradientDescentOptimizer", name)
	        	.setAttr("LR", DataType.FLOAT)
	            .addInput(lr)
	            .build().output(0);
	        return ret;
	      }
	}
}
